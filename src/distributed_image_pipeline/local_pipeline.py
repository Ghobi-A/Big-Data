"""Local (non-Spark) preprocessing baseline.

Uses exactly the same preprocessing function, label mapping and TFRecord output
schema as the Spark pipeline so throughput comparisons are like-for-like.
"""

from __future__ import annotations

import glob
import logging
import random
import time
from pathlib import Path

from .config import PipelineConfig
from .labels import DEFAULT_CLASSES, label_from_path, validate_classes
from .preprocessing import PreprocessingError, ProcessingStats, preprocess_file
from .results import PipelineRunResult

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg")


def list_image_files(input_uri: str) -> list[str]:
    """List input images from a local directory, local glob, or gs:// glob.

    Results are sorted so downstream sampling and sharding are deterministic.
    """
    if input_uri.startswith("gs://"):
        import tensorflow as tf

        files = tf.io.gfile.glob(input_uri)
    else:
        p = Path(input_uri)
        if p.is_dir():
            files = [
                str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
            ]
        else:
            files = glob.glob(input_uri, recursive=True)
    return sorted(files)


def sample_files(files: list[str], sample_rate: float, seed: int) -> list[str]:
    """Deterministically sample a sorted file list. Full list when rate == 1."""
    if sample_rate >= 1.0:
        return files
    k = max(1, round(len(files) * sample_rate))
    rng = random.Random(seed)
    return sorted(rng.sample(files, k))


def _chunk(files: list[str], n_chunks: int) -> list[list[str]]:
    n_chunks = min(n_chunks, max(1, len(files)))
    size = -(-len(files) // n_chunks)  # ceil division
    return [files[i : i + size] for i in range(0, len(files), size)]


def run_local_pipeline(
    config: PipelineConfig,
    classes=DEFAULT_CLASSES,
    output_format: str = "tfrecord",
) -> PipelineRunResult:
    """Run the single-process local baseline.

    ``output_format`` is ``"tfrecord"`` (default; requires TensorFlow) or
    ``"jpeg"`` (writes processed JPEGs mirroring the class layout; no TF).
    """
    if output_format not in ("tfrecord", "jpeg"):
        raise ValueError(f"unknown output_format {output_format!r}")
    valid_classes = validate_classes(classes)

    all_files = list_image_files(config.input_uri)
    if not all_files:
        raise FileNotFoundError(f"no input images found under {config.input_uri!r}")
    files = sample_files(all_files, config.sample_rate, config.seed)
    if config.is_sampled:
        logger.info(
            "SAMPLED RUN: %d of %d files (rate=%.3f, seed=%d) — not a full-dataset result",
            len(files), len(all_files), config.sample_rate, config.seed,
        )

    out_dir = Path(config.output_uri)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = ProcessingStats()
    output_paths: list[str] = []
    start = time.perf_counter()

    if output_format == "tfrecord":
        from .tfrecords import write_tfrecord_shard

        shards = _chunk(files, config.partitions)
        for idx, shard_files in enumerate(shards):
            examples = []
            for path in shard_files:
                try:
                    img = preprocess_file(path, config.target_height, config.target_width)
                    examples.append(
                        (img, label_from_path(path), config.target_height,
                         config.target_width, path)
                    )
                    stats.processed += 1
                except (PreprocessingError, OSError, ValueError) as exc:
                    stats.record_failure(path, exc, config.error_policy)
            shard_path = str(out_dir / f"flowers-{idx:03d}.tfrec")
            write_tfrecord_shard(shard_path, examples, valid_classes)
            output_paths.append(shard_path)
    else:
        for path in files:
            try:
                img = preprocess_file(path, config.target_height, config.target_width)
                label = label_from_path(path)
                dest = out_dir / label
                dest.mkdir(parents=True, exist_ok=True)
                out_path = dest / Path(path).name
                out_path.write_bytes(img)
                output_paths.append(str(out_path))
                stats.processed += 1
            except (PreprocessingError, OSError, ValueError) as exc:
                stats.record_failure(path, exc, config.error_policy)

    runtime = time.perf_counter() - start
    size_mb = sum(Path(p).stat().st_size for p in output_paths if Path(p).exists()) / 1e6
    result = PipelineRunResult(
        mode="local",
        input_files=len(files),
        processed_files=stats.processed,
        failed_files=stats.failed,
        skipped_files=stats.skipped,
        partitions=config.partitions,
        runtime_seconds=runtime,
        throughput_images_per_second=stats.processed / runtime if runtime > 0 else 0.0,
        output_files=len(output_paths),
        output_size_mb=round(size_mb, 3),
        sample_rate=config.sample_rate,
        seed=config.seed,
        sampled=config.is_sampled,
        output_paths=output_paths,
    )
    logger.info(
        "local pipeline: %d processed, %d failed, %d skipped in %.2fs (%.1f img/s)",
        result.processed_files, result.failed_files, result.skipped_files,
        result.runtime_seconds, result.throughput_images_per_second,
    )
    return result
