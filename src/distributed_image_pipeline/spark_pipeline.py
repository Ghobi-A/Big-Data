"""Spark preprocessing pipeline.

Distributes the *same* Pillow-based preprocessing and TFRecord schema as the
local baseline across Spark partitions. Each partition writes one TFRecord
shard directly from the executor, so no image data is collected to the driver.

Design notes relative to the original coursework implementation:

- File paths are parallelised straight into ``partitions`` slices; there is no
  extra ``repartition`` step (the coursework repartitioned after mapping,
  shuffling full image tensors across the cluster).
- Sampling happens deterministically on the driver (sorted files + seeded
  ``random.Random``), not via ``RDD.sample``, so a given (rate, seed) always
  selects the same files in both execution modes.
- Unknown labels raise instead of silently becoming class 0.

TensorFlow must be importable on executors (TFRecordWriter); on Dataproc use a
pip initialisation action or a custom image.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from .config import PipelineConfig
from .labels import DEFAULT_CLASSES, label_from_path, validate_classes
from .local_pipeline import list_image_files, sample_files
from .results import PipelineRunResult

logger = logging.getLogger(__name__)


def _read_file_bytes(path: str) -> bytes:
    if path.startswith("gs://"):
        import tensorflow as tf

        return tf.io.gfile.GFile(path, "rb").read()
    with open(path, "rb") as f:
        return f.read()


def _process_partition(
    index: int,
    paths,
    output_uri: str,
    classes: Sequence[bytes],
    target_height: int,
    target_width: int,
    error_policy: str,
):
    """Executor-side: preprocess a partition's files and write one shard.

    Yields a single stats dict for the partition.
    """
    from .preprocessing import PreprocessingError, ProcessingStats, preprocess_image_bytes
    from .tfrecords import write_tfrecord_shard

    paths = list(paths)
    stats = ProcessingStats()
    examples = []
    for path in paths:
        try:
            data = _read_file_bytes(path)
            img = preprocess_image_bytes(data, target_height, target_width)
            examples.append((img, label_from_path(path), target_height, target_width, path))
            stats.processed += 1
        except (PreprocessingError, OSError, ValueError) as exc:
            stats.record_failure(path, exc, error_policy)

    shard_path = None
    if examples:
        sep = "" if output_uri.endswith("/") else "/"
        shard_path = f"{output_uri}{sep}flowers-{index:03d}.tfrec"
        write_tfrecord_shard(shard_path, examples, classes)

    yield {
        "index": index,
        "processed": stats.processed,
        "failed": stats.failed,
        "skipped": stats.skipped,
        "shard_path": shard_path,
    }


def run_spark_pipeline(
    config: PipelineConfig,
    classes=DEFAULT_CLASSES,
    spark_context=None,
) -> PipelineRunResult:
    """Run the distributed preprocessing pipeline and return run metadata."""
    import pyspark

    valid_classes = validate_classes(classes)
    sc = spark_context or pyspark.SparkContext.getOrCreate()

    all_files = list_image_files(config.input_uri)
    if not all_files:
        raise FileNotFoundError(f"no input images found under {config.input_uri!r}")
    files = sample_files(all_files, config.sample_rate, config.seed)
    if config.is_sampled:
        logger.info(
            "SAMPLED RUN: %d of %d files (rate=%.3f, seed=%d) — not a full-dataset result",
            len(files), len(all_files), config.sample_rate, config.seed,
        )

    if not config.output_uri.startswith("gs://"):
        from pathlib import Path

        Path(config.output_uri).mkdir(parents=True, exist_ok=True)

    output_uri = config.output_uri
    th, tw = config.target_height, config.target_width
    policy = config.error_policy

    start = time.perf_counter()
    rdd = sc.parallelize(files, config.partitions)
    partition_stats = rdd.mapPartitionsWithIndex(
        lambda idx, it: _process_partition(idx, it, output_uri, valid_classes, th, tw, policy)
    ).collect()
    runtime = time.perf_counter() - start

    processed = sum(s["processed"] for s in partition_stats)
    failed = sum(s["failed"] for s in partition_stats)
    skipped = sum(s["skipped"] for s in partition_stats)
    shards = [s["shard_path"] for s in partition_stats if s["shard_path"]]

    size_mb = None
    if not output_uri.startswith("gs://"):
        from pathlib import Path

        size_mb = round(
            sum(Path(p).stat().st_size for p in shards if Path(p).exists()) / 1e6, 3
        )

    result = PipelineRunResult(
        mode="spark",
        input_files=len(files),
        processed_files=processed,
        failed_files=failed,
        skipped_files=skipped,
        partitions=config.partitions,
        runtime_seconds=runtime,
        throughput_images_per_second=processed / runtime if runtime > 0 else 0.0,
        output_files=len(shards),
        output_size_mb=size_mb,
        sample_rate=config.sample_rate,
        seed=config.seed,
        sampled=config.is_sampled,
        output_paths=shards,
    )
    logger.info(
        "spark pipeline: %d processed, %d failed, %d skipped, %d partitions, "
        "%d shards in %.2fs (%.1f img/s)",
        processed, failed, skipped, config.partitions, len(shards),
        runtime, result.throughput_images_per_second,
    )
    return result
