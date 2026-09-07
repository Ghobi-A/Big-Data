"""Benchmark orchestration: repeated preprocessing runs and I/O benchmarks.

Every run is recorded as one CSV row. Fields that cannot be measured in the
current environment (e.g. worker count outside Dataproc, cost without user
supplied prices) are left null — never fabricated.
"""

from __future__ import annotations

import io
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import ClusterCost, PipelineConfig
from .results import collect_environment_metadata, git_commit_sha, write_run_metadata
from .tf_input import list_tfrecord_files, load_and_preprocess_jpeg

logger = logging.getLogger(__name__)

RUN_COLUMNS = [
    "run_id",
    "timestamp",
    "git_commit",
    "execution_mode",
    "input_format",
    "dataset_size",
    "input_files",
    "processed_files",
    "failed_files",
    "skipped_files",
    "partitions",
    "workers",
    "target_size",
    "sample_rate",
    "sampled",
    "seed",
    "repeat",
    "runtime_seconds",
    "throughput_images_per_second",
    "output_files",
    "output_size_mb",
    "estimated_cost",
    "estimated_cost_per_1k_images",
]


@dataclass
class BenchmarkPlan:
    """Cartesian benchmark grid: modes x partitions x sample_rates x repeats."""

    input_uri: str
    output_uri: str
    modes: list[str] = field(default_factory=lambda: ["local", "spark"])
    partitions: list[int] = field(default_factory=lambda: [1])
    repeats: int = 5
    sample_rates: list[float] = field(default_factory=lambda: [1.0])
    target_height: int = 192
    target_width: int = 192
    seed: int = 42
    results_csv: str = "reports/tables/benchmark_runs.csv"
    metadata_dir: str | None = "reports/run_metadata"
    worker_hourly_cost: float | None = None
    master_hourly_cost: float | None = None
    num_workers: int | None = None
    num_masters: int = 1

    def __post_init__(self) -> None:
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        bad = [m for m in self.modes if m not in ("local", "spark")]
        if bad:
            raise ValueError(f"unknown modes: {bad}")
        if self.output_uri.startswith("gs://") and "local" in self.modes:
            raise ValueError("local benchmark mode requires a local scratch output path")


def estimate_run_cost(
    runtime_seconds: float,
    processed_files: int,
    cost: ClusterCost | None,
) -> tuple[float | None, float | None]:
    """Estimated (total cost, cost per 1k images). None when prices are unknown.

    Estimates cover job runtime only; cluster startup/idle time is excluded.
    """
    if cost is None or cost.total_hourly_cost <= 0:
        return None, None
    total = (runtime_seconds / 3600.0) * cost.total_hourly_cost
    per_1k = total / processed_files * 1000 if processed_files else None
    return round(total, 6), round(per_1k, 6) if per_1k is not None else None


def _run_once(mode: str, config: PipelineConfig):
    if mode == "local":
        from .local_pipeline import run_local_pipeline

        return run_local_pipeline(config, output_format="tfrecord")
    from .spark_pipeline import run_spark_pipeline

    return run_spark_pipeline(config)


def _run_output_uri(root: str, suffix: str) -> str:
    """Join a benchmark scratch root without corrupting ``gs://`` URIs."""
    if root.startswith("gs://"):
        return f"{root.rstrip('/')}/{suffix}"
    return str(Path(root) / suffix)


def _write_results_csv(df: pd.DataFrame, results_csv: str) -> None:
    """Append benchmark rows locally or atomically rewrite the small GCS CSV.

    Object storage does not provide normal file append semantics, so GCS
    results are read (when present), concatenated, and rewritten. Benchmark
    result tables are tiny compared with the image data, making this both
    reliable and inexpensive.
    """
    if results_csv.startswith("gs://"):
        import tensorflow as tf

        frames = []
        if tf.io.gfile.exists(results_csv):
            with tf.io.gfile.GFile(results_csv, "r") as handle:
                existing_text = handle.read()
            if existing_text.strip():
                frames.append(pd.read_csv(io.StringIO(existing_text)))
        frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        with tf.io.gfile.GFile(results_csv, "w") as handle:
            combined.to_csv(handle, index=False)
        return

    results_path = Path(results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header = not results_path.exists()
    df.to_csv(results_path, mode="a", header=header, index=False)


def run_benchmark(plan: BenchmarkPlan) -> pd.DataFrame:
    """Execute the benchmark grid and append rows to ``plan.results_csv``."""
    cluster_cost = None
    if plan.worker_hourly_cost is not None or plan.master_hourly_cost is not None:
        cluster_cost = ClusterCost(
            worker_hourly_cost=plan.worker_hourly_cost or 0.0,
            master_hourly_cost=plan.master_hourly_cost or 0.0,
            num_workers=plan.num_workers or 0,
            num_masters=plan.num_masters,
        )

    git_sha = git_commit_sha()
    rows = []
    for mode in plan.modes:
        for partitions in plan.partitions:
            for sample_rate in plan.sample_rates:
                for repeat in range(plan.repeats):
                    run_id = uuid.uuid4().hex[:12]
                    suffix = f"{mode}-p{partitions}-r{repeat}-{run_id}"
                    out_dir = _run_output_uri(plan.output_uri, suffix)
                    config = PipelineConfig(
                        input_uri=plan.input_uri,
                        output_uri=out_dir,
                        partitions=partitions,
                        sample_rate=sample_rate,
                        target_height=plan.target_height,
                        target_width=plan.target_width,
                        seed=plan.seed,
                    )
                    logger.info(
                        "benchmark run %s: mode=%s partitions=%d sample_rate=%.3f repeat=%d",
                        run_id,
                        mode,
                        partitions,
                        sample_rate,
                        repeat,
                    )
                    result = _run_once(mode, config)
                    est_cost, est_cost_1k = estimate_run_cost(
                        result.runtime_seconds, result.processed_files, cluster_cost
                    )
                    rows.append({
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "git_commit": git_sha,
                        "execution_mode": mode,
                        "input_format": "jpeg",
                        "dataset_size": result.input_files,
                        "input_files": result.input_files,
                        "processed_files": result.processed_files,
                        "failed_files": result.failed_files,
                        "skipped_files": result.skipped_files,
                        "partitions": partitions,
                        "workers": plan.num_workers,
                        "target_size": f"{plan.target_height}x{plan.target_width}",
                        "sample_rate": sample_rate,
                        "sampled": result.sampled,
                        "seed": plan.seed,
                        "repeat": repeat,
                        "runtime_seconds": result.runtime_seconds,
                        "throughput_images_per_second": result.throughput_images_per_second,
                        "output_files": result.output_files,
                        "output_size_mb": result.output_size_mb,
                        "estimated_cost": est_cost,
                        "estimated_cost_per_1k_images": est_cost_1k,
                    })
                    if plan.metadata_dir:
                        write_run_metadata(
                            plan.metadata_dir,
                            run_id,
                            extra={"benchmark_row": rows[-1]},
                        )
                    # Benchmark outputs are scratch data. Local outputs are removed
                    # immediately; GCS scratch cleanup is left to the caller/workflow.
                    if not out_dir.startswith("gs://"):
                        local_out = Path(out_dir)
                        if local_out.exists():
                            shutil.rmtree(local_out, ignore_errors=True)

    df = pd.DataFrame(rows, columns=RUN_COLUMNS)
    _write_results_csv(df, plan.results_csv)
    return df


def run_io_benchmark(
    jpeg_input: str,
    tfrecord_input: str,
    batch_size: int = 32,
    repeats: int = 5,
    target_size: tuple[int, int] = (192, 192),
    results_csv: str = "reports/tables/io_benchmark_runs.csv",
) -> pd.DataFrame:
    """Compare raw-JPEG and TFRecord input throughput on equivalent image geometry.

    The raw-JPEG path pays JPEG decode plus the same aspect-preserving resize
    and centre-crop geometry used to create the TFRecords. The TFRecord path
    reads images that have already undergone that preprocessing. Both paths
    therefore yield tensors of the same target shape, making the measured
    difference attributable to input/preprocessing strategy rather than a
    different resize operation.
    """
    import tensorflow as tf

    from .local_pipeline import list_image_files
    from .tfrecords import parse_image_and_label

    th, tw = target_size

    def jpeg_dataset():
        files = list_image_files(jpeg_input)
        if not files:
            raise FileNotFoundError(f"no JPEGs under {jpeg_input!r}")
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(
            lambda p: load_and_preprocess_jpeg(p, th, tw),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    tfrecord_files = list_tfrecord_files(tfrecord_input)
    if not tfrecord_files:
        raise FileNotFoundError(f"no TFRecords under {tfrecord_input!r}")

    def tfrecord_dataset():
        ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
        ds = ds.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    rows = []
    for input_format, build_dataset in (
        ("jpeg", jpeg_dataset),
        ("tfrecord", tfrecord_dataset),
    ):
        for repeat in range(repeats):
            start = time.perf_counter()
            count = 0
            for batch in build_dataset():
                if input_format == "jpeg":
                    images = batch
                else:
                    images, _ = batch
                count += int(images.shape[0])
            runtime = time.perf_counter() - start
            rows.append({
                "input_format": input_format,
                "repeat": repeat,
                "examples": count,
                "runtime_seconds": runtime,
                "throughput_images_per_second": count / runtime if runtime else 0.0,
                "target_size": f"{th}x{tw}",
                "batch_size": batch_size,
            })

    df = pd.DataFrame(rows)
    results_path = Path(results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    return df
