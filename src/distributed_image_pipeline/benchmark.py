"""Benchmark orchestration: repeated preprocessing runs and I/O benchmarks.

Every run is recorded as one CSV row. Fields that cannot be measured in the
current environment (e.g. worker count outside Dataproc, cost without user
supplied prices) are left null — never fabricated.
"""

from __future__ import annotations

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
    scratch = Path(plan.output_uri)
    for mode in plan.modes:
        for partitions in plan.partitions:
            for sample_rate in plan.sample_rates:
                for repeat in range(plan.repeats):
                    run_id = uuid.uuid4().hex[:12]
                    out_dir = scratch / f"{mode}-p{partitions}-r{repeat}-{run_id}"
                    config = PipelineConfig(
                        input_uri=plan.input_uri,
                        output_uri=str(out_dir),
                        partitions=partitions,
                        sample_rate=sample_rate,
                        target_height=plan.target_height,
                        target_width=plan.target_width,
                        seed=plan.seed,
                    )
                    logger.info(
                        "benchmark run %s: mode=%s partitions=%d sample_rate=%.3f repeat=%d",
                        run_id, mode, partitions, sample_rate, repeat,
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
                            plan.metadata_dir, run_id,
                            extra={"benchmark_row": rows[-1]},
                        )
                    # benchmark outputs are scratch data; clean up local shards
                    if out_dir.exists():
                        shutil.rmtree(out_dir, ignore_errors=True)

    df = pd.DataFrame(rows, columns=RUN_COLUMNS)
    results_path = Path(plan.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header = not results_path.exists()
    df.to_csv(results_path, mode="a", header=header, index=False)
    return df


def run_io_benchmark(
    jpeg_input: str,
    tfrecord_input: str,
    batch_size: int = 32,
    repeats: int = 5,
    target_size: tuple[int, int] = (192, 192),
    results_csv: str = "reports/tables/io_benchmark_runs.csv",
) -> pd.DataFrame:
    """Compare tf.data input throughput: raw JPEG decode+resize vs TFRecord parse.

    Both paths yield batched uint8 image tensors of ``target_size`` so the
    comparison is like-for-like: the JPEG path pays decode+resize per epoch,
    the TFRecord path reads already-preprocessed images (that preprocessing
    cost is measured separately by the preprocessing benchmark).
    """
    import tensorflow as tf

    from .local_pipeline import list_image_files
    from .tfrecords import parse_image_and_label

    th, tw = target_size

    def jpeg_dataset():
        files = list_image_files(jpeg_input)
        if not files:
            raise FileNotFoundError(f"no JPEGs under {jpeg_input!r}")

        def load(path):
            img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            img = tf.image.resize(img, [th, tw])
            return tf.cast(img, tf.uint8)

        return tf.data.Dataset.from_tensor_slices(files).map(
            load, num_parallel_calls=tf.data.AUTOTUNE
        )

    def tfrecord_dataset():
        shards = list_image_files(tfrecord_input) if tfrecord_input.endswith(
            (".tfrec", ".tfrecord")
        ) else None
        if shards is None:
            p = Path(tfrecord_input)
            shards = sorted(str(f) for f in p.rglob("*.tfrec")) if p.is_dir() else sorted(
                tf.io.gfile.glob(tfrecord_input)
            )
        if not shards:
            raise FileNotFoundError(f"no TFRecord shards under {tfrecord_input!r}")
        ds = tf.data.TFRecordDataset(shards, num_parallel_reads=tf.data.AUTOTUNE)
        return ds.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).map(
            lambda img, lbl: tf.ensure_shape(img, [th, tw, 3])
        )

    git_sha = git_commit_sha()
    env = collect_environment_metadata()
    rows = []
    for fmt, builder in (("jpeg", jpeg_dataset), ("tfrecord", tfrecord_dataset)):
        for repeat in range(repeats):
            ds = builder().batch(batch_size).prefetch(tf.data.AUTOTUNE)
            n_samples = 0
            batch_times = []
            start = time.perf_counter()
            prev = start
            for batch in ds:
                now = time.perf_counter()
                batch_times.append(now - prev)
                prev = now
                n_samples += int(batch.shape[0]) if not isinstance(batch, tuple) else int(
                    batch[0].shape[0]
                )
            epoch_time = time.perf_counter() - start
            rows.append({
                "run_id": uuid.uuid4().hex[:12],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_sha,
                "input_format": fmt,
                "repeat": repeat,
                "batch_size": batch_size,
                "samples": n_samples,
                "epoch_time_seconds": epoch_time,
                "samples_per_second": n_samples / epoch_time if epoch_time > 0 else 0.0,
                "mean_batch_latency_seconds": sum(batch_times) / len(batch_times)
                if batch_times else None,
                "tensorflow_version": env["tensorflow_version"],
            })
            logger.info(
                "io benchmark %s repeat %d: %d samples in %.2fs (%.1f samples/s)",
                fmt, repeat, n_samples, epoch_time, rows[-1]["samples_per_second"],
            )

    df = pd.DataFrame(rows)
    path = Path(results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return df
