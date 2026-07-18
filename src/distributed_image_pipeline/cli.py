"""Unified command-line interface.

Run ``python -m distributed_image_pipeline.cli --help`` for usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import ConfigError, load_config
from .labels import DEFAULT_CLASSES

logger = logging.getLogger(__name__)


def _setup_logging(verbosity: str) -> None:
    logging.basicConfig(
        level=getattr(logging, verbosity.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _add_common_pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True, help="input directory, glob, or gs:// glob")
    p.add_argument("--output", required=True, help="output directory or gs:// prefix")
    p.add_argument("--partitions", type=int, default=1, help="partition/shard count (default 1)")
    p.add_argument(
        "--sample-rate", type=float, default=1.0,
        help="fraction of the dataset to process (default 1.0 = full dataset). "
             "Sampled runs are labelled as such in all outputs.",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument(
        "--target-size", type=int, nargs=2, default=[192, 192],
        metavar=("HEIGHT", "WIDTH"), help="output image size (default 192 192)",
    )
    p.add_argument(
        "--error-policy", choices=["raise", "skip", "log"], default="log",
        help="behaviour for corrupt/unreadable images (default log)",
    )
    p.add_argument("--config-file", default=None, help="optional JSON config file")


def _config_from_args(args: argparse.Namespace):
    return load_config(
        config_file=args.config_file,
        input_uri=args.input,
        output_uri=args.output,
        partitions=args.partitions,
        sample_rate=args.sample_rate,
        seed=args.seed,
        target_height=args.target_size[0],
        target_width=args.target_size[1],
        error_policy=args.error_policy,
    )


def _print_result(result) -> None:
    d = result.to_dict()
    d.pop("output_paths", None)
    print(json.dumps(d, indent=2))


def cmd_preprocess_local(args) -> int:
    from .local_pipeline import run_local_pipeline

    result = run_local_pipeline(_config_from_args(args), output_format=args.output_format)
    _print_result(result)
    return 0


def cmd_preprocess_spark(args) -> int:
    from .spark_pipeline import run_spark_pipeline

    result = run_spark_pipeline(_config_from_args(args))
    _print_result(result)
    return 0


def cmd_write_tfrecords(args) -> int:
    # TFRecord-only variant of the local pipeline (kept as an explicit command).
    from .local_pipeline import run_local_pipeline

    result = run_local_pipeline(_config_from_args(args), output_format="tfrecord")
    _print_result(result)
    return 0


def cmd_benchmark(args) -> int:
    from .benchmark import BenchmarkPlan, run_benchmark

    plan = BenchmarkPlan(
        input_uri=args.input,
        output_uri=args.output,
        modes=args.modes,
        partitions=args.partitions,
        repeats=args.repeats,
        sample_rates=args.sample_rate,
        target_height=args.target_size[0],
        target_width=args.target_size[1],
        seed=args.seed,
        results_csv=args.results,
        metadata_dir=args.metadata_dir,
        worker_hourly_cost=args.worker_hourly_cost,
        master_hourly_cost=args.master_hourly_cost,
        num_workers=args.num_workers,
        num_masters=args.num_masters,
    )
    df = run_benchmark(plan)
    print(f"wrote {len(df)} benchmark rows to {args.results}")
    if args.summary:
        from .metrics import summarize_benchmark

        summary = summarize_benchmark(df)
        summary.to_csv(args.summary, index=False)
        print(f"wrote summary to {args.summary}")
    return 0


def cmd_io_benchmark(args) -> int:
    from .benchmark import run_io_benchmark

    df = run_io_benchmark(
        jpeg_input=args.jpeg_input,
        tfrecord_input=args.tfrecord_input,
        batch_size=args.batch_size,
        repeats=args.repeats,
        target_size=tuple(args.target_size),
        results_csv=args.results,
    )
    print(f"wrote {len(df)} I/O benchmark rows to {args.results}")
    return 0


def cmd_train(args) -> int:
    from .training import TrainingSpec, train_and_compare

    spec = TrainingSpec(
        jpeg_input=args.jpeg_input,
        tfrecord_input=args.tfrecord_input,
        classes=DEFAULT_CLASSES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_size=tuple(args.target_size),
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        splits_dir=args.splits_dir,
        results_csv=args.results,
    )
    df = train_and_compare(spec, formats=args.formats)
    print(df.to_string(index=False))
    return 0


def cmd_report(args) -> int:
    from .report import generate_report

    path = generate_report(
        benchmark_csv=args.input,
        output_md=args.output,
        figures_dir=args.figures_dir,
        training_csv=args.training_csv,
        io_csv=args.io_csv,
    )
    print(f"wrote report to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distributed_image_pipeline",
        description="Distributed image ML pipeline: preprocessing, TFRecords, "
                    "benchmarking, training and reporting.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess-local", help="local single-process preprocessing baseline")
    _add_common_pipeline_args(p)
    p.add_argument("--output-format", choices=["tfrecord", "jpeg"], default="tfrecord")
    p.set_defaults(func=cmd_preprocess_local)

    p = sub.add_parser("preprocess-spark", help="Spark distributed preprocessing")
    _add_common_pipeline_args(p)
    p.set_defaults(func=cmd_preprocess_spark)

    p = sub.add_parser("write-tfrecords", help="preprocess images and write TFRecord shards")
    _add_common_pipeline_args(p)
    p.set_defaults(func=cmd_write_tfrecords)

    p = sub.add_parser("benchmark", help="run repeated preprocessing benchmarks")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="scratch output directory for benchmark runs")
    p.add_argument("--modes", nargs="+", default=["local", "spark"], choices=["local", "spark"])
    p.add_argument("--partitions", type=int, nargs="+", default=[1])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--sample-rate", type=float, nargs="+", default=[1.0],
        help="one or more dataset fractions (default 1.0 = full dataset)",
    )
    p.add_argument("--target-size", type=int, nargs=2, default=[192, 192],
                   metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results", default="reports/tables/benchmark_runs.csv")
    p.add_argument("--summary", default="reports/tables/benchmark_summary.csv")
    p.add_argument("--metadata-dir", default="reports/run_metadata")
    p.add_argument("--worker-hourly-cost", type=float, default=None,
                   help="hourly cost per worker node for cost ESTIMATION (no default pricing)")
    p.add_argument("--master-hourly-cost", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None,
                   help="actual worker count, if known (distinct from partitions)")
    p.add_argument("--num-masters", type=int, default=1)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("io-benchmark", help="raw JPEG vs TFRecord input-pipeline benchmark")
    p.add_argument("--jpeg-input", required=True, help="directory or glob of raw JPEGs")
    p.add_argument("--tfrecord-input", required=True, help="directory or glob of .tfrec shards")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--target-size", type=int, nargs=2, default=[192, 192],
                   metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--results", default="reports/tables/io_benchmark_runs.csv")
    p.set_defaults(func=cmd_io_benchmark)

    p = sub.add_parser("train", help="downstream training comparison (JPEG vs TFRecord input)")
    p.add_argument("--jpeg-input", required=True)
    p.add_argument("--tfrecord-input", required=True)
    p.add_argument("--formats", nargs="+", default=["jpeg", "tfrecord"],
                   choices=["jpeg", "tfrecord"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--target-size", type=int, nargs=2, default=[192, 192],
                   metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--results", default="reports/tables/training_runs.csv")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("report", help="generate the markdown benchmark report and figures")
    p.add_argument("--input", default="reports/tables/benchmark_runs.csv")
    p.add_argument("--output", default="reports/benchmark_report.md")
    p.add_argument("--figures-dir", default="reports/figures")
    p.add_argument("--training-csv", default=None)
    p.add_argument("--io-csv", default=None)
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)
    try:
        return args.func(args)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
