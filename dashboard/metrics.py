"""Pure metric computations for the Streamlit dashboard.

These helpers validate the committed benchmark CSVs before use so the app
fails with a clear message instead of crashing on malformed or partial data.
"""

from __future__ import annotations

import pandas as pd


class DashboardDataError(ValueError):
    """Raised when a committed results file is missing required data."""


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DashboardDataError(
            f"The {name} file is missing required columns: {', '.join(missing)}."
        )


def best_mode_row(summary_df: pd.DataFrame, mode: str) -> pd.Series:
    """Return the highest-throughput summary row for an execution mode."""
    require_columns(
        summary_df, ["execution_mode", "throughput_mean_ips", "partitions"], "benchmark summary"
    )
    rows = summary_df[summary_df["execution_mode"] == mode]
    if rows.empty:
        raise DashboardDataError(
            f"The benchmark summary contains no rows for execution mode '{mode}'."
        )
    return rows.sort_values("throughput_mean_ips", ascending=False).iloc[0]


def throughput_speedup(best_spark: pd.Series, best_local: pd.Series) -> float:
    local_ips = float(best_local["throughput_mean_ips"])
    if local_ips <= 0:
        raise DashboardDataError("Local baseline throughput is zero; cannot compute speedup.")
    return float(best_spark["throughput_mean_ips"]) / local_ips


def io_format_mean(io_df: pd.DataFrame, input_format: str) -> float:
    require_columns(io_df, ["input_format", "samples_per_second"], "I/O benchmark")
    rows = io_df[io_df["input_format"] == input_format]
    if rows.empty:
        raise DashboardDataError(
            f"The I/O benchmark contains no rows for input format '{input_format}'."
        )
    return float(rows["samples_per_second"].mean())


def io_speedup(jpeg_ips: float, tfrecord_ips: float) -> float:
    if jpeg_ips <= 0:
        raise DashboardDataError("JPEG I/O throughput is zero; cannot compute speedup.")
    return tfrecord_ips / jpeg_ips


def training_summary_table(training_df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        training_df,
        ["input_format", "epoch", "training_time_seconds", "samples_per_second",
         "validation_accuracy"],
        "training benchmark",
    )
    summary = training_df.groupby("input_format", as_index=False).agg(
        mean_epoch_s=("training_time_seconds", "mean"),
        total_s=("training_time_seconds", "sum"),
        mean_samples_per_second=("samples_per_second", "mean"),
    )
    final_accuracy = (
        training_df.sort_values("epoch")
        .groupby("input_format", as_index=False)
        .tail(1)[["input_format", "validation_accuracy"]]
    )
    return summary.merge(final_accuracy, on="input_format", how="left")


def training_format_row(training_summary: pd.DataFrame, input_format: str) -> pd.Series:
    rows = training_summary[training_summary["input_format"] == input_format]
    if rows.empty:
        raise DashboardDataError(
            f"The training benchmark contains no rows for input format '{input_format}'."
        )
    return rows.iloc[0]


def epoch_delta_pct(jpeg_row: pd.Series, tfrecord_row: pd.Series) -> float:
    jpeg_epoch_s = float(jpeg_row["mean_epoch_s"])
    if jpeg_epoch_s <= 0:
        raise DashboardDataError("JPEG mean epoch time is zero; cannot compute relative change.")
    return (jpeg_epoch_s - float(tfrecord_row["mean_epoch_s"])) / jpeg_epoch_s * 100


def failed_file_stats(runs_df: pd.DataFrame) -> tuple[int, int]:
    """Return (total failure events across all repeats, worst single run).

    Repeats reprocess the same inputs, so summing across runs counts repeated
    failure events, not distinct files — both views are returned explicitly.
    """
    require_columns(runs_df, ["failed_files"], "benchmark runs")
    failed = runs_df["failed_files"].fillna(0)
    return int(failed.sum()), int(failed.max())
