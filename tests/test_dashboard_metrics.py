import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from metrics import (  # noqa: E402
    DashboardDataError,
    best_mode_row,
    epoch_delta_pct,
    failed_file_stats,
    io_format_mean,
    io_speedup,
    throughput_speedup,
    training_format_row,
    training_summary_table,
)

TABLES_DIR = Path(__file__).resolve().parents[1] / "reports" / "tables"


def _summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "execution_mode": ["local", "spark", "spark"],
            "partitions": [1, 2, 4],
            "throughput_mean_ips": [100.0, 150.0, 200.0],
        }
    )


def test_best_mode_row_picks_highest_throughput():
    row = best_mode_row(_summary_df(), "spark")
    assert row["partitions"] == 4
    assert row["throughput_mean_ips"] == 200.0


def test_best_mode_row_missing_mode_raises():
    with pytest.raises(DashboardDataError):
        best_mode_row(_summary_df(), "dataproc")


def test_best_mode_row_missing_column_raises():
    with pytest.raises(DashboardDataError):
        best_mode_row(pd.DataFrame({"execution_mode": ["spark"]}), "spark")


def test_throughput_speedup():
    df = _summary_df()
    speedup = throughput_speedup(best_mode_row(df, "spark"), best_mode_row(df, "local"))
    assert speedup == pytest.approx(2.0)


def test_throughput_speedup_zero_baseline_raises():
    zero = pd.Series({"throughput_mean_ips": 0.0})
    fast = pd.Series({"throughput_mean_ips": 10.0})
    with pytest.raises(DashboardDataError):
        throughput_speedup(fast, zero)


def test_io_format_mean_and_speedup():
    io_df = pd.DataFrame(
        {
            "input_format": ["jpeg", "jpeg", "tfrecord", "tfrecord"],
            "samples_per_second": [100.0, 200.0, 600.0, 600.0],
        }
    )
    jpeg = io_format_mean(io_df, "jpeg")
    tfrecord = io_format_mean(io_df, "tfrecord")
    assert jpeg == pytest.approx(150.0)
    assert io_speedup(jpeg, tfrecord) == pytest.approx(4.0)
    with pytest.raises(DashboardDataError):
        io_format_mean(io_df, "parquet")
    with pytest.raises(DashboardDataError):
        io_speedup(0.0, tfrecord)


def test_training_summary_table_and_row_selection():
    training_df = pd.DataFrame(
        {
            "input_format": ["jpeg", "jpeg", "tfrecord", "tfrecord"],
            "epoch": [0, 1, 0, 1],
            "training_time_seconds": [10.0, 8.0, 6.0, 4.0],
            "samples_per_second": [100.0, 120.0, 150.0, 180.0],
            "validation_accuracy": [0.4, 0.5, 0.45, 0.55],
        }
    )
    summary = training_summary_table(training_df)
    jpeg = training_format_row(summary, "jpeg")
    tfrecord = training_format_row(summary, "tfrecord")
    assert jpeg["mean_epoch_s"] == pytest.approx(9.0)
    assert jpeg["validation_accuracy"] == pytest.approx(0.5)
    assert tfrecord["total_s"] == pytest.approx(10.0)
    assert epoch_delta_pct(jpeg, tfrecord) == pytest.approx((9.0 - 5.0) / 9.0 * 100)
    with pytest.raises(DashboardDataError):
        training_format_row(summary, "parquet")


def test_epoch_delta_pct_zero_baseline_raises():
    zero = pd.Series({"mean_epoch_s": 0.0})
    with pytest.raises(DashboardDataError):
        epoch_delta_pct(zero, pd.Series({"mean_epoch_s": 1.0}))


def test_failed_file_stats_reports_events_and_worst_run():
    runs_df = pd.DataFrame({"failed_files": [2, 2, 0, None]})
    total_events, worst_run = failed_file_stats(runs_df)
    assert total_events == 4
    assert worst_run == 2
    with pytest.raises(DashboardDataError):
        failed_file_stats(pd.DataFrame({"runtime_seconds": [1.0]}))


@pytest.mark.skipif(not TABLES_DIR.exists(), reason="committed benchmark tables not present")
def test_committed_tables_satisfy_dashboard_schema():
    summary_df = pd.read_csv(TABLES_DIR / "benchmark_summary.csv")
    runs_df = pd.read_csv(TABLES_DIR / "benchmark_runs.csv")
    io_df = pd.read_csv(TABLES_DIR / "io_benchmark_runs.csv")
    training_df = pd.read_csv(TABLES_DIR / "training_runs.csv")

    best_spark = best_mode_row(summary_df, "spark")
    best_local = best_mode_row(summary_df, "local")
    assert throughput_speedup(best_spark, best_local) > 0

    jpeg = io_format_mean(io_df, "jpeg")
    tfrecord = io_format_mean(io_df, "tfrecord")
    assert io_speedup(jpeg, tfrecord) > 0

    summary = training_summary_table(training_df)
    jpeg_row = training_format_row(summary, "jpeg")
    tfrecord_row = training_format_row(summary, "tfrecord")
    assert isinstance(epoch_delta_pct(jpeg_row, tfrecord_row), float)

    total_events, worst_run = failed_file_stats(runs_df)
    assert total_events >= worst_run >= 0
