import pandas as pd
import pytest

from distributed_image_pipeline.metrics import (
    bootstrap_interval,
    compare_groups,
    parallel_efficiency,
    partition_scaling,
    speedup,
    summarize_benchmark,
    worker_scaling,
)


def make_runs():
    rows = []
    for mode, base in (("local", 10.0), ("spark", 6.0)):
        for partitions in (1, 2):
            for repeat in range(3):
                rt = base / partitions + repeat * 0.1
                rows.append({
                    "execution_mode": mode,
                    "input_format": "jpeg",
                    "partitions": partitions,
                    "sample_rate": 1.0,
                    "dataset_size": 100,
                    "repeat": repeat,
                    "runtime_seconds": rt,
                    "throughput_images_per_second": 100 / rt,
                    "processed_files": 100,
                    "failed_files": 0,
                    "workers": None,
                })
    return pd.DataFrame(rows)


def test_summarize_benchmark():
    summary = summarize_benchmark(make_runs())
    assert len(summary) == 4
    row = summary[
        (summary.execution_mode == "local") & (summary.partitions == 1)
    ].iloc[0]
    assert row.n_runs == 3
    assert row.runtime_mean_s == pytest.approx(10.1)
    assert row.runtime_median_s == pytest.approx(10.1)
    assert row.runtime_min_s == pytest.approx(10.0)
    assert row.runtime_max_s == pytest.approx(10.2)
    assert row.runtime_std_s == pytest.approx(0.1, abs=1e-3)


def test_bootstrap_interval_contains_mean():
    lo, hi = bootstrap_interval([10.0, 10.1, 10.2, 9.9], seed=1)
    assert lo <= 10.05 <= hi
    with pytest.raises(ValueError):
        bootstrap_interval([])


def test_speedup_and_efficiency():
    assert speedup(10.0, 2.5) == 4.0
    assert parallel_efficiency(4.0, 8) == 0.5
    with pytest.raises(ValueError):
        speedup(10.0, 0)
    with pytest.raises(ValueError):
        parallel_efficiency(4.0, 0)


def test_partition_scaling_not_worker_scaling():
    summary = summarize_benchmark(make_runs())
    ps = partition_scaling(summary, mode="spark")
    assert "runtime_ratio_vs_min_partitions" in ps.columns
    p1 = ps[ps.partitions == 1].iloc[0]
    p2 = ps[ps.partitions == 2].iloc[0]
    assert p1.runtime_ratio_vs_min_partitions == pytest.approx(1.0)
    assert p2.runtime_ratio_vs_min_partitions > 1.5


def test_worker_scaling_empty_without_worker_counts():
    summary = summarize_benchmark(make_runs())
    ws = worker_scaling(summary, serial_runtime=10.0)
    assert ws.empty  # partitions are never treated as workers


def test_worker_scaling_with_workers():
    summary = summarize_benchmark(make_runs())
    summary["workers"] = 4
    ws = worker_scaling(summary, serial_runtime=10.0)
    assert not ws.empty
    assert (ws["parallel_efficiency"] == ws["speedup"] / 4).all()


def test_compare_groups():
    a = [10.0, 10.2, 10.1, 9.9, 10.0]
    b = [5.0, 5.1, 4.9, 5.2, 5.0]
    res = compare_groups(a, b, "local", "spark")
    assert res.mean_difference == pytest.approx(5.0, abs=0.01)
    assert res.method == "mann-whitney-u"
    assert res.p_value is not None and res.p_value < 0.05
    assert res.bootstrap_diff_low <= res.mean_difference <= res.bootstrap_diff_high


def test_compare_groups_small_samples_descriptive():
    res = compare_groups([1.0], [2.0])
    assert res.method == "descriptive"
    assert res.p_value is None
