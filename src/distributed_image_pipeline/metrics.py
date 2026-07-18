"""Aggregation, scaling and statistical comparison of benchmark runs.

Repeated-run variability is summarised with descriptive statistics and
(optionally) bootstrap percentile intervals over the observed repeats. These
are resampling intervals over a small number of repeats on one machine — they
are *not* population confidence intervals, and are labelled accordingly.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

GROUP_KEYS = ["execution_mode", "input_format", "partitions", "sample_rate", "dataset_size"]


def summarize_benchmark(runs: pd.DataFrame, group_keys: Sequence[str] = GROUP_KEYS) -> pd.DataFrame:
    """Aggregate repeated runs per configuration."""
    keys = [k for k in group_keys if k in runs.columns]
    agg = runs.groupby(keys, dropna=False).agg(
        n_runs=("runtime_seconds", "count"),
        runtime_mean_s=("runtime_seconds", "mean"),
        runtime_std_s=("runtime_seconds", "std"),
        runtime_median_s=("runtime_seconds", "median"),
        runtime_min_s=("runtime_seconds", "min"),
        runtime_max_s=("runtime_seconds", "max"),
        throughput_mean_ips=("throughput_images_per_second", "mean"),
        throughput_std_ips=("throughput_images_per_second", "std"),
        processed_files_mean=("processed_files", "mean"),
        failed_files_total=("failed_files", "sum"),
    ).reset_index()
    return agg.round(4)


def bootstrap_interval(
    values: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap percentile interval of the mean over observed repeats.

    A resampling interval, not a formal population confidence interval.
    """
    values = list(values)
    if not values:
        raise ValueError("no values to bootstrap")
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_resamples)
    )
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return lo, hi


def speedup(serial_runtime: float, parallel_runtime: float) -> float:
    if parallel_runtime <= 0:
        raise ValueError("parallel_runtime must be positive")
    return serial_runtime / parallel_runtime


def parallel_efficiency(speedup_value: float, num_workers: int) -> float:
    """Speedup divided by *worker* count. Only valid with a true worker count —
    partition count is not a worker count and must not be passed here."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    return speedup_value / num_workers


def partition_scaling(summary: pd.DataFrame, mode: str = "spark") -> pd.DataFrame:
    """Runtime ratio versus the 1-partition baseline for one execution mode.

    This is *partition scaling* (task-parallelism within fixed resources), and
    is reported separately from true worker scaling.
    """
    sub = summary[summary["execution_mode"] == mode].copy()
    base = sub.loc[sub["partitions"] == sub["partitions"].min()]
    if base.empty:
        return sub.assign(runtime_ratio_vs_min_partitions=pd.NA)
    baseline = float(base["runtime_mean_s"].iloc[0])
    sub["runtime_ratio_vs_min_partitions"] = baseline / sub["runtime_mean_s"]
    return sub


def worker_scaling(
    summary: pd.DataFrame,
    serial_runtime: float,
) -> pd.DataFrame:
    """True worker scaling: speedup and parallel efficiency vs worker count.

    Requires rows with a non-null ``workers`` column (e.g. Dataproc runs).
    """
    sub = summary.dropna(subset=["workers"]).copy() if "workers" in summary else summary.iloc[0:0]
    if sub.empty:
        return sub
    sub["speedup"] = serial_runtime / sub["runtime_mean_s"]
    sub["parallel_efficiency"] = sub["speedup"] / sub["workers"].astype(float)
    return sub


@dataclass
class ComparisonResult:
    label_a: str
    label_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_difference: float
    relative_difference_pct: float
    method: str
    p_value: float | None
    bootstrap_diff_low: float | None
    bootstrap_diff_high: float | None

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def compare_groups(
    values_a: Sequence[float],
    values_b: Sequence[float],
    label_a: str = "A",
    label_b: str = "B",
    n_resamples: int = 2000,
    seed: int = 0,
) -> ComparisonResult:
    """Compare two sets of repeated measurements.

    Uses the Mann-Whitney U test (non-parametric; no normality assumption is
    justified for a handful of runtimes) when scipy is available, plus a
    bootstrap interval on the difference of means. With very few repeats the
    test has little power — the report presents both statistical and practical
    differences and does not overstate significance.
    """
    values_a, values_b = list(values_a), list(values_b)
    if not values_a or not values_b:
        raise ValueError("both groups need at least one value")
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)

    p_value = None
    method = "descriptive"
    if len(values_a) >= 3 and len(values_b) >= 3:
        try:
            from scipy.stats import mannwhitneyu

            p_value = float(mannwhitneyu(values_a, values_b).pvalue)
            method = "mann-whitney-u"
        except ImportError:
            pass

    rng = random.Random(seed)
    diffs = sorted(
        sum(rng.choices(values_a, k=len(values_a))) / len(values_a)
        - sum(rng.choices(values_b, k=len(values_b))) / len(values_b)
        for _ in range(n_resamples)
    )
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[min(n_resamples - 1, int(0.975 * n_resamples))]

    return ComparisonResult(
        label_a=label_a,
        label_b=label_b,
        n_a=len(values_a),
        n_b=len(values_b),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_difference=mean_a - mean_b,
        relative_difference_pct=(mean_a - mean_b) / mean_b * 100 if mean_b else float("nan"),
        method=method,
        p_value=p_value,
        bootstrap_diff_low=lo,
        bootstrap_diff_high=hi,
    )
