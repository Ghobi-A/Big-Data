"""Markdown benchmark report and portfolio figures.

Every table and figure is derived from actual benchmark CSVs; nothing is
invented. Questions that the collected data cannot answer are explicitly
reported as "not yet measured".
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .metrics import compare_groups, partition_scaling, summarize_benchmark

logger = logging.getLogger(__name__)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _errorbar_by_group(ax, summary, x_col, y_col, err_col, group_col):
    for name, group in summary.groupby(group_col):
        g = group.sort_values(x_col)
        ax.errorbar(
            g[x_col], g[y_col], yerr=g[err_col].fillna(0), marker="o", capsize=3, label=str(name)
        )
    ax.legend(title=group_col)


def make_figures(
    runs: pd.DataFrame, summary: pd.DataFrame, figures_dir: str | Path
) -> list[Path]:
    """Generate the portfolio figures supported by the available data."""
    plt = _plt()
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    def save(fig, name):
        path = out / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(path)

    n_datasets = summary["dataset_size"].nunique()
    ds_note = f"dataset size(s): {sorted(summary['dataset_size'].unique())}"

    # Runtime / throughput by configuration
    cfg_labels = summary.apply(
        lambda r: f"{r.execution_mode}\np{int(r.partitions)}\nn={int(r.dataset_size)}", axis=1
    )
    for y, err, name, ylabel in [
        ("runtime_mean_s", "runtime_std_s", "runtime_by_configuration.png",
         "mean runtime (s)"),
        ("throughput_mean_ips", "throughput_std_ips", "throughput_by_configuration.png",
         "mean throughput (images/s)"),
    ]:
        fig, ax = plt.subplots(figsize=(max(6, len(summary) * 0.9), 4))
        ax.bar(range(len(summary)), summary[y], yerr=summary[err].fillna(0), capsize=3)
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(cfg_labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by configuration (error bars: std over repeats; {ds_note})")
        save(fig, name)

    # Runtime / throughput vs partitions
    if summary["partitions"].nunique() > 1:
        for y, err, name, ylabel in [
            ("runtime_mean_s", "runtime_std_s", "runtime_vs_partitions.png", "mean runtime (s)"),
            ("throughput_mean_ips", "throughput_std_ips", "throughput_vs_partitions.png",
             "mean throughput (images/s)"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            _errorbar_by_group(ax, summary, "partitions", y, err, "execution_mode")
            ax.set_xlabel("partitions")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs partition count ({ds_note})")
            save(fig, name)

    # Dataset-size scaling
    if n_datasets > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        _errorbar_by_group(
            ax, summary, "dataset_size", "runtime_mean_s", "runtime_std_s", "execution_mode"
        )
        ax.set_xlabel("dataset size (images)")
        ax.set_ylabel("mean runtime (s)")
        ax.set_title("Runtime vs dataset size (error bars: std over repeats)")
        save(fig, "runtime_vs_dataset_size.png")

    # True worker scaling (only when worker counts were recorded)
    workers = runs.dropna(subset=["workers"]) if "workers" in runs else runs.iloc[0:0]
    if not workers.empty and workers["workers"].nunique() > 1:
        wsum = summarize_benchmark(workers, ["execution_mode", "workers"])
        serial = wsum.loc[wsum["workers"] == wsum["workers"].min(), "runtime_mean_s"].iloc[0]
        wsum["speedup"] = serial / wsum["runtime_mean_s"]
        wsum["efficiency"] = wsum["speedup"] / wsum["workers"].astype(float)
        for y, name, ylabel in [
            ("speedup", "speedup_vs_workers.png", "speedup vs smallest cluster"),
            ("efficiency", "scaling_efficiency.png", "parallel efficiency"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            for mode, g in wsum.groupby("execution_mode"):
                g = g.sort_values("workers")
                ax.plot(g["workers"], g[y], marker="o", label=mode)
            ax.set_xlabel("workers")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.set_title(f"{ylabel} vs worker count ({ds_note})")
            save(fig, name)

    return made


def make_io_figure(io_df: pd.DataFrame, figures_dir: str | Path) -> Path | None:
    plt = _plt()
    if io_df.empty:
        return None
    agg = io_df.groupby("input_format")["samples_per_second"].agg(["mean", "std", "count"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(agg.index, agg["mean"], yerr=agg["std"].fillna(0), capsize=4)
    ax.set_ylabel("samples/second")
    ax.set_title(
        f"Input pipeline throughput: raw JPEG vs TFRecord\n"
        f"(mean over {int(agg['count'].min())} repeats; error bars: std)"
    )
    path = Path(figures_dir) / "jpeg_vs_tfrecord_throughput.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_training_figure(train_df: pd.DataFrame, figures_dir: str | Path) -> Path | None:
    plt = _plt()
    if train_df.empty:
        return None
    agg = train_df.groupby("input_format")["training_time_seconds"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(agg.index, agg["mean"], yerr=agg["std"].fillna(0), capsize=4)
    ax.set_ylabel("mean epoch time (s)")
    ax.set_title("Training epoch time by input format (error bars: std over epochs)")
    path = Path(figures_dir) / "training_epoch_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.3f}"
        return "" if v is None else str(v)

    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    lines.extend(
        "| " + " | ".join(fmt(row[c]) for c in cols) + " |" for _, row in df.iterrows()
    )
    return "\n".join(lines)


def _answer(question: str, answer: str) -> str:
    return f"**{question}**\n\n{answer}\n"


def generate_report(
    benchmark_csv: str,
    output_md: str,
    figures_dir: str = "reports/figures",
    training_csv: str | None = None,
    io_csv: str | None = None,
) -> Path:
    """Generate the markdown benchmark report from measured CSVs."""
    runs = pd.read_csv(benchmark_csv)
    if runs.empty:
        raise ValueError(f"{benchmark_csv} contains no benchmark runs")
    summary = summarize_benchmark(runs)
    figures = make_figures(runs, summary, figures_dir)

    sections: list[str] = []
    sections.append("# Benchmark report\n")
    sections.append(
        f"_Generated from measured runs in `{benchmark_csv}`. All values are actual "
        "measurements on the hardware recorded in `reports/run_metadata/`; "
        "cost figures, when present, are "
        "estimates from user-supplied prices._\n"
    )
    sampled = runs[runs["sampled"] == True]  # noqa: E712
    if not sampled.empty:
        sections.append(
            f"> **Note:** {len(sampled)} of {len(runs)} runs used dataset sampling "
            "(see `sample_rate` column). Sampled runs are labelled and are not "
            "full-dataset results.\n"
        )

    sections.append("## Summary by configuration\n")
    sections.append(_md_table(summary) + "\n")

    sections.append("## Key observations\n")

    # Q1: local vs spark
    local = runs[runs.execution_mode == "local"]["runtime_seconds"]
    spark = runs[runs.execution_mode == "spark"]["runtime_seconds"]
    if len(local) >= 1 and len(spark) >= 1:
        cmp = compare_groups(list(local), list(spark), "local", "spark")
        direction = "faster" if cmp.mean_difference > 0 else "slower"
        stat = (
            f" (Mann-Whitney p={cmp.p_value:.3f}; bootstrap interval of the mean difference "
            f"[{cmp.bootstrap_diff_low:.2f}, {cmp.bootstrap_diff_high:.2f}] s — a resampling "
            f"interval over repeats, not a population confidence interval)"
            if cmp.p_value is not None
            else " (too few repeats for a statistical test; descriptive only)"
        )
        sections.append(_answer(
            "Does Spark improve preprocessing throughput over the local baseline?",
            f"Across all recorded runs, Spark was on average {direction} than local: "
            f"mean local runtime {cmp.mean_a:.2f} s vs mean Spark runtime {cmp.mean_b:.2f} s"
            f"{stat}. Whether this is a *practical* difference depends on the runtime "
            f"magnitudes above.",
        ))
    else:
        sections.append(_answer(
            "Does Spark improve preprocessing throughput over the local baseline?",
            "Not yet measured: runs for both execution modes are required.",
        ))

    # Q2: partition plateau
    spark_summary = partition_scaling(summary, mode="spark")
    if spark_summary["partitions"].nunique() > 1:
        best = spark_summary.loc[spark_summary["runtime_mean_s"].idxmin()]
        sections.append(_answer(
            "At what partition count does performance plateau?",
            f"The lowest mean Spark runtime was at {int(best.partitions)} partitions "
            f"({best.runtime_mean_s:.2f} s). See `runtime_vs_partitions.png`. "
            "This is partition scaling on fixed resources — it is not worker scaling.",
        ))
    else:
        sections.append(_answer(
            "At what partition count does performance plateau?",
            "Not yet measured: benchmark runs across multiple partition counts are required.",
        ))

    # Q3: worker scaling
    if "workers" in runs and runs["workers"].notna().any() and runs["workers"].nunique() > 1:
        sections.append(_answer(
            "Does increasing workers produce near-linear scaling?",
            "See `speedup_vs_workers.png` and `scaling_efficiency.png` for measured "
            "speedup and parallel efficiency.",
        ))
    else:
        sections.append(_answer(
            "Does increasing workers produce near-linear scaling?",
            "Not yet measured: no runs with recorded worker counts (requires Dataproc "
            "runs with `--num-workers`). Partition counts are not treated as workers.",
        ))

    # Q4: stability
    stability = summary[summary["n_runs"] >= 2]
    if not stability.empty:
        cv = (stability["runtime_std_s"] / stability["runtime_mean_s"] * 100).max()
        sections.append(_answer(
            "How stable are runtimes across repeated runs?",
            f"Worst-case coefficient of variation across configurations with repeats: "
            f"{cv:.1f}% (std/mean of runtime). Per-configuration std is in the summary table.",
        ))
    else:
        sections.append(_answer(
            "How stable are runtimes across repeated runs?",
            "Not yet measured: repeated runs are required.",
        ))

    # Q5: dataset size
    if summary["dataset_size"].nunique() > 1:
        sections.append(_answer(
            "How does dataset size affect runtime?",
            "See `runtime_vs_dataset_size.png` for the measured relationship.",
        ))
    else:
        sections.append(_answer(
            "How does dataset size affect runtime?",
            "Not yet measured: runs at multiple dataset sizes (via `--sample-rate`) "
            "are required.",
        ))

    # Q6/Q7: I/O + training
    io_df = pd.read_csv(io_csv) if io_csv and Path(io_csv).exists() else pd.DataFrame()
    if not io_df.empty:
        fig = make_io_figure(io_df, figures_dir)
        agg = io_df.groupby("input_format")["samples_per_second"].mean()
        if {"jpeg", "tfrecord"} <= set(agg.index):
            ratio = agg["tfrecord"] / agg["jpeg"]
            sections.append(_answer(
                "Does TFRecord improve input throughput?",
                f"Measured input throughput: TFRecord {agg['tfrecord']:.1f} samples/s vs "
                f"raw JPEG {agg['jpeg']:.1f} samples/s (ratio {ratio:.2f}x, TFRecord reads "
                f"pre-processed images while the JPEG path decodes and resizes per epoch). "
                f"See `{fig.name if fig else ''}`.",
            ))
    else:
        sections.append(_answer(
            "Does TFRecord improve input throughput?",
            "Not yet measured: run `io-benchmark` to collect data.",
        ))

    train_df = (
        pd.read_csv(training_csv) if training_csv and Path(training_csv).exists()
        else pd.DataFrame()
    )
    if not train_df.empty:
        fig = make_training_figure(train_df, figures_dir)
        agg = train_df.groupby("input_format").agg(
            mean_epoch_s=("training_time_seconds", "mean"),
            total_s=("training_time_seconds", "sum"),
            final_val_acc=("validation_accuracy", "last"),
        )
        sections.append(_answer(
            "Does TFRecord reduce TensorFlow epoch time?",
            "Measured training comparison (same model, split, seed):\n\n"
            + _md_table(agg.reset_index()),
        ))
    else:
        sections.append(_answer(
            "Does TFRecord reduce TensorFlow epoch time?",
            "Not yet measured: run `train` to collect data.",
        ))

    # Q8/Q9: cost
    if runs["estimated_cost"].notna().any():
        cost_summary = runs.dropna(subset=["estimated_cost"]).groupby(
            ["execution_mode", "partitions"]
        ).agg(
            mean_estimated_cost=("estimated_cost", "mean"),
            mean_cost_per_1k=("estimated_cost_per_1k_images", "mean"),
            mean_throughput=("throughput_images_per_second", "mean"),
        ).reset_index()
        best = cost_summary.loc[
            (cost_summary["mean_throughput"] / cost_summary["mean_estimated_cost"]).idxmax()
        ]
        sections.append(_answer(
            "What is the estimated cost-performance trade-off, and which configuration "
            "gives the best throughput per unit cost?",
            "_All cost figures are estimates from user-supplied hourly prices and cover "
            "job runtime only (no cluster startup/idle time)._\n\n"
            + _md_table(cost_summary)
            + f"\n\nBest estimated throughput per unit cost: {best.execution_mode} with "
            f"{int(best.partitions)} partitions.",
        ))
    else:
        sections.append(_answer(
            "What is the estimated cost-performance trade-off?",
            "Not estimated: no hourly prices were supplied "
            "(`--worker-hourly-cost` / `--master-hourly-cost`).",
        ))

    sections.append("## Limitations\n")
    sections.append(
        "- These results were collected on the specific hardware recorded in "
        "`reports/run_metadata/` and may not transfer to other machines or clusters.\n"
        "- The flowers dataset (~3,7k images) is small; conclusions about scaling "
        "behaviour may not generalise to much larger workloads.\n"
        "- Repeated-run intervals are bootstrap resampling intervals over a small "
        "number of repeats, not formal population confidence intervals.\n"
        "- Dataproc cluster startup time is excluded from runtime and cost figures "
        "unless stated otherwise.\n"
    )

    if figures:
        sections.append("## Figures\n")
        sections.extend(f"![{p.stem}]({Path(figures_dir).name}/{p.name})\n" for p in figures)

    out = Path(output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections))
    logger.info("report written to %s (%d figures)", out, len(figures))
    return out
