from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "reports" / "tables"
REPORT_PATH = ROOT / "reports" / "benchmark_report.md"

BENCHMARK_SUMMARY = TABLES_DIR / "benchmark_summary.csv"
BENCHMARK_RUNS = TABLES_DIR / "benchmark_runs.csv"
IO_RUNS = TABLES_DIR / "io_benchmark_runs.csv"
TRAINING_RUNS = TABLES_DIR / "training_runs.csv"

st.set_page_config(
    page_title="Distributed Image ML Benchmark",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_report(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


summary_df = load_csv(BENCHMARK_SUMMARY)
runs_df = load_csv(BENCHMARK_RUNS)
io_df = load_csv(IO_RUNS)
training_df = load_csv(TRAINING_RUNS)
report_text = load_report(REPORT_PATH)

required = {
    "benchmark summary": summary_df,
    "benchmark runs": runs_df,
    "I/O benchmark": io_df,
    "training benchmark": training_df,
}
missing = [name for name, df in required.items() if df.empty]
if missing:
    st.error(
        "The dashboard is missing measured result files: " + ", ".join(missing) + ". "
        "Run the full benchmark workflow and commit the generated CSVs under reports/tables/."
    )
    st.stop()

try:
    best_spark = best_mode_row(summary_df, "spark")
    best_local = best_mode_row(summary_df, "local")
    spark_speedup = throughput_speedup(best_spark, best_local)

    jpeg_io = io_format_mean(io_df, "jpeg")
    tfrecord_io = io_format_mean(io_df, "tfrecord")
    io_gain = io_speedup(jpeg_io, tfrecord_io)

    training_summary = training_summary_table(training_df)
    jpeg_train = training_format_row(training_summary, "jpeg")
    tf_train = training_format_row(training_summary, "tfrecord")
    epoch_delta = epoch_delta_pct(jpeg_train, tf_train)

    if "dataset_size" not in summary_df.columns:
        raise DashboardDataError("The benchmark summary is missing the dataset_size column.")
    processed_images = int(summary_df["dataset_size"].max())
    failed_files_total_events, failed_files_worst_run = failed_file_stats(runs_df)
    best_partitions = int(best_spark["partitions"])
except DashboardDataError as exc:
    st.error(
        f"Committed benchmark results are malformed or incomplete: {exc} "
        "Re-run the full benchmark workflow and commit the regenerated CSVs "
        "under reports/tables/."
    )
    st.stop()

if "git_commit" in runs_df.columns and runs_df["git_commit"].notna().any():
    commit_sha = str(runs_df["git_commit"].dropna().iloc[0])
else:
    commit_sha = "unknown"

st.title("Distributed Image ML Benchmark")
st.caption(
    "Measured benchmarking of local vs PySpark preprocessing, JPEG vs TFRecord input pipelines, "
    "and downstream TensorFlow training performance."
)

with st.sidebar:
    st.header("Experiment context")
    st.metric("Dataset", f"{processed_images:,} images")
    st.write("5 flower classes")
    st.write("192 × 192 target resolution")
    st.write("5 repeated preprocessing runs per configuration")
    st.write("Local-mode Spark on a GitHub Actions runner")
    st.caption(f"Benchmark commit: `{commit_sha[:12]}`")

(
    overview_tab,
    preprocessing_tab,
    io_tab,
    training_tab,
    explorer_tab,
    methodology_tab,
) = st.tabs(
    [
        "Overview",
        "Preprocessing",
        "JPEG vs TFRecord",
        "Training",
        "Experiment Explorer",
        "Methodology",
    ]
)

with overview_tab:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Images processed", f"{processed_images:,}")
    k2.metric("Best throughput", f"{best_spark['throughput_mean_ips']:,.0f} img/s")
    k3.metric("Spark throughput gain", f"{spark_speedup:.2f}×")
    k4.metric("TFRecord I/O gain", f"{io_gain:.2f}×")
    k5.metric("Best Spark partitions", f"{best_partitions}")
    k6.metric(
        "Failed files (worst run)",
        f"{failed_files_worst_run}",
        help=(
            "Highest failed-file count in any single run. Repeats reprocess the same "
            f"inputs, so the {failed_files_total_events} total failure events across all "
            "runs are not distinct files."
        ),
    )

    st.subheader("Key findings")
    st.markdown(
        f"""
- **PySpark reached the highest preprocessing throughput at {best_partitions} partitions:**
  **{best_spark['throughput_mean_ips']:,.1f} images/s** with a mean runtime of
  **{best_spark['runtime_mean_s']:.2f}s**.
- The strongest local baseline reached **{best_local['throughput_mean_ips']:,.1f} images/s**,
  giving the best Spark configuration a **{spark_speedup:.2f}× throughput advantage** in this
  GitHub Actions environment.
- TFRecord delivered **{tfrecord_io:,.1f} samples/s** versus **{jpeg_io:,.1f} samples/s** for
  raw JPEG input, a **{io_gain:.2f}× mean input-pipeline throughput increase**. JPEG throughput
  varied noticeably between repeats, so the mean gain should be read alongside the per-repeat
  spread shown in the JPEG vs TFRecord tab.
- The much larger I/O improvement translated into only a modest difference in end-to-end CNN
  epoch time, indicating that model computation remained a substantial bottleneck.
        """
    )
    st.info(
        "These are local-mode Spark partition-scaling results on one GitHub Actions runner. "
        "They are not multi-worker Dataproc scaling results."
    )

    overview_chart = summary_df.copy()
    fig = px.line(
        overview_chart,
        x="partitions",
        y="throughput_mean_ips",
        error_y="throughput_std_ips",
        color="execution_mode",
        markers=True,
        labels={
            "partitions": "Partitions",
            "throughput_mean_ips": "Mean throughput (images/s)",
            "execution_mode": "Execution mode",
        },
        title="Preprocessing throughput by partition count",
    )
    st.plotly_chart(fig, width="stretch")

with preprocessing_tab:
    st.subheader("Local vs PySpark preprocessing")
    mode_options = sorted(summary_df["execution_mode"].dropna().unique().tolist())
    selected_modes = st.multiselect(
        "Execution modes",
        mode_options,
        default=mode_options,
    )
    partition_options = sorted(summary_df["partitions"].dropna().astype(int).unique().tolist())
    selected_partitions = st.multiselect(
        "Partitions",
        partition_options,
        default=partition_options,
    )

    filtered_summary = summary_df[
        summary_df["execution_mode"].isin(selected_modes)
        & summary_df["partitions"].isin(selected_partitions)
    ]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            filtered_summary,
            x="partitions",
            y="runtime_mean_s",
            error_y="runtime_std_s",
            color="execution_mode",
            markers=True,
            labels={
                "partitions": "Partitions",
                "runtime_mean_s": "Mean runtime (s)",
                "execution_mode": "Execution mode",
            },
            title="Runtime vs partitions",
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = px.line(
            filtered_summary,
            x="partitions",
            y="throughput_mean_ips",
            error_y="throughput_std_ips",
            color="execution_mode",
            markers=True,
            labels={
                "partitions": "Partitions",
                "throughput_mean_ips": "Mean throughput (images/s)",
                "execution_mode": "Execution mode",
            },
            title="Throughput vs partitions",
        )
        st.plotly_chart(fig, width="stretch")

    run_filter = runs_df[
        runs_df["execution_mode"].isin(selected_modes)
        & runs_df["partitions"].isin(selected_partitions)
    ].copy()
    run_filter["configuration"] = (
        run_filter["execution_mode"].astype(str)
        + " / "
        + run_filter["partitions"].astype(str)
        + " partitions"
    )
    fig = px.box(
        run_filter,
        x="configuration",
        y="runtime_seconds",
        points="all",
        labels={"configuration": "Configuration", "runtime_seconds": "Runtime (s)"},
        title="Repeated-run runtime variability",
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        filtered_summary.sort_values(["execution_mode", "partitions"]),
        width="stretch",
        hide_index=True,
    )

with io_tab:
    st.subheader("JPEG vs TFRecord input pipeline")
    io_summary = (
        io_df.groupby("input_format", as_index=False)
        .agg(
            mean_samples_per_second=("samples_per_second", "mean"),
            std_samples_per_second=("samples_per_second", "std"),
            mean_epoch_time_seconds=("epoch_time_seconds", "mean"),
            mean_batch_latency_seconds=("mean_batch_latency_seconds", "mean"),
        )
    )

    a, b, c = st.columns(3)
    a.metric("JPEG input throughput", f"{jpeg_io:,.0f} samples/s")
    b.metric("TFRecord input throughput", f"{tfrecord_io:,.0f} samples/s")
    c.metric("Relative improvement (mean)", f"{io_gain:.2f}×")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            io_summary,
            x="input_format",
            y="mean_samples_per_second",
            error_y="std_samples_per_second",
            labels={
                "input_format": "Input format",
                "mean_samples_per_second": "Mean samples/s",
            },
            title="Input throughput",
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = px.box(
            io_df,
            x="input_format",
            y="epoch_time_seconds",
            points="all",
            labels={
                "input_format": "Input format",
                "epoch_time_seconds": "Full-dataset iteration time (s)",
            },
            title="Full-dataset iteration time across repeats",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        "**Interpretation:** TFRecord substantially increased standalone input-pipeline "
        "throughput on average, though JPEG timings vary noticeably between repeats. The "
        "downstream training experiment shows whether that I/O improvement materially reduces "
        "end-to-end training time once CNN computation is included."
    )

with training_tab:
    st.subheader("Downstream TensorFlow training comparison")
    st.dataframe(training_summary, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            training_df,
            x="epoch",
            y="training_time_seconds",
            color="input_format",
            markers=True,
            labels={
                "epoch": "Epoch",
                "training_time_seconds": "Training time (s)",
                "input_format": "Input format",
            },
            title="Training time by epoch",
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = px.line(
            training_df,
            x="epoch",
            y="validation_accuracy",
            color="input_format",
            markers=True,
            labels={
                "epoch": "Epoch",
                "validation_accuracy": "Validation accuracy",
                "input_format": "Input format",
            },
            title="Validation accuracy by epoch",
        )
        st.plotly_chart(fig, width="stretch")

    st.info(
        f"Mean epoch time changed from {jpeg_train['mean_epoch_s']:.2f}s with JPEG to "
        f"{tf_train['mean_epoch_s']:.2f}s with TFRecord ({epoch_delta:.1f}% faster on "
        "average). This is a mean across epochs of a single seeded training run per input "
        "format, not across repeated independent training experiments. Validation accuracy "
        "is shown as a sanity check rather than evidence that TFRecord improves predictive "
        "performance."
    )

with explorer_tab:
    st.subheader("Experiment explorer")
    datasets = {
        "Benchmark summary": summary_df,
        "Raw benchmark runs": runs_df,
        "JPEG vs TFRecord I/O runs": io_df,
        "Training runs": training_df,
    }
    selected_name = st.selectbox("Dataset", list(datasets))
    selected_df = datasets[selected_name]
    st.dataframe(selected_df, width="stretch", hide_index=True)
    st.download_button(
        "Download CSV",
        data=selected_df.to_csv(index=False).encode("utf-8"),
        file_name=selected_name.lower().replace(" ", "_") + ".csv",
        mime="text/csv",
    )

    if report_text:
        with st.expander("Measured benchmark report"):
            st.markdown(report_text)

with methodology_tab:
    st.subheader("Experiment methodology")
    st.markdown(
        """
```text
3,670 raw JPEG images
        ↓
Shared image preprocessing semantics
        ↓
┌──────────────────┬──────────────────┐
│ Local baseline   │ Local-mode Spark │
└──────────────────┴──────────────────┘
        ↓
Partitioned TFRecord shards
        ↓
JPEG vs TFRecord input benchmark
        ↓
Same CNN + same persisted train/validation split
        ↓
Runtime, throughput, validation metrics and repeated-run analysis
```

### Controls

- Five repeated preprocessing runs per configuration.
- Partition sweep: 1, 2, 4 and 8.
- Deterministic seed and full-dataset processing.
- Same target image geometry for the JPEG and TFRecord comparison.
- Same CNN architecture, optimiser, batch size, seed and data split for training comparisons.
- Partition scaling is kept distinct from true multi-worker scaling.

### Limitations

- The dataset is relatively small, so fixed Spark and TensorFlow overheads matter more than they
  would at larger scale.
- Spark results here are from local mode on a single GitHub Actions runner, not a multi-node
  Dataproc cluster.
- Throughput depends on the runner hardware and runtime environment.
- The training experiment is designed to measure pipeline efficiency, not state-of-the-art image
  classification accuracy.
        """
    )

st.divider()
st.caption(
    "All headline metrics are computed from committed measured benchmark CSVs. "
    "No benchmark values are hardcoded into the dashboard."
)
