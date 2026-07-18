import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from distributed_image_pipeline.benchmark import RUN_COLUMNS  # noqa: E402
from distributed_image_pipeline.report import generate_report  # noqa: E402


def synthetic_runs(tmp_path, modes=("local", "spark"), partitions=(1, 2), repeats=3):
    rng = np.random.default_rng(0)
    rows = []
    for mode in modes:
        for p in partitions:
            for r in range(repeats):
                rt = float((10 if mode == "local" else 6) / p + rng.uniform(0, 0.3))
                rows.append({
                    "run_id": f"{mode}{p}{r}",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "git_commit": "abc123",
                    "execution_mode": mode,
                    "input_format": "jpeg",
                    "dataset_size": 100,
                    "input_files": 100,
                    "processed_files": 100,
                    "failed_files": 0,
                    "skipped_files": 0,
                    "partitions": p,
                    "workers": None,
                    "target_size": "32x32",
                    "sample_rate": 1.0,
                    "sampled": False,
                    "seed": 42,
                    "repeat": r,
                    "runtime_seconds": rt,
                    "throughput_images_per_second": 100 / rt,
                    "output_files": p,
                    "output_size_mb": 1.0,
                    "estimated_cost": None,
                    "estimated_cost_per_1k_images": None,
                })
    df = pd.DataFrame(rows, columns=RUN_COLUMNS)
    csv = tmp_path / "runs.csv"
    df.to_csv(csv, index=False)
    return csv


def test_generate_report(tmp_path):
    csv = synthetic_runs(tmp_path)
    out = generate_report(
        benchmark_csv=str(csv),
        output_md=str(tmp_path / "report.md"),
        figures_dir=str(tmp_path / "figures"),
    )
    text = out.read_text()
    assert "Benchmark report" in text
    assert "Summary by configuration" in text
    # measured questions answered, unmeasured explicitly marked
    assert "mean local runtime" in text
    assert "Not yet measured" in text  # e.g. worker scaling, IO, training
    assert "not a population confidence interval" in text.lower() or "resampling" in text
    figures = list((tmp_path / "figures").glob("*.png"))
    names = {f.name for f in figures}
    assert "runtime_by_configuration.png" in names
    assert "throughput_by_configuration.png" in names
    assert "runtime_vs_partitions.png" in names
    # no worker data -> no fabricated worker-scaling figures
    assert "speedup_vs_workers.png" not in names


def test_generate_report_empty_csv(tmp_path):
    csv = tmp_path / "empty.csv"
    pd.DataFrame(columns=RUN_COLUMNS).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        generate_report(str(csv), str(tmp_path / "r.md"), str(tmp_path / "f"))


def test_generate_report_with_training_and_io(tmp_path):
    csv = synthetic_runs(tmp_path)
    io_csv = tmp_path / "io.csv"
    pd.DataFrame([
        {"input_format": fmt, "repeat": r, "samples_per_second": sps + r,
         "epoch_time_seconds": 1.0, "batch_size": 8, "samples": 100}
        for fmt, sps in (("jpeg", 100.0), ("tfrecord", 300.0)) for r in range(3)
    ]).to_csv(io_csv, index=False)
    train_csv = tmp_path / "train.csv"
    pd.DataFrame([
        {"input_format": fmt, "epoch": e, "training_time_seconds": t + e,
         "samples_per_second": 50.0, "validation_accuracy": 0.4, "validation_loss": 1.2}
        for fmt, t in (("jpeg", 5.0), ("tfrecord", 4.0)) for e in range(2)
    ]).to_csv(train_csv, index=False)

    out = generate_report(
        benchmark_csv=str(csv),
        output_md=str(tmp_path / "report.md"),
        figures_dir=str(tmp_path / "figures"),
        training_csv=str(train_csv),
        io_csv=str(io_csv),
    )
    text = out.read_text()
    assert "Measured input throughput" in text
    assert "Measured training comparison" in text
    assert (tmp_path / "figures" / "jpeg_vs_tfrecord_throughput.png").exists()
    assert (tmp_path / "figures" / "training_epoch_time.png").exists()
