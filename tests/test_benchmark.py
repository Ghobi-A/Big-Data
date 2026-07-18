import pytest

pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

import pandas as pd  # noqa: E402

from distributed_image_pipeline.benchmark import (  # noqa: E402
    RUN_COLUMNS,
    BenchmarkPlan,
    estimate_run_cost,
    run_benchmark,
    run_io_benchmark,
)
from distributed_image_pipeline.config import ClusterCost, PipelineConfig  # noqa: E402
from distributed_image_pipeline.local_pipeline import run_local_pipeline  # noqa: E402


def test_plan_validation():
    with pytest.raises(ValueError):
        BenchmarkPlan(input_uri="a", output_uri="b", repeats=0)
    with pytest.raises(ValueError):
        BenchmarkPlan(input_uri="a", output_uri="b", modes=["dask"])
    with pytest.raises(ValueError, match="local benchmark mode requires"):
        BenchmarkPlan(input_uri="a", output_uri="gs://bucket/scratch", modes=["local"])


def test_estimate_cost_none_without_prices():
    assert estimate_run_cost(60.0, 100, None) == (None, None)
    assert estimate_run_cost(60.0, 100, ClusterCost()) == (None, None)


def test_estimate_cost_with_prices():
    cost = ClusterCost(worker_hourly_cost=0.10, master_hourly_cost=0.12, num_workers=2)
    total, per_1k = estimate_run_cost(3600.0, 500, cost)
    assert total == pytest.approx(0.32)
    assert per_1k == pytest.approx(0.64)


def test_run_benchmark_local_smoke(flowers_dir, tmp_path):
    plan = BenchmarkPlan(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "scratch"),
        modes=["local"],
        partitions=[1, 2],
        repeats=2,
        target_height=32,
        target_width=32,
        results_csv=str(tmp_path / "runs.csv"),
        metadata_dir=str(tmp_path / "meta"),
    )
    df = run_benchmark(plan)
    assert len(df) == 4  # 2 partitions x 2 repeats
    assert list(df.columns) == RUN_COLUMNS
    assert (df["runtime_seconds"] > 0).all()
    assert (df["throughput_images_per_second"] > 0).all()
    assert (df["processed_files"] == 20).all()
    assert df["estimated_cost"].isna().all()  # no prices supplied -> null, not invented
    assert df["workers"].isna().all()  # unknown workers stay null
    on_disk = pd.read_csv(plan.results_csv)
    assert len(on_disk) == 4
    assert len(list((tmp_path / "meta").glob("*.json"))) == 4


def test_run_benchmark_appends(flowers_dir, tmp_path):
    kwargs = dict(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "scratch"),
        modes=["local"],
        partitions=[1],
        repeats=1,
        target_height=32,
        target_width=32,
        results_csv=str(tmp_path / "runs.csv"),
        metadata_dir=None,
    )
    run_benchmark(BenchmarkPlan(**kwargs))
    run_benchmark(BenchmarkPlan(**kwargs))
    assert len(pd.read_csv(kwargs["results_csv"])) == 2


def test_sampled_benchmark_labelled(flowers_dir, tmp_path):
    plan = BenchmarkPlan(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "scratch"),
        modes=["local"],
        partitions=[1],
        repeats=1,
        sample_rates=[0.5],
        target_height=32,
        target_width=32,
        results_csv=str(tmp_path / "runs.csv"),
        metadata_dir=None,
    )
    df = run_benchmark(plan)
    assert df["sampled"].all()
    assert (df["sample_rate"] == 0.5).all()
    assert (df["seed"] == 42).all()


def test_io_benchmark_accepts_direct_tfrecord_file(flowers_dir, tmp_path):
    cfg = PipelineConfig(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "tfrecords"),
        partitions=1,
        target_height=32,
        target_width=32,
    )
    result = run_local_pipeline(cfg, output_format="tfrecord")
    assert len(result.output_paths) == 1

    df = run_io_benchmark(
        jpeg_input=str(flowers_dir),
        tfrecord_input=result.output_paths[0],
        batch_size=4,
        repeats=1,
        target_size=(32, 32),
        results_csv=str(tmp_path / "io.csv"),
    )
    assert sorted(df.input_format.unique()) == ["jpeg", "tfrecord"]
    assert (df.samples > 0).all()
    assert (df.samples_per_second > 0).all()
