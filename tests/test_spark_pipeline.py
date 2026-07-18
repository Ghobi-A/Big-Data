import pytest

pytest.importorskip("pyspark")
pytest.importorskip("tensorflow")
pytestmark = [pytest.mark.spark, pytest.mark.tf]

from distributed_image_pipeline.config import PipelineConfig  # noqa: E402
from distributed_image_pipeline.spark_pipeline import run_spark_pipeline  # noqa: E402
from distributed_image_pipeline.tfrecords import count_records  # noqa: E402


def test_spark_pipeline_smoke(flowers_dir, tmp_path, spark_context):
    cfg = PipelineConfig(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "out"),
        partitions=2,
        target_height=48,
        target_width=48,
    )
    result = run_spark_pipeline(cfg, spark_context=spark_context)
    assert result.mode == "spark"
    assert result.processed_files == 20
    assert result.failed_files == 0
    assert result.partitions == 2
    assert result.output_files == 2
    assert count_records(result.output_paths) == 20
    assert result.throughput_images_per_second > 0


def test_spark_partition_count_respected(flowers_dir, tmp_path, spark_context):
    cfg = PipelineConfig(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "out4"),
        partitions=4,
        target_height=32,
        target_width=32,
    )
    result = run_spark_pipeline(cfg, spark_context=spark_context)
    assert result.output_files == 4
    assert result.processed_files == 20


def test_spark_deterministic_sampling_matches_local(flowers_dir, tmp_path, spark_context):
    from distributed_image_pipeline.local_pipeline import list_image_files, sample_files

    files = list_image_files(str(flowers_dir))
    expected = sample_files(files, 0.5, seed=11)
    cfg = PipelineConfig(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "sampled"),
        partitions=2,
        sample_rate=0.5,
        seed=11,
        target_height=32,
        target_width=32,
    )
    result = run_spark_pipeline(cfg, spark_context=spark_context)
    assert result.sampled
    assert result.input_files == len(expected)
    assert count_records(result.output_paths) == len(expected)
