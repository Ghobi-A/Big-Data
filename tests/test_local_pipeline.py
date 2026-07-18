import pytest

from distributed_image_pipeline.config import PipelineConfig
from distributed_image_pipeline.local_pipeline import (
    list_image_files,
    run_local_pipeline,
    sample_files,
)


def test_list_image_files_sorted(flowers_dir):
    files = list_image_files(str(flowers_dir))
    assert len(files) == 20
    assert files == sorted(files)


def test_sample_files_deterministic(flowers_dir):
    files = list_image_files(str(flowers_dir))
    s1 = sample_files(files, 0.5, seed=7)
    s2 = sample_files(files, 0.5, seed=7)
    s3 = sample_files(files, 0.5, seed=8)
    assert s1 == s2
    assert len(s1) == 10
    assert s1 != s3  # different seed selects a different subset


def test_full_dataset_is_default(flowers_dir):
    files = list_image_files(str(flowers_dir))
    assert sample_files(files, 1.0, seed=1) == files


def test_local_pipeline_jpeg_output(flowers_dir, tmp_path):
    cfg = PipelineConfig(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "out"),
        target_height=64,
        target_width=64,
    )
    result = run_local_pipeline(cfg, output_format="jpeg")
    assert result.mode == "local"
    assert result.processed_files == 20
    assert result.failed_files == 0
    assert result.output_files == 20
    assert result.runtime_seconds > 0
    assert result.throughput_images_per_second > 0
    assert not result.sampled


@pytest.mark.tf
def test_local_pipeline_tfrecord_output(flowers_dir, tmp_path):
    pytest.importorskip("tensorflow")
    from distributed_image_pipeline.tfrecords import count_records

    cfg = PipelineConfig(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "out"),
        partitions=3,
        target_height=48,
        target_width=48,
    )
    result = run_local_pipeline(cfg, output_format="tfrecord")
    assert result.processed_files == 20
    assert result.output_files == 3  # one shard per partition
    assert count_records(result.output_paths) == 20
    assert result.output_size_mb and result.output_size_mb > 0


def test_local_pipeline_corrupt_file_counted(flowers_dir, tmp_path):
    (flowers_dir / "roses" / "corrupt.jpg").write_bytes(b"not a jpeg")
    cfg = PipelineConfig(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "out"),
        target_height=32,
        target_width=32,
        error_policy="log",
    )
    result = run_local_pipeline(cfg, output_format="jpeg")
    assert result.processed_files == 20
    assert result.failed_files == 1


def test_local_pipeline_sampled_run_labelled(flowers_dir, tmp_path):
    cfg = PipelineConfig(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "out"),
        sample_rate=0.5,
        seed=3,
        target_height=32,
        target_width=32,
    )
    result = run_local_pipeline(cfg, output_format="jpeg")
    assert result.sampled
    assert result.sample_rate == 0.5
    assert result.seed == 3
    assert result.input_files == 10


def test_local_pipeline_empty_input_raises(tmp_path):
    cfg = PipelineConfig(
        input_uri=str(tmp_path / "nothing"),
        output_uri=str(tmp_path / "out"),
    )
    with pytest.raises(FileNotFoundError):
        run_local_pipeline(cfg, output_format="jpeg")
