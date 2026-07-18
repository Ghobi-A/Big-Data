import pytest

pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

import pandas as pd  # noqa: E402

from distributed_image_pipeline.config import PipelineConfig  # noqa: E402
from distributed_image_pipeline.labels import DEFAULT_CLASSES  # noqa: E402
from distributed_image_pipeline.local_pipeline import (  # noqa: E402
    list_image_files,
    run_local_pipeline,
)
from distributed_image_pipeline.training import (  # noqa: E402
    TrainingSpec,
    build_model,
    make_stratified_split,
    persist_split,
    train_and_compare,
)


def test_spec_validation(tmp_path):
    with pytest.raises(ValueError):
        TrainingSpec(jpeg_input="a", tfrecord_input="b", epochs=0)
    with pytest.raises(ValueError):
        TrainingSpec(jpeg_input="a", tfrecord_input="b", validation_fraction=1.5)


def test_stratified_split_deterministic_and_disjoint(flowers_dir):
    files = list_image_files(str(flowers_dir))
    t1, v1 = make_stratified_split(files, DEFAULT_CLASSES, 0.25, seed=5)
    t2, v2 = make_stratified_split(files, DEFAULT_CLASSES, 0.25, seed=5)
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(v1, v2)
    assert set(t1.source).isdisjoint(set(v1.source))
    # stratified: every class appears in both splits
    assert set(v1.class_id) == set(range(5))
    assert set(t1.class_id) == set(range(5))
    assert len(t1) + len(v1) == 20


def test_persist_split(flowers_dir, tmp_path):
    files = list_image_files(str(flowers_dir))
    t, v = make_stratified_split(files, DEFAULT_CLASSES, 0.25, seed=5)
    tp, vp = persist_split(t, v, str(tmp_path / "splits"))
    assert tp.exists() and vp.exists()
    assert len(pd.read_csv(tp)) == len(t)


def test_model_input_shape():
    model = build_model(5, (32, 32), seed=0)
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 5)


def test_train_and_compare_tiny(flowers_dir, tmp_path):
    # produce TFRecords for the tfrecord path with the real pipeline
    cfg = PipelineConfig(
        input_uri=str(flowers_dir),
        output_uri=str(tmp_path / "tfrecords"),
        partitions=2,
        target_height=32,
        target_width=32,
    )
    run_local_pipeline(cfg, output_format="tfrecord")

    spec = TrainingSpec(
        jpeg_input=str(flowers_dir),
        tfrecord_input=str(tmp_path / "tfrecords"),
        epochs=1,
        batch_size=4,
        target_size=(32, 32),
        seed=1,
        validation_fraction=0.25,
        splits_dir=str(tmp_path / "splits"),
        results_csv=str(tmp_path / "training_runs.csv"),
    )
    df = train_and_compare(spec)
    assert sorted(df.input_format.unique()) == ["jpeg", "tfrecord"]
    assert len(df) == 2  # one epoch per format
    assert (df.training_time_seconds > 0).all()
    assert (df.samples_per_second > 0).all()
    assert df.validation_accuracy.between(0, 1).all()
    # both formats trained on the identical split
    assert df.train_samples.nunique() == 1
    assert df.validation_samples.nunique() == 1
    assert (tmp_path / "splits" / "train.csv").exists()
    assert pd.read_csv(spec.results_csv).shape[0] == 2
