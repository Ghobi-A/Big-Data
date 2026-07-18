import pytest

tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from distributed_image_pipeline.labels import DEFAULT_CLASSES  # noqa: E402
from distributed_image_pipeline.tfrecords import (  # noqa: E402
    count_records,
    parse_example,
    parse_image_and_label,
    read_tfrecord_dataset,
    serialize_example,
    write_tfrecord_shard,
)

from .conftest import make_jpeg_bytes  # noqa: E402


def test_serialize_parse_roundtrip():
    img = make_jpeg_bytes(192, 192)
    raw = serialize_example(img, b"roses", DEFAULT_CLASSES, 192, 192, "/home/user/roses/x.jpg")
    parsed = parse_example(raw)
    assert parsed["class"].numpy() == 2
    assert parsed["height"].numpy() == 192
    assert parsed["source"].numpy() == b"roses/x.jpg"  # only label/basename, no personal paths
    image, label = parse_image_and_label(raw)
    assert image.shape == (192, 192, 3)
    assert int(label.numpy()) == 2


def test_serialize_unknown_label_raises():
    with pytest.raises(ValueError, match="unknown label"):
        serialize_example(make_jpeg_bytes(), b"orchid", DEFAULT_CLASSES, 48, 64)


def test_single_example_shard(tmp_path):
    path = str(tmp_path / "one.tfrec")
    n = write_tfrecord_shard(
        path, [(make_jpeg_bytes(32, 32), b"daisy", 32, 32, "a.jpg")], DEFAULT_CLASSES
    )
    assert n == 1
    assert count_records([path]) == 1


def test_multiple_examples_and_shards(tmp_path):
    paths = []
    for shard in range(3):
        path = str(tmp_path / f"s{shard}.tfrec")
        examples = [
            (make_jpeg_bytes(32, 32), DEFAULT_CLASSES[i], 32, 32, f"f{shard}_{i}.jpg")
            for i in range(5)
        ]
        assert write_tfrecord_shard(path, examples, DEFAULT_CLASSES) == 5
        paths.append(path)
    assert count_records(paths) == 15
    labels = sorted(
        int(lbl.numpy()) for _, lbl in read_tfrecord_dataset(paths)
    )
    assert labels == sorted(list(range(5)) * 3)


def test_corrupted_record_fails_parsing(tmp_path):
    path = str(tmp_path / "bad.tfrec")
    with tf.io.TFRecordWriter(path) as w:
        w.write(b"garbage bytes that are not an Example")
    ds = tf.data.TFRecordDataset([path]).map(parse_example)
    with pytest.raises(tf.errors.InvalidArgumentError):
        list(ds)


def test_write_shard_rejects_duplicate_classes(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        write_tfrecord_shard(str(tmp_path / "x.tfrec"), [], [b"a", b"a"])
