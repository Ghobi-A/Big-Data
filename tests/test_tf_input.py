from pathlib import Path

import pytest


tf = pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from distributed_image_pipeline.tf_input import (  # noqa: E402
    list_tfrecord_files,
    resize_and_center_crop_tensor,
)


def test_resize_and_center_crop_tensor_shape():
    image = tf.zeros([40, 100, 3], dtype=tf.uint8)
    output = resize_and_center_crop_tensor(image, (32, 32))
    assert output.shape == (32, 32, 3)


def test_resize_and_center_crop_rejects_invalid_target():
    image = tf.zeros([40, 100, 3], dtype=tf.uint8)
    with pytest.raises(ValueError, match="target dimensions must be positive"):
        resize_and_center_crop_tensor(image, (0, 32))


def test_list_tfrecord_files_supports_file_directory_and_glob(tmp_path):
    a = tmp_path / "a.tfrec"
    b = tmp_path / "nested" / "b.tfrecord"
    b.parent.mkdir()
    a.write_bytes(b"")
    b.write_bytes(b"")
    (tmp_path / "ignore.txt").write_text("x")

    assert list_tfrecord_files(str(a)) == [str(a)]
    assert list_tfrecord_files(str(tmp_path)) == sorted([str(a), str(b)])
    assert list_tfrecord_files(str(tmp_path / "**" / "*.tfrec")) == [str(a)]


def test_list_tfrecord_files_rejects_non_tfrecord_file(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("x")
    assert list_tfrecord_files(str(path)) == []
