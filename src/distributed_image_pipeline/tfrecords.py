"""TFRecord serialisation and parsing.

Schema (tf.train.Example features):

===============  ==============  =======================================
Feature          Type            Description
===============  ==============  =======================================
``image``        bytes (1)       JPEG-encoded image bytes
``class``        int64 (1)       integer class ID (index into class list)
``height``       int64 (1)       image height in pixels
``width``        int64 (1)       image width in pixels
``basename``     bytes (1)       original file basename only — directory
                                 components are stripped so no personal
                                 filesystem paths leak into records
===============  ==============  =======================================

TensorFlow is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

from .labels import label_to_index, validate_classes


def _tf():
    import tensorflow as tf

    return tf


def serialize_example(
    image_bytes: bytes,
    label: bytes | str,
    classes: Sequence[bytes],
    height: int,
    width: int,
    source_name: str = "",
) -> bytes:
    """Serialise one image + label into a tf.train.Example byte string.

    Unknown labels raise ``ValueError`` (strict mapping, no silent class 0).
    """
    tf = _tf()
    class_id = label_to_index(label, classes)
    basename = os.path.basename(source_name) if source_name else ""
    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        "class": tf.train.Feature(int64_list=tf.train.Int64List(value=[class_id])),
        "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "basename": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[basename.encode("utf-8")])
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def parse_example(serialized) -> dict:
    """Parse a serialised Example back into a feature dict (graph-mode safe)."""
    tf = _tf()
    spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "basename": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    return tf.io.parse_single_example(serialized, spec)


def parse_image_and_label(serialized):
    """Parse to a decoded ``(image, class_id)`` pair for training pipelines."""
    tf = _tf()
    features = parse_example(serialized)
    image = tf.io.decode_jpeg(features["image"], channels=3)
    return image, features["class"]


def write_tfrecord_shard(
    path: str,
    examples: Iterable[tuple[bytes, bytes | str, int, int, str]],
    classes: Sequence[bytes],
) -> int:
    """Write ``(image_bytes, label, height, width, source_name)`` tuples to one shard.

    Returns the number of records written.
    """
    tf = _tf()
    valid_classes = validate_classes(classes)
    count = 0
    with tf.io.TFRecordWriter(path) as writer:
        for image_bytes, label, height, width, source_name in examples:
            writer.write(
                serialize_example(image_bytes, label, valid_classes, height, width, source_name)
            )
            count += 1
    return count


def read_tfrecord_dataset(paths: Sequence[str]):
    """Build a parsed ``tf.data.Dataset`` of (image, class_id) from shard paths."""
    tf = _tf()
    ds = tf.data.TFRecordDataset(list(paths), num_parallel_reads=tf.data.AUTOTUNE)
    return ds.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)


def count_records(paths: Sequence[str]) -> int:
    tf = _tf()
    ds = tf.data.TFRecordDataset(list(paths))
    return int(ds.reduce(0, lambda acc, _: acc + 1).numpy())
