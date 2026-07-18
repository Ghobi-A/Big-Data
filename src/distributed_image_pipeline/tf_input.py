"""TensorFlow input helpers shared by JPEG and TFRecord benchmarks.

The raw-JPEG path uses the same geometric preprocessing semantics as the
preprocessing pipeline that produces TFRecords: RGB decode, aspect-preserving
resize so the image covers the target box, then centre-crop to the requested
size. This keeps input-format comparisons focused on storage/preprocessing
strategy rather than on different image geometry.
"""

from __future__ import annotations

import glob
from pathlib import Path


def resize_and_center_crop_tensor(image, target_size: tuple[int, int]):
    """Resize while preserving aspect ratio, then centre-crop.

    ``image`` is a TensorFlow image tensor with shape ``[H, W, C]``.
    TensorFlow is imported lazily so the package core remains lightweight.
    """
    import tensorflow as tf

    target_height, target_width = target_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target dimensions must be positive")

    image = tf.convert_to_tensor(image)
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)

    scale = tf.maximum(
        tf.cast(target_width, tf.float32) / width,
        tf.cast(target_height, tf.float32) / height,
    )
    new_width = tf.maximum(
        target_width,
        tf.cast(tf.round(width * scale), tf.int32),
    )
    new_height = tf.maximum(
        target_height,
        tf.cast(tf.round(height * scale), tf.int32),
    )

    resized = tf.image.resize(image, [new_height, new_width], method="bilinear")
    offset_height = (new_height - target_height) // 2
    offset_width = (new_width - target_width) // 2
    cropped = tf.image.crop_to_bounding_box(
        resized,
        offset_height,
        offset_width,
        target_height,
        target_width,
    )
    return tf.ensure_shape(cropped, [target_height, target_width, 3])


def load_and_preprocess_jpeg(path, target_size: tuple[int, int]):
    """Read a JPEG, decode to RGB, and apply the shared geometric transform."""
    import tensorflow as tf

    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return resize_and_center_crop_tensor(image, target_size)


def list_tfrecord_files(input_uri: str) -> list[str]:
    """Resolve TFRecord files from a local file, directory, glob, or ``gs://`` URI."""
    if input_uri.startswith("gs://"):
        import tensorflow as tf

        return sorted(tf.io.gfile.glob(input_uri))

    path = Path(input_uri)
    if path.is_file():
        return [str(path)] if path.suffix.lower() in {".tfrec", ".tfrecord"} else []
    if path.is_dir():
        return sorted(
            str(item)
            for item in path.rglob("*")
            if item.suffix.lower() in {".tfrec", ".tfrecord"}
        )

    return sorted(
        item
        for item in glob.glob(input_uri, recursive=True)
        if Path(item).suffix.lower() in {".tfrec", ".tfrecord"}
    )
