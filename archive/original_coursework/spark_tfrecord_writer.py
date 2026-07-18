"""Spark TFRecord writer utilities."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import pyspark

DEFAULT_CLASSES = [b"daisy", b"dandelion", b"roses", b"sunflowers", b"tulips"]


def _bytestring_feature(list_of_bytestrings: Sequence[bytes]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints: Sequence[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def to_tfrecord(img_bytes: bytes, label: bytes, classes: Sequence[bytes]) -> tf.train.Example:
    class_num = int(np.argmax(np.array(classes) == label))
    feature = {
        "image": _bytestring_feature([img_bytes]),
        "class": _int_feature([class_num]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def decode_jpeg_and_label(filepath: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    bits = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(bits)
    label = tf.strings.split(tf.expand_dims(filepath, axis=-1), sep="/")
    label2 = label.values[-2]
    return image, label2


def resize_and_crop_image(
    input_layer: Tuple[tf.Tensor, tf.Tensor],
    target_size: Sequence[int],
) -> Tuple[tf.Tensor, tf.Tensor]:
    image, label = input_layer
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = target_size[1]
    th = target_size[0]
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(
        resize_crit < 1,
        lambda: tf.image.resize(image, [w * tw / w, h * tw / w]),
        lambda: tf.image.resize(image, [w * th / h, h * th / h]),
    )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label


def recompress_image(input_layer: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    image, label = input_layer
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label


def write_tfrecord_partition(
    index: int,
    data_partition: Iterable[Tuple[tf.Tensor, tf.Tensor]],
    gcs_output: str,
    classes: Sequence[bytes],
) -> List[str]:
    tfrecord_filename = f"{gcs_output}{index}.tfrec"
    with tf.io.TFRecordWriter(tfrecord_filename) as tfrecord_file:
        for img, lbl in data_partition:
            example = to_tfrecord(img.numpy(), lbl.numpy(), classes)
            tfrecord_file.write(example.SerializeToString())
    return [tfrecord_filename]


def write_tfrecords_with_spark(
    gcs_pattern: str,
    gcs_output: str,
    classes: Sequence[bytes] = DEFAULT_CLASSES,
    partitions: int = 2,
    sample_rate: float = 0.02,
    target_size: Sequence[int] = (192, 192),
    spark_context: Optional[pyspark.SparkContext] = None,
) -> List[str]:
    sc = spark_context or pyspark.SparkContext.getOrCreate()
    file_paths = tf.io.gfile.glob(gcs_pattern)
    file_rdd = sc.parallelize(file_paths, partitions)
    sampled = file_rdd.sample(False, sample_rate)
    decoded = sampled.map(decode_jpeg_and_label)
    resized = decoded.map(lambda item: resize_and_crop_image(item, target_size))
    recompressed = resized.map(recompress_image)
    partitioned = recompressed.repartition(partitions)
    tf_records_rdd = partitioned.mapPartitionsWithIndex(
        lambda index, partition: write_tfrecord_partition(index, partition, gcs_output, classes)
    )
    return tf_records_rdd.collect()


if __name__ == "__main__":
    PROJECT = "big-data-cw2-18002699"
    GCS_PATTERN = "gs://flowers-public/*/*.jpg"
    BUCKET = f"gs://{PROJECT}-storage"
    GCS_OUTPUT = f"{BUCKET}/tfrecords-jpeg-192x192-2/flowers"

    write_tfrecords_with_spark(GCS_PATTERN, GCS_OUTPUT)
