"""Downstream training comparison: raw JPEG vs TFRecord input pipelines.

The classifier is deliberately small — the question is whether the input
format changes training throughput, not model accuracy. Both pipelines use
the same split, model, seed, optimiser, batch size and epoch count.

The raw-JPEG path applies the same geometric preprocessing semantics used when
TFRecords are created: RGB decode, aspect-preserving resize, then centre-crop.
The TFRecord path reads the already-preprocessed image. This keeps the
comparison focused on storage/input strategy rather than different image
transformations.

Split leakage safety: the train/validation split is derived from the *source
JPEG file list* (deterministic, stratified by class) and persisted as CSV
manifests. The TFRecord pipeline filters records by their stored ``source``
key (``<label>/<basename>``) against the same manifests, so both formats see
identical example sets.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .labels import DEFAULT_CLASSES, label_from_path, label_to_index, validate_classes
from .local_pipeline import list_image_files
from .tf_input import list_tfrecord_files, load_and_preprocess_jpeg
from .tfrecords import source_key

logger = logging.getLogger(__name__)


@dataclass
class TrainingSpec:
    jpeg_input: str
    tfrecord_input: str
    classes: tuple = DEFAULT_CLASSES
    epochs: int = 3
    batch_size: int = 32
    target_size: tuple[int, int] = (192, 192)
    seed: int = 42
    validation_fraction: float = 0.2
    splits_dir: str = "data/splits"
    results_csv: str = "reports/tables/training_runs.csv"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not (0 < self.validation_fraction < 1):
            raise ValueError("validation_fraction must be in (0, 1)")


def make_stratified_split(
    files: list[str],
    classes,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic per-class (stratified) split with no overlap."""
    valid_classes = validate_classes(classes)
    rows = []
    for path in sorted(files):
        label = label_from_path(path)
        rows.append({
            "path": path,
            "source": source_key(path),
            "label": label,
            "class_id": label_to_index(label, valid_classes),
        })
    df = pd.DataFrame(rows)
    rng = random.Random(seed)
    val_idx: list[int] = []
    for _, group in df.groupby("class_id"):
        indices = list(group.index)
        rng.shuffle(indices)
        n_val = max(1, round(len(indices) * validation_fraction))
        val_idx.extend(indices[:n_val])
    val_mask = df.index.isin(val_idx)
    train_df, val_df = df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)
    overlap = set(train_df.source) & set(val_df.source)
    if overlap:
        raise RuntimeError(f"train/validation overlap detected: {sorted(overlap)[:5]}")
    return train_df, val_df


def persist_split(
    train_df: pd.DataFrame, val_df: pd.DataFrame, splits_dir: str
) -> tuple[Path, Path]:
    d = Path(splits_dir)
    d.mkdir(parents=True, exist_ok=True)
    train_path, val_path = d / "train.csv", d / "validation.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    return train_path, val_path


def build_model(num_classes: int, target_size: tuple[int, int], seed: int):
    """A small CNN. Identical (same seed) for both input formats."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    h, w = target_size
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(h, w, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def _jpeg_dataset(df: pd.DataFrame, target_size, batch_size, shuffle, seed):
    import tensorflow as tf

    def load(path, label):
        image = load_and_preprocess_jpeg(path, target_size)
        return tf.cast(image, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].tolist(), df["class_id"].tolist())
    )
    if shuffle:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    return (
        ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _tfrecord_dataset(shards, manifest: pd.DataFrame, target_size, batch_size, shuffle, seed):
    import tensorflow as tf

    from .tfrecords import parse_example

    h, w = target_size
    wanted = tf.constant(sorted(manifest["source"].tolist()))

    def in_manifest(serialized):
        features = parse_example(serialized)
        return tf.reduce_any(tf.equal(features["source"], wanted))

    def to_pair(serialized):
        features = parse_example(serialized)
        img = tf.io.decode_jpeg(features["image"], channels=3)
        img = tf.cast(tf.ensure_shape(img, [h, w, 3]), tf.float32)
        return img, features["class"]

    ds = tf.data.TFRecordDataset(shards, num_parallel_reads=tf.data.AUTOTUNE).filter(in_manifest)
    if shuffle:
        ds = ds.shuffle(len(manifest), seed=seed, reshuffle_each_iteration=True)
    return (
        ds.map(to_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


class _EpochTimer:
    def __init__(self):
        self.epoch_times: list[float] = []

    def keras_callback(self):
        import tensorflow as tf

        timer = self

        class Callback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                timer._start = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                timer.epoch_times.append(time.perf_counter() - timer._start)

        return Callback()


def train_one_format(
    spec: TrainingSpec,
    input_format: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Train the shared model on one input format; returns per-epoch rows."""

    if input_format == "jpeg":
        train_ds = _jpeg_dataset(train_df, spec.target_size, spec.batch_size, True, spec.seed)
        val_ds = _jpeg_dataset(val_df, spec.target_size, spec.batch_size, False, spec.seed)
    elif input_format == "tfrecord":
        shards = list_tfrecord_files(spec.tfrecord_input)
        if not shards:
            raise FileNotFoundError(f"no TFRecord shards under {spec.tfrecord_input!r}")
        train_ds = _tfrecord_dataset(
            shards, train_df, spec.target_size, spec.batch_size, True, spec.seed
        )
        val_ds = _tfrecord_dataset(
            shards, val_df, spec.target_size, spec.batch_size, False, spec.seed
        )
    else:
        raise ValueError(f"unknown input_format {input_format!r}")

    model = build_model(len(spec.classes), spec.target_size, spec.seed)
    timer = _EpochTimer()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=spec.epochs,
        verbose=0,
        callbacks=[timer.keras_callback()],
    )
    n_train = len(train_df)
    rows = []
    for epoch in range(spec.epochs):
        epoch_time = timer.epoch_times[epoch]
        rows.append({
            "input_format": input_format,
            "epoch": epoch,
            "training_time_seconds": epoch_time,
            "samples_per_second": n_train / epoch_time if epoch_time > 0 else None,
            "validation_accuracy": float(history.history["val_accuracy"][epoch]),
            "validation_loss": float(history.history["val_loss"][epoch]),
            "train_samples": n_train,
            "validation_samples": len(val_df),
            "batch_size": spec.batch_size,
            "seed": spec.seed,
        })
    return pd.DataFrame(rows)


def train_and_compare(
    spec: TrainingSpec,
    formats: list[str] | None = None,
) -> pd.DataFrame:
    """Run the training comparison for the requested formats and persist results."""
    if not formats:
        formats = ["jpeg", "tfrecord"]
    files = list_image_files(spec.jpeg_input)
    if not files:
        raise FileNotFoundError(f"no JPEGs under {spec.jpeg_input!r}")
    train_df, val_df = make_stratified_split(
        files, spec.classes, spec.validation_fraction, spec.seed
    )
    persist_split(train_df, val_df, spec.splits_dir)
    logger.info(
        "split: %d train / %d validation (seed=%d)", len(train_df), len(val_df), spec.seed
    )

    frames = [train_one_format(spec, fmt, train_df, val_df) for fmt in formats]
    df = pd.concat(frames, ignore_index=True)
    path = Path(spec.results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return df
