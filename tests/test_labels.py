import pytest

from distributed_image_pipeline.labels import (
    DEFAULT_CLASSES,
    label_from_path,
    label_to_index,
    validate_classes,
)


def test_valid_labels():
    for i, cls in enumerate(DEFAULT_CLASSES):
        assert label_to_index(cls, DEFAULT_CLASSES) == i


def test_str_labels_accepted():
    assert label_to_index("roses", DEFAULT_CLASSES) == 2


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="unknown label"):
        label_to_index(b"orchid", DEFAULT_CLASSES)


def test_unknown_label_never_maps_to_zero():
    # Regression: the coursework's argmax approach silently returned 0 here.
    with pytest.raises(ValueError):
        label_to_index(b"not-a-flower", DEFAULT_CLASSES)


def test_duplicate_classes_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        label_to_index(b"daisy", [b"daisy", b"roses", b"daisy"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_classes([b"a", b"a"])


def test_empty_class_list_rejected():
    with pytest.raises(ValueError, match="empty"):
        label_to_index(b"daisy", [])
    with pytest.raises(ValueError):
        validate_classes([])


def test_empty_label_in_classes_rejected():
    with pytest.raises(ValueError):
        validate_classes([b"daisy", b""])


def test_label_from_path():
    assert label_from_path("gs://bucket/flowers/tulips/img1.jpg") == "tulips"
    assert label_from_path("data/raw/daisy/x.jpg") == "daisy"
    with pytest.raises(ValueError):
        label_from_path("file.jpg")
