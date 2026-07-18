"""Strict label handling.

The original coursework mapped labels to class indices with
``np.argmax(np.array(classes) == label)``, which silently maps any *unknown*
label to class 0. This module replaces that with strict, validated mapping.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_CLASSES: tuple[bytes, ...] = (b"daisy", b"dandelion", b"roses", b"sunflowers", b"tulips")


def validate_classes(classes: Sequence[bytes]) -> tuple[bytes, ...]:
    """Validate a class list: non-empty, no duplicates, bytes labels."""
    if not classes:
        raise ValueError("class list must not be empty")
    normalized = []
    for c in classes:
        if isinstance(c, str):
            c = c.encode("utf-8")
        if not isinstance(c, bytes):
            raise ValueError(f"class labels must be bytes or str, got {type(c).__name__}")
        if not c:
            raise ValueError("class labels must be non-empty")
        normalized.append(c)
    if len(set(normalized)) != len(normalized):
        dupes = sorted({c for c in normalized if normalized.count(c) > 1})
        raise ValueError(f"duplicate class labels: {dupes}")
    return tuple(normalized)


def label_to_index(label: bytes | str, classes: Sequence[bytes]) -> int:
    """Map a label to its class index, raising ``ValueError`` for unknown labels."""
    valid = validate_classes(classes)
    if isinstance(label, str):
        label = label.encode("utf-8")
    try:
        return valid.index(label)
    except ValueError:
        raise ValueError(
            f"unknown label {label!r}; known classes: {list(valid)}"
        ) from None


def label_from_path(path: str) -> str:
    """Extract the class label from a ``.../<label>/<file>.jpg`` style path."""
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"cannot derive label from path {path!r}: no parent directory")
    return parts[-2]
