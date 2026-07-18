"""Robust image preprocessing shared by the local and Spark pipelines.

Uses Pillow so that preprocessing has no TensorFlow dependency and is
byte-for-byte identical between execution modes: decode JPEG -> convert to
RGB -> aspect-preserving resize -> centre-crop -> re-encode as JPEG.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

JPEG_QUALITY = 95


class PreprocessingError(Exception):
    """Raised when an image cannot be preprocessed."""


@dataclass
class ProcessingStats:
    """Counts of per-file outcomes for a preprocessing run."""

    processed: int = 0
    failed: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)

    def record_failure(self, path: str, error: Exception, policy: str) -> None:
        if policy == "raise":
            raise PreprocessingError(f"failed to preprocess {path}: {error}") from error
        if policy == "log":
            logger.warning("failed to preprocess %s: %s", path, error)
            self.failed += 1
            self.failures.append(path)
        else:  # skip
            self.skipped += 1


def preprocess_image_bytes(
    data: bytes,
    target_height: int,
    target_width: int,
) -> bytes:
    """Decode, convert to RGB, aspect-preserving resize, centre-crop, re-encode.

    Raises :class:`PreprocessingError` for empty, corrupted or unsupported input.
    """
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"target dimensions must be positive, got {target_height}x{target_width}")
    if not data:
        raise PreprocessingError("empty file")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PreprocessingError(f"cannot decode image: {exc}") from exc

    if img.mode != "RGB":  # greyscale, RGBA, palette, CMYK -> RGB
        img = img.convert("RGB")

    w, h = img.size
    if w == 0 or h == 0:
        raise PreprocessingError(f"degenerate image size {w}x{h}")

    # Scale so the image covers the target box, preserving aspect ratio.
    scale = max(target_width / w, target_height / h)
    new_w = max(target_width, round(w * scale))
    new_h = max(target_height, round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    img = img.crop((left, top, left + target_width, top + target_height))

    if img.size != (target_width, target_height):
        raise PreprocessingError(
            f"output size {img.size} does not match target {(target_width, target_height)}"
        )

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue()


def preprocess_file(
    path: str,
    target_height: int,
    target_width: int,
    read_bytes=None,
) -> bytes:
    """Read a file and preprocess it. ``read_bytes`` is injectable for GCS backends."""
    if read_bytes is None:
        with open(path, "rb") as f:
            data = f.read()
    else:
        data = read_bytes(path)
    return preprocess_image_bytes(data, target_height, target_width)
