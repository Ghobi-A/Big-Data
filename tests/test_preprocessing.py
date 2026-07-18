import io

import pytest
from PIL import Image

from distributed_image_pipeline.preprocessing import (
    PreprocessingError,
    ProcessingStats,
    preprocess_image_bytes,
)

from .conftest import make_jpeg_bytes


def decoded(data):
    return Image.open(io.BytesIO(data))


@pytest.mark.parametrize("w,h", [(64, 48), (48, 64), (200, 50), (50, 200), (192, 192)])
def test_output_dimensions(w, h):
    out = preprocess_image_bytes(make_jpeg_bytes(w, h), 192, 192)
    img = decoded(out)
    assert img.size == (192, 192)
    assert img.mode == "RGB"


def test_non_square_target():
    out = preprocess_image_bytes(make_jpeg_bytes(300, 300), 96, 128)
    assert decoded(out).size == (128, 96)  # PIL size is (width, height)


def test_greyscale_converted_to_rgb():
    out = preprocess_image_bytes(make_jpeg_bytes(64, 64, mode="L"), 32, 32)
    assert decoded(out).mode == "RGB"


def test_rgba_png_converted():
    img = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = preprocess_image_bytes(buf.getvalue(), 32, 32)
    assert decoded(out).mode == "RGB"
    assert decoded(out).size == (32, 32)


def test_empty_file_raises():
    with pytest.raises(PreprocessingError, match="empty"):
        preprocess_image_bytes(b"", 32, 32)


def test_corrupt_file_raises():
    with pytest.raises(PreprocessingError):
        preprocess_image_bytes(b"not a jpeg at all", 32, 32)


def test_truncated_jpeg_raises():
    data = make_jpeg_bytes(64, 64)
    with pytest.raises(PreprocessingError):
        preprocess_image_bytes(data[: len(data) // 4], 32, 32)


def test_invalid_target_raises():
    with pytest.raises(ValueError):
        preprocess_image_bytes(make_jpeg_bytes(), 0, 32)


def test_stats_policy_raise():
    stats = ProcessingStats()
    with pytest.raises(PreprocessingError):
        stats.record_failure("x.jpg", ValueError("bad"), "raise")


def test_stats_policy_log_and_skip():
    stats = ProcessingStats()
    stats.record_failure("x.jpg", ValueError("bad"), "log")
    stats.record_failure("y.jpg", ValueError("bad"), "skip")
    assert stats.failed == 1
    assert stats.skipped == 1
    assert stats.failures == ["x.jpg"]
