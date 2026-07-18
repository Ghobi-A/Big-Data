import io
import random

import pytest
from PIL import Image

CLASSES = [b"daisy", b"dandelion", b"roses", b"sunflowers", b"tulips"]


def make_jpeg_bytes(width=64, height=48, color=(255, 0, 0), mode="RGB"):
    img = Image.new(mode, (width, height), color if mode == "RGB" else 128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def flowers_dir(tmp_path):
    """A tiny synthetic dataset with the five flower classes (4 images each)."""
    rng = random.Random(0)
    root = tmp_path / "flowers"
    for cls in CLASSES:
        d = root / cls.decode()
        d.mkdir(parents=True)
        for i in range(4):
            w, h = rng.randint(40, 120), rng.randint(40, 120)
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            (d / f"img_{i}.jpg").write_bytes(make_jpeg_bytes(w, h, color))
    return root


@pytest.fixture(scope="session")
def spark_context():
    pyspark = pytest.importorskip("pyspark")
    sc = pyspark.SparkContext.getOrCreate(
        pyspark.SparkConf().setMaster("local[2]").setAppName("tests")
    )
    yield sc
    sc.stop()
