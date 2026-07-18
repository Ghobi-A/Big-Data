"""Distributed image ML pipeline.

A reproducible benchmark and data-engineering case study: local vs Spark image
preprocessing, TFRecord serialisation, and downstream TensorFlow training.

Heavy dependencies (PySpark, TensorFlow) are imported lazily inside the modules
that need them, so `import distributed_image_pipeline` works with only the core
dependencies installed.
"""

__version__ = "0.1.0"
