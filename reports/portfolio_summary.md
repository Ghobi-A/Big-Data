# Portfolio summary — Distributed Image ML Pipeline

## Project summary

A reproducible data-engineering and ML systems case study that preprocesses a
five-class, ~3,700-image dataset with both a local baseline and a PySpark
pipeline, serialises outputs to TFRecord shards, and benchmarks how partitioning,
storage format and distributed execution affect preprocessing throughput and
downstream TensorFlow training performance.

## Technical stack

Python packaging (`pyproject.toml`, src layout) · PySpark · TensorFlow ·
Pillow · pandas · Google Cloud Dataproc + GCS (optional, configured via
environment) · pytest + coverage · Ruff · GitHub Actions · Docker · matplotlib

## Key challenges

- Distributed preprocessing with identical semantics to the local baseline
  (shared preprocessing code, deterministic sampling, same TFRecord schema)
  so benchmark comparisons are fair
- Raw-JPEG vs TFRecord input comparison using equivalent geometric image
  preprocessing rather than different resize strategies
- Cloud I/O against GCS with no hardcoded personal project or bucket
- Partition-count configuration and measurement, kept distinct from true
  worker scaling
- Deterministic, leakage-safe train/validation splits shared between raw
  JPEG and TFRecord input pipelines
- Streaming Spark partition output directly to TFRecord shards to avoid
  accumulating an entire partition of processed image bytes in executor memory
- Reproducibility: seeds, git SHA, library versions and, where available,
  Dataproc cluster metadata recorded per benchmark run

## Results

**Results pending reproducible benchmark execution.** This file intentionally
contains no performance numbers. Quantified results should be populated only
from executed benchmarks (`reports/tables/*.csv`) via
`python -m distributed_image_pipeline.cli report`, which writes
`reports/benchmark_report.md`.

## CV-ready bullets (pre-benchmark, verified by code and tests)

- Built a PySpark and TensorFlow image-processing pipeline targeting Google
  Cloud Dataproc to preprocess a ~3,700-image, five-class dataset for downstream
  ML workloads.
- Distributed JPEG decoding, aspect-preserving resizing, centre-cropping and
  recompression across Spark partitions, streaming outputs into partitioned
  TFRecord shards with a documented schema.
- Designed a reproducible benchmark framework comparing local and distributed
  preprocessing, Spark partition configurations, raw JPEG vs TFRecord I/O,
  and downstream TensorFlow training performance with repeated runs,
  statistical comparison and automated reporting.
- Added packaging, unit/integration tests, local Spark and TFRecord round-trip
  coverage, and GitHub Actions CI that runs without cloud credentials.

Quantified bullets such as "increased preprocessing throughput by X× versus the
local baseline" must be added only after running the benchmarks and should use
values generated from measured result files.
