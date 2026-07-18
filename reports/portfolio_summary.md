# Portfolio summary — Distributed Image ML Pipeline

## Project summary

A reproducible data-engineering case study that preprocesses a five-class,
~3,700-image flower dataset with both a local baseline and a PySpark pipeline,
serialises the output to TFRecord shards, and benchmarks how partitioning,
storage format and distributed execution affect preprocessing throughput and
downstream TensorFlow training performance. Originally MSc Big Data
coursework, rebuilt as a tested, packaged, CI-verified project.

## Technical stack

Python packaging (`pyproject.toml`, src layout) · PySpark · TensorFlow ·
Pillow · pandas · Google Cloud Dataproc + GCS (optional, configured via
environment) · pytest + coverage · Ruff · GitHub Actions · Docker · matplotlib

## Key challenges

- Distributed preprocessing with identical semantics to the local baseline
  (shared preprocessing code, deterministic sampling, same TFRecord schema)
  so benchmark comparisons are fair
- Cloud I/O against GCS with no hardcoded personal project or bucket
- Partition-count configuration and measurement, kept distinct from true
  worker scaling
- Deterministic, leakage-safe train/validation splits shared between raw
  JPEG and TFRecord input pipelines
- Reproducibility: seeds, git SHA, library versions and (where available)
  Dataproc cluster metadata recorded per benchmark run

## Results

**Results pending reproducible benchmark execution.** This file intentionally
contains no performance numbers: quantified results are populated only from
executed benchmarks (`reports/tables/*.csv`) via
`python -m distributed_image_pipeline.cli report`, which writes
`reports/benchmark_report.md`. See that report for measured findings.

## CV-ready bullets (pre-benchmark, verified by code and tests)

- Built a PySpark and TensorFlow pipeline targeting Google Cloud Dataproc to
  preprocess a ~3,700-image, five-class image dataset for downstream ML
  workloads.
- Distributed JPEG decoding, resizing, centre-cropping and recompression
  across Spark partitions, serialising outputs into partitioned TFRecord
  files with a documented schema.
- Designed a reproducible benchmark framework comparing local and distributed
  preprocessing, Spark partition configurations, raw JPEG vs TFRecord I/O,
  and downstream TensorFlow training performance, with repeated runs,
  statistical comparison and automated reporting.
- Added packaging, unit/integration tests (including local Spark and
  TFRecord round-trip tests) and GitHub Actions CI that runs entirely
  without cloud credentials.

Quantified bullets (e.g. "increased preprocessing throughput by X× versus the
local baseline") must be added **only** after running the benchmarks, using
values from `reports/benchmark_report.md`.
