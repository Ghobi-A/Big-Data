# Architecture

```text
Raw JPEG (local dir or gs://)
   ↓
Local / Spark preprocessing        src/distributed_image_pipeline/
   ↓                                 preprocessing.py  (shared Pillow logic)
Resize + crop + recompress           local_pipeline.py / spark_pipeline.py
   ↓
TFRecord shards                      tfrecords.py (documented schema)
   ↓
TensorFlow input pipeline            tf_input.py / training.py
   ↓
Image classification (small CNN)
   ↓
Benchmark + training metrics         benchmark.py / metrics.py / report.py
```

## Modules

- **`config.py`** — validated frozen `PipelineConfig`; values resolved from
  CLI overrides > environment variables (`GCP_PROJECT_ID`, `GCS_INPUT_URI`,
  …) > optional JSON config file. No personal GCP defaults exist anywhere.
- **`labels.py`** — strict label↔index mapping; unknown labels raise and
  duplicate/empty class lists are rejected.
- **`preprocessing.py`** — Pillow-based decode → RGB → aspect-preserving
  resize → centre-crop → JPEG re-encode, with `raise`/`skip`/`log` error
  policies and per-file outcome counters. No TensorFlow dependency, so the
  same bytes-level preprocessing logic runs in local and Spark execution.
- **`local_pipeline.py`** — single-process baseline. Lists files
  deterministically, optionally samples with a seed, and writes TFRecord shards
  or processed JPEGs. Returns `PipelineRunResult`.
- **`spark_pipeline.py`** — parallelises the sorted file list into N
  partitions; each executor partition preprocesses records and streams them
  directly into one TFRecord shard. No image payloads are collected to the
  driver and no full partition of processed image bytes is accumulated before
  writing.
- **`tfrecords.py`** — Example schema (`image`, `class`, `height`, `width`,
  `source`), serialisation/parsing, shard writing and record counting. The
  `source` feature keeps only `<label>/<basename>` so no personal paths leak.
- **`tf_input.py`** — TensorFlow input helpers shared by benchmark/training
  code. Raw JPEGs use RGB decode, aspect-preserving resize and centre-crop;
  TFRecord discovery supports direct files, directories and globs.
- **`benchmark.py`** — grid runner (modes × partitions × sample rates ×
  repeats) appending rows to `reports/tables/benchmark_runs.csv`; raw JPEG vs
  TFRecord `tf.data` input benchmark; cost estimation from user-supplied
  prices only.
- **`metrics.py`** — per-configuration aggregation, bootstrap resampling
  intervals, speedup/parallel-efficiency (worker counts only), partition
  scaling (kept separate), Mann-Whitney + bootstrap comparisons.
- **`training.py`** — leakage-safe stratified split persisted to
  `data/splits/`; identical small CNN trained from raw-JPEG and TFRecord input
  pipelines using the same split, seed, optimiser, batch size and epoch count.
  The raw-JPEG path applies the same geometric transform used when TFRecords
  are created so storage-format comparisons do not also change image geometry.
- **`report.py`** — markdown report + matplotlib figures generated strictly
  from measured CSVs; unanswerable questions are marked "not yet measured".
- **`results.py`** — `PipelineRunResult` and reproducibility metadata
  (git SHA, library versions, optional Dataproc metadata).
- **`cli.py`** — one argparse CLI exposing every stage.

## Execution environments

- **Local:** everything runs on one machine; Spark in `local[*]` mode.
- **Dataproc:** submit `preprocess-spark`/`benchmark` as PySpark jobs. The
  package must be distributed to executors and TensorFlow installed on the
  cluster. Cluster identity comes from configuration; nothing is hardcoded.

## Data flow guarantees

- Local and Spark preprocessing paths share preprocessing code, label mapping,
  sampling and output schema.
- JPEG and TFRecord input comparisons use equivalent target image geometry.
- Sampling is deterministic (sorted files + seeded RNG) and always recorded.
- Benchmark rows never contain fabricated values: unknown fields are null.
