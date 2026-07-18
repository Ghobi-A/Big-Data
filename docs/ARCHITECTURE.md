# Architecture

```
Raw JPEG (local dir or gs://)
   ↓
Local / Spark preprocessing        src/distributed_image_pipeline/
   ↓                                 preprocessing.py  (shared Pillow logic)
Resize + crop + recompress           local_pipeline.py / spark_pipeline.py
   ↓
TFRecord shards                      tfrecords.py (documented schema)
   ↓
TensorFlow input pipeline            training.py (JPEG path vs TFRecord path)
   ↓
Image classification (small CNN)
   ↓
Benchmark + training metrics         benchmark.py / metrics.py / report.py
```

## Modules

- **`config.py`** — validated frozen `PipelineConfig`; values resolved from
  CLI overrides > environment variables (`GCP_PROJECT_ID`, `GCS_INPUT_URI`,
  …) > optional JSON config file. No personal GCP defaults exist anywhere.
- **`labels.py`** — strict label↔index mapping; unknown labels raise,
  duplicate/empty class lists rejected.
- **`preprocessing.py`** — Pillow-based decode → RGB → aspect-preserving
  resize → centre-crop → JPEG re-encode, with `raise`/`skip`/`log` error
  policies and per-file outcome counters. No TensorFlow dependency, so the
  exact same bytes-level logic runs in both execution modes.
- **`local_pipeline.py`** — single-process baseline. Lists files
  (sorted, deterministic), optionally samples deterministically, writes
  TFRecord shards (or processed JPEGs). Returns `PipelineRunResult`.
- **`spark_pipeline.py`** — parallelises the sorted file list into N
  partitions; each executor partition preprocesses its files and writes one
  TFRecord shard directly (no driver collection, no extra repartition
  shuffle). Returns the same `PipelineRunResult` shape.
- **`tfrecords.py`** — Example schema (`image`, `class`, `height`, `width`,
  `source`), serialisation/parsing, shard writing, record counting. The
  `source` feature keeps only `<label>/<basename>` so no personal paths leak.
- **`benchmark.py`** — grid runner (modes × partitions × sample rates ×
  repeats) appending rows to `reports/tables/benchmark_runs.csv`; raw JPEG vs
  TFRecord `tf.data` input benchmark; cost estimation from user-supplied
  prices only.
- **`metrics.py`** — per-configuration aggregation, bootstrap resampling
  intervals, speedup/parallel-efficiency (worker counts only), partition
  scaling (kept separate), Mann-Whitney + bootstrap comparisons.
- **`training.py`** — leakage-safe stratified split persisted to
  `data/splits/`; identical small CNN trained from a raw-JPEG pipeline and a
  TFRecord pipeline filtered to the same split manifests.
- **`report.py`** — markdown report + matplotlib figures generated strictly
  from measured CSVs; unanswerable questions are marked "not yet measured".
- **`results.py`** — `PipelineRunResult` and reproducibility metadata
  (git SHA, library versions, optional Dataproc metadata).
- **`cli.py`** — one argparse CLI exposing every stage.

## Execution environments

- **Local:** everything runs on one machine; Spark in `local[*]` mode.
- **Dataproc:** submit `preprocess-spark`/`benchmark` as PySpark jobs. The
  package must be distributed to executors (e.g. `--py-files` with a built
  wheel) and TensorFlow installed on the cluster (pip initialisation action).
  Cluster identity comes from configuration; nothing is hardcoded.

## Data flow guarantees

- Local and Spark paths share preprocessing code, label mapping, sampling and
  output schema, so benchmark comparisons are like-for-like.
- Sampling is deterministic (sorted files + seeded RNG) and always recorded.
- Benchmark rows never contain fabricated values: unknown fields are null.
