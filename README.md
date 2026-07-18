# Distributed Image ML Pipeline: Spark, TFRecord and GCP Benchmarking

**How do distributed preprocessing, Spark partitioning and TFRecord
serialisation affect the throughput and downstream training performance of an
image-based machine-learning pipeline?**

A reproducible benchmark and data-engineering case study: the same image
preprocessing (decode → RGB → aspect-preserving resize → centre-crop →
re-encode) runs as a local baseline and as a PySpark pipeline, writes
partitioned TFRecord shards, and feeds downstream TensorFlow input and training
comparisons — with repeated runs, statistics and automated reporting.

## Key findings

**Results pending reproducible benchmark execution.** No performance numbers
appear in this repository until they are measured; run the benchmarks below,
then `report` regenerates [`reports/benchmark_report.md`](reports/) from the
recorded CSVs.

## Architecture

```text
Raw JPEG
   ↓
Local / Spark preprocessing
   ↓
Aspect-preserving resize + centre-crop + recompress
   ↓
TFRecord
   ↓
TensorFlow input pipeline
   ↓
Image classification
   ↓
Benchmark + training metrics
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) ·
[`docs/DATA_CARD.md`](docs/DATA_CARD.md)

## Dataset

The TensorFlow **flowers** dataset: **3,670 JPEG images** in **five classes**
(`daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`), labelled by directory.
The repository does not redistribute the images. Dataset provenance and usage
notes are documented in [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .
pip install -e ".[spark,tensorflow]"
pip install -e ".[dev]"
pip install -e ".[spark,tensorflow,cloud,dev,notebook,stats]"
```

Notes:

- The `tensorflow` extra installs `tensorflow-cpu` on Linux/Windows and plain
  `tensorflow` on macOS where `tensorflow-cpu` wheels are not published.
- Spark needs a Java runtime (JRE 11+ recommended).
- The `stats` extra adds SciPy + matplotlib for statistical tests and figures.

## Reproduction

Download and unpack the dataset, then:

```bash
# 1. Local baseline
python -m distributed_image_pipeline.cli preprocess-local \
  --input data/raw/flower_photos --output data/processed/tfrecords \
  --target-size 192 192 --partitions 8

# 2. Spark (local mode or against GCS/Dataproc data)
python -m distributed_image_pipeline.cli preprocess-spark \
  --input "data/raw/flower_photos" --output data/processed/tfrecords-spark \
  --partitions 8 --target-size 192 192

# 3. Repeated preprocessing benchmark (full dataset is the default)
python -m distributed_image_pipeline.cli benchmark \
  --input data/raw/flower_photos --output /tmp/bench-scratch \
  --modes local spark --partitions 1 2 4 8 --repeats 5 \
  --results reports/tables/benchmark_runs.csv \
  --summary reports/tables/benchmark_summary.csv

# 4. Raw JPEG vs TFRecord input benchmark
python -m distributed_image_pipeline.cli io-benchmark \
  --jpeg-input data/raw/flower_photos \
  --tfrecord-input data/processed/tfrecords --repeats 5

# 5. Downstream training comparison (same model, split and seed per format)
python -m distributed_image_pipeline.cli train \
  --jpeg-input data/raw/flower_photos \
  --tfrecord-input data/processed/tfrecords --epochs 3

# 6. Report + figures (generated only from measured CSVs)
python -m distributed_image_pipeline.cli report \
  --input reports/tables/benchmark_runs.csv \
  --output reports/benchmark_report.md \
  --training-csv reports/tables/training_runs.csv \
  --io-csv reports/tables/io_benchmark_runs.csv
```

For GCP, configure via environment; no personal project or bucket identifiers
are hardcoded:

```bash
export GCP_PROJECT_ID=your-project
export GCP_REGION=your-region
export GCS_INPUT_URI="gs://your-bucket/flowers/*/*.jpg"
export GCS_OUTPUT_URI=gs://your-bucket/tfrecords
export DATAPROC_CLUSTER=your-cluster
```

Cost estimation never uses hardcoded prices — pass `--worker-hourly-cost` /
`--master-hourly-cost` and figures are labelled estimates. True worker
scaling (speedup, parallel efficiency) is only computed when `--num-workers`
is recorded; partition sweeps on one machine are reported separately as
partition scaling.

## Benchmark design

- **Repeats:** 5 per configuration by default; mean/std/median/min/max are
  aggregated to `benchmark_summary.csv`. Intervals are bootstrap resampling
  intervals over repeats, not population confidence intervals.
- **Partitions:** configurable sweep (for example 1, 2, 4, 8).
- **Workers:** recorded only when actually known (for example Dataproc); never
  conflated with partitions.
- **Dataset sizes:** deterministic seeded subsampling (`--sample-rate`), with
  full dataset = 1.0 as the default. Sampled runs are always labelled.
- **Metrics:** runtime, throughput (images/s), processed/failed/skipped counts,
  output size, and per-run reproducibility metadata.
- **I/O fairness:** the raw-JPEG path applies the same geometric preprocessing
  semantics used to create the TFRecords — RGB decode, aspect-preserving resize
  and centre-crop — so the comparison does not mix storage format with a
  different resize strategy.
- **Training fairness:** JPEG and TFRecord training use the same persisted
  train/validation split, model architecture, seed, optimiser, batch size and
  epoch count.

## Implementation details

- Local and Spark execution share the same Pillow preprocessing function,
  strict label mapping and TFRecord schema.
- Spark partitions stream processed examples directly into their TFRecord
  shard instead of accumulating a full partition of image bytes in memory.
- TFRecord discovery supports a direct `.tfrec`/`.tfrecord` file, a directory,
  a recursive local glob, or a `gs://` glob.
- Sampling is deterministic from a sorted file list and explicit seed.
- Unknown labels raise errors rather than silently mapping to class zero.
- Unavailable benchmark fields remain null rather than being fabricated.

## Testing and CI

```bash
pip install -e ".[spark,tensorflow,dev,stats]"
ruff check src tests
pytest --cov=distributed_image_pipeline
```

GitHub Actions runs on every push/PR: Ruff, pytest + coverage, a package
install check, and end-to-end smoke tests covering local preprocessing,
local-mode Spark, TFRecord round trips, a tiny benchmark + report, and one-epoch
TensorFlow training on synthetic images. No GCP credentials are required for
standard CI. A separate `workflow_dispatch` workflow exists for optional cloud
benchmarks and requires user-provided repository secrets.

## Repository structure

```text
src/distributed_image_pipeline/   # packaged pipeline and benchmark code
tests/                            # unit + integration tests (no GCP required)
notebooks/portfolio_analysis.ipynb# analysis of generated benchmark results
docs/                             # data card, experiment design, architecture
reports/                          # tables, figures, run metadata, reports
data/                             # local datasets and split manifests (untracked)
```

## Limitations

- Small dataset (~3,700 images): fixed overheads such as Spark startup and
  TensorFlow graph tracing weigh more than they would at much larger scale.
- Local Spark benchmarks measure partition scaling on one machine, not true
  multi-worker distributed speedup.
- Cloud experiments depend on user-provisioned Dataproc resources; cluster
  startup overhead is excluded from job runtimes unless explicitly recorded.
- Hardware and runtime environment can materially affect throughput, so all
  measured results should be interpreted alongside the recorded run metadata.
- JPEG and TFRecord training inputs use equivalent geometric transforms, but
  the preprocessing implementations use different image libraries at their
  respective stages; the experiment is designed to compare pipeline strategy,
  not claim byte-identical decoded tensors.

## Licence

Code is MIT-licensed (see [`LICENSE`](LICENSE)). Dataset images retain their
original licences and are not redistributed in this repository.
