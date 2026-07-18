# Distributed Image ML Pipeline: Spark, TFRecord and GCP Benchmarking

**How do distributed preprocessing, Spark partitioning and TFRecord
serialisation affect the throughput and downstream training performance of an
image-based machine-learning pipeline?**

A reproducible benchmark and data-engineering case study: the same image
preprocessing (decode → RGB → aspect-preserving resize → centre-crop →
re-encode) runs as a local baseline and as a PySpark pipeline, writes
partitioned TFRecord shards, and feeds a downstream TensorFlow training
comparison — with repeated runs, statistics and automated reporting.

> **Historical note:** this repository originated as MSc Big Data coursework
> (INM432, 2023) and has since been refactored into a reproducible portfolio
> project. The original coursework is preserved untouched in
> [`archive/original_coursework/`](archive/original_coursework/).

## Key findings

**Results pending reproducible benchmark execution.** No performance numbers
appear in this repository until they are measured; run the benchmarks below,
then `report` regenerates [`reports/benchmark_report.md`](reports/) from the
recorded CSVs.

## Architecture

```
Raw JPEG
   ↓
Local / Spark preprocessing
   ↓
Resize + crop + recompress
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
Source: `http://download.tensorflow.org/example_images/flower_photos.tgz`
(TensorFlow team). Images are Flickr photos under individual (mostly CC-BY)
licences listed in the archive's `LICENSE.txt`; this repository does not
redistribute them. See the [data card](docs/DATA_CARD.md).

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .                                   # core (config, preprocessing)
pip install -e ".[spark,tensorflow]"               # local + Spark + TFRecord pipelines
pip install -e ".[dev]"                            # tests + lint
pip install -e ".[spark,tensorflow,cloud,dev,notebook]"  # everything
```

Notes:

- The `tensorflow` extra installs `tensorflow-cpu` on Linux/Windows and plain
  `tensorflow` on macOS (where `tensorflow-cpu` wheels are not published).
- Spark needs a Java runtime (JRE 11+ recommended).
- The `stats` extra adds scipy + matplotlib for statistical tests and figures.

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

For GCP, configure via environment (no personal identifiers are hardcoded):

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
  intervals over repeats (documented; not population confidence intervals).
- **Partitions:** configurable sweep (e.g. 1 2 4 8).
- **Workers:** recorded only when actually known (Dataproc); never conflated
  with partitions.
- **Dataset sizes:** deterministic seeded subsampling (`--sample-rate`,
  full dataset = 1.0 is the default); sampled runs are always labelled.
- **Metrics:** runtime, throughput (images/s), processed/failed/skipped
  counts, output size, per-run reproducibility metadata
  (`reports/run_metadata/`).

## Testing and CI

```bash
pip install -e ".[spark,tensorflow,dev,stats]"
ruff check src tests
pytest --cov=distributed_image_pipeline
```

GitHub Actions runs on every push/PR: Ruff, pytest + coverage, a package
install check, and end-to-end smoke tests (local preprocessing, local-mode
Spark, TFRecord round trip, tiny benchmark + report, one-epoch training) on a
synthetic dataset — **no GCP credentials required**. A separate
`workflow_dispatch` workflow exists for optional cloud benchmarks and needs
repository secrets.

## Repository structure

```text
src/distributed_image_pipeline/   # packaged pipeline (config, labels,
                                  # preprocessing, tfrecords, local/spark
                                  # pipelines, benchmark, metrics, training,
                                  # report, cli)
tests/                            # unit + integration tests (no GCP needed)
notebooks/portfolio_analysis.ipynb# analysis of generated results
notebooks/archive/                # sanitised original coursework notebook
archive/original_coursework/      # untouched MSc coursework (notebook, PDF)
docs/                             # data card, experiment design, architecture
reports/                          # tables, figures, run metadata, reports
data/                             # local datasets and split manifests (untracked)
```

## Limitations

- Small dataset (~3,700 images): fixed overheads (Spark startup, TF graph
  tracing) weigh more than they would at scale; results may not generalise
  to much larger workloads.
- Cloud experiments depend on user-provisioned Dataproc resources; Dataproc
  cluster startup overhead is excluded from job runtimes unless stated.
- All results are specific to the recorded hardware
  (`reports/run_metadata/`).

## Licence

Code is MIT-licensed (see [LICENSE](LICENSE)). Dataset images retain their
original licences.
