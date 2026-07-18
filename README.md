# Distributed Image ML Pipeline: Spark, TFRecord and GCP Benchmarking

**How do distributed preprocessing, Spark partitioning and TFRecord serialisation affect the throughput and downstream training performance of an image-based machine-learning pipeline?**

A reproducible data-engineering and ML benchmarking project. The same image preprocessing semantics run through a local baseline and a PySpark pipeline, outputs are serialised to partitioned TFRecord shards, and downstream TensorFlow input/training performance is measured with repeated experiments and automated reporting.

## Measured results

The committed benchmark results were produced on a GitHub Actions runner using the full **3,670-image**, five-class flowers dataset.

- **Best preprocessing configuration:** local-mode PySpark with 8 partitions, **1,109.6 images/s** mean throughput and **3.31s** mean runtime.
- **Best local baseline:** 2 partitions, **486.8 images/s** mean throughput.
- **Best Spark throughput advantage:** **2.28×** over the strongest local baseline in the tested environment.
- **TFRecord input throughput:** **9,978.1 samples/s** vs **2,179.1 samples/s** for raw JPEG on average across 5 repeats, a **4.58× mean** increase. JPEG throughput varied substantially between repeats (roughly 1,430–2,690 samples/s), so the mean gain should be read alongside that spread; TFRecord throughput was highly stable.
- **Mean CNN epoch time:** **8.68s** with JPEG vs **8.54s** with TFRecord, averaged over the 3 epochs of a single seeded training run per input format (not repeated independent training experiments). The large standalone I/O gain translated into only a modest end-to-end training-time improvement, indicating model computation remained a substantial bottleneck.
- **Failed files:** 0 across the recorded preprocessing benchmark runs.

These Spark figures measure **partition scaling on one GitHub Actions runner**, not multi-worker Dataproc scaling.

Measured CSVs are committed under [`reports/tables/`](reports/tables/).

## Streamlit dashboard

A recruiter-facing interactive dashboard is available at [`dashboard/app.py`](dashboard/app.py). It calculates all headline metrics directly from the committed measured CSVs rather than hardcoding benchmark values.

Run locally:

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The dashboard includes:

- Overview KPIs and findings
- Local vs PySpark runtime and throughput by partition count
- JPEG vs TFRecord I/O comparison
- TensorFlow training-time and validation-metric comparison
- Raw experiment explorer with CSV downloads
- Methodology, controls and limitations

For Streamlit Community Cloud, deploy `dashboard/app.py` as the main file. No raw image dataset, Spark runtime, TensorFlow installation, cloud credentials or API keys are required to render the dashboard.

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
   ↓
Streamlit results dashboard
```

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) · [`docs/DATA_CARD.md`](docs/DATA_CARD.md)

## Dataset

The TensorFlow **flowers** dataset contains **3,670 JPEG images** across five classes: `daisy`, `dandelion`, `roses`, `sunflowers`, and `tulips`.

The repository does not redistribute the images. Dataset provenance and usage notes are documented in [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .
pip install -e ".[spark,tensorflow]"
pip install -e ".[dev]"
pip install -e ".[spark,tensorflow,cloud,dev,notebook,stats]"
```

For the results dashboard only:

```bash
pip install -r requirements.txt
```

Notes:

- The `tensorflow` extra installs `tensorflow-cpu` on Linux/Windows and plain `tensorflow` on macOS where `tensorflow-cpu` wheels are not published.
- Spark needs a Java runtime (JRE 11+ recommended).
- The `stats` extra adds SciPy + matplotlib for statistical tests and figures.

## Reproduce the benchmarks

Download and unpack the dataset, then:

```bash
# 1. Build TFRecord shards
python -m distributed_image_pipeline.cli preprocess-local \
  --input data/raw/flower_photos \
  --output data/processed/tfrecords \
  --target-size 192 192 \
  --partitions 8

# 2. Repeated local and local-Spark preprocessing benchmark
python -m distributed_image_pipeline.cli benchmark \
  --input data/raw/flower_photos \
  --output /tmp/bench-scratch \
  --modes local spark \
  --partitions 1 2 4 8 \
  --repeats 5 \
  --results reports/tables/benchmark_runs.csv \
  --summary reports/tables/benchmark_summary.csv

# 3. Raw JPEG vs TFRecord input benchmark
python -m distributed_image_pipeline.cli io-benchmark \
  --jpeg-input data/raw/flower_photos \
  --tfrecord-input data/processed/tfrecords \
  --repeats 5

# 4. Downstream training comparison
python -m distributed_image_pipeline.cli train \
  --jpeg-input data/raw/flower_photos \
  --tfrecord-input data/processed/tfrecords \
  --epochs 3

# 5. Generate report and figures
python -m distributed_image_pipeline.cli report \
  --input reports/tables/benchmark_runs.csv \
  --output reports/benchmark_report.md \
  --training-csv reports/tables/training_runs.csv \
  --io-csv reports/tables/io_benchmark_runs.csv
```

The repository also provides a manually triggered GitHub Actions workflow that runs the full local benchmark in a clean hosted environment.

## GCP configuration

Cloud identifiers are supplied through configuration rather than being hardcoded:

```bash
export GCP_PROJECT_ID=your-project
export GCP_REGION=your-region
export GCS_INPUT_URI="gs://your-bucket/flowers/*/*.jpg"
export GCS_OUTPUT_URI=gs://your-bucket/tfrecords
export DATAPROC_CLUSTER=your-cluster
```

Cost estimation never uses hardcoded prices. Hourly prices must be supplied explicitly, and cost outputs are labelled as estimates.

True worker scaling is kept separate from partition scaling. Worker speedup and parallel efficiency should only be reported when actual worker counts are recorded from a distributed cluster experiment.

## Benchmark design

- **Repeats:** 5 per preprocessing configuration.
- **Partitions:** 1, 2, 4 and 8 in the committed experiment.
- **Dataset:** full 3,670-image dataset for every recorded preprocessing run.
- **Metrics:** runtime, images/s, processed/failed/skipped files, output size and reproducibility metadata.
- **I/O fairness:** raw JPEG applies equivalent RGB decode, aspect-preserving resize and centre-crop semantics used to create TFRecords.
- **Training fairness:** JPEG and TFRecord training use the same persisted train/validation split, model architecture, seed, optimiser, batch size and epoch count.
- **Statistics:** repeated-run summaries include mean, standard deviation, median, min/max and resampling-based comparisons where supported.

## Implementation details

- Local and Spark execution share the same Pillow preprocessing logic, strict label mapping and TFRecord schema.
- Spark partitions stream processed examples directly into TFRecord shards instead of holding a full partition of processed images in memory.
- TFRecord discovery supports direct files, directories, recursive local globs and `gs://` globs.
- Sampling is deterministic from a sorted file list and explicit seed.
- Unknown labels raise errors rather than silently mapping to class zero.
- Unavailable benchmark fields remain null instead of being fabricated.

## Testing and CI

```bash
pip install -e ".[spark,tensorflow,dev,stats]"
ruff check src tests
pytest --cov=distributed_image_pipeline
```

Standard GitHub Actions CI runs Ruff, pytest with coverage, package installation checks and end-to-end smoke tests covering local preprocessing, local-mode Spark, TFRecord round trips, benchmark/report generation and TensorFlow training on synthetic images.

No GCP credentials are required for standard CI.

## Repository structure

```text
src/distributed_image_pipeline/   # packaged pipeline and benchmark code
tests/                            # unit + integration tests
dashboard/app.py                  # Streamlit benchmark dashboard
dashboard/README.md               # dashboard deployment instructions
notebooks/portfolio_analysis.ipynb# notebook analysis of benchmark results
docs/                             # data card, experiment design, architecture
reports/tables/                   # committed measured benchmark CSVs
reports/                          # generated reports and figures
requirements.txt                  # lightweight dashboard dependencies
data/                             # local datasets and split manifests (untracked)
```

## Limitations

- The dataset is relatively small, so fixed Spark startup and TensorFlow graph overheads matter more than they would at larger scale.
- Local-mode Spark results measure partition scaling on one machine, not multi-worker distributed scaling.
- Throughput depends on runner hardware and runtime conditions.
- The standalone TFRecord I/O improvement should not be interpreted as an equivalent end-to-end training speedup.
- The small CNN is used to study pipeline performance, not to claim state-of-the-art flower classification accuracy.

## Licence

Code is MIT-licensed (see [`LICENSE`](LICENSE)). Dataset images retain their original licences and are not redistributed in this repository.
