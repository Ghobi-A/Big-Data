# Experiment design

## Research question

How do distributed preprocessing, Spark partitioning and TFRecord
serialisation affect the throughput and downstream training performance of an
image-based machine-learning pipeline?

## Sub-questions

1. Raw JPEG preprocessing performance (local baseline)
2. Distributed preprocessing performance (Spark)
3. Spark partition-count effects
4. Worker-count effects (where Dataproc resources are available)
5. Raw JPEG vs TFRecord input performance
6. Downstream TensorFlow training throughput by input format
7. Scaling efficiency
8. Estimated cloud cost
9. Reproducibility across repeated benchmark runs

## Baseline

Single-process local preprocessing (`preprocess-local`): read JPEG → decode →
RGB convert → aspect-preserving resize → centre-crop → re-encode → TFRecord
shard. The Spark pipeline uses the same preprocessing function, label mapping,
deterministic sampling and output schema, differing only in execution engine.

## I/O comparison fairness

The raw-JPEG and TFRecord input benchmarks must produce equivalent target image
geometry.

- Raw JPEG: decode RGB → aspect-preserving resize → centre-crop → batch.
- TFRecord: read preprocessed record → decode → batch.

This intentionally measures the effect of input/preprocessing strategy rather
than comparing TFRecord against a raw-JPEG path using a different direct-stretch
resize. The image libraries used at different pipeline stages are not expected
to produce byte-identical tensors; the controlled variable is the geometric
transformation and final tensor shape.

## Training comparison fairness

Both input formats use:

- the same deterministic train/validation split manifests,
- the same model architecture,
- the same random seed,
- the same optimiser,
- the same batch size,
- the same target image size,
- the same number of epochs.

The TFRecord manifests are matched through the stored source key so the same
examples are assigned to train and validation for both formats.

## Variables

**Independent:**

- Execution mode: `local`, `spark`
- Partition count: e.g. 1, 2, 4, 8 (`--partitions 1 2 4 8`)
- Dataset size: fractions via deterministic sampling, e.g. 0.1/0.25/0.5/1.0
- Worker count (cloud runs only, recorded via `--num-workers`)
- Input format for I/O and training experiments: raw JPEG vs TFRecord

**Dependent (recorded per run):**

- Wall-clock runtime (seconds)
- Throughput (images/second; samples/second for input benchmarks)
- Processed / failed / skipped file counts
- Output shard count and size
- Training: per-epoch time, samples/s, validation accuracy and loss

**Controlled:**

- Target image size (default 192×192), JPEG quality, seed, class list,
  train/validation split (persisted manifests), model, optimiser, batch size,
  epochs.

## Repeated runs

Default `--repeats 5` per configuration. Aggregation reports mean, standard
deviation, median, min and max runtime plus mean/std throughput
(`reports/tables/benchmark_summary.csv`). Repeated-run intervals are bootstrap
percentile intervals over the observed repeats — resampling intervals, not
formal population confidence intervals, and they are labelled as such.

## Partition scaling vs worker scaling

Partition count is **not** worker count. Partition sweeps on a fixed machine
are reported as *partition scaling*. Speedup (`serial_runtime /
parallel_runtime`) and parallel efficiency (`speedup / workers`) are computed
**only** when a true worker count is recorded (Dataproc runs).

## Dataset-size scaling

Deterministic subsampling (sorted file list + seeded sampler, identical
across modes) at explicit fractions. Sampled runs record `sample_rate`,
`seed` and a `sampled=True` flag, and are never presented as full-dataset
results. Used to test whether runtime scales linearly with dataset size.

## Statistical comparison

Where both groups have ≥3 repeats: Mann-Whitney U (non-parametric; runtimes
from few repeats are not assumed normal) plus a bootstrap interval on the
difference of means. With fewer repeats, only descriptive statistics are
reported. The report distinguishes statistical difference from practical
performance difference.

## Cost estimation

No cloud prices are hardcoded. Users supply `--worker-hourly-cost` /
`--master-hourly-cost`; estimated cost = runtime_hours × total hourly cluster
cost, plus cost per 1,000 images. All such figures are labelled estimates and
exclude cluster startup/idle time.

## Hardware and reproducibility

Every run writes `reports/run_metadata/<run_id>.json` with seed, git commit,
Python/PySpark/TensorFlow versions, OS, and Dataproc cluster metadata
(machine type, worker count) when the metadata server is reachable. Local
runs never fail when cloud metadata is unavailable.
