# Running a real multi-worker Dataproc benchmark

The Spark numbers in the main README (2.28x preprocessing speedup, 4.58x
TFRecord throughput gain) were measured with **local-mode Spark on a single
GitHub Actions runner** (`Spark.getOrCreate()`, partition scaling only). That
is a legitimate way to measure how partition count affects Spark's own
scheduling overhead, but it is not a distributed multi-worker benchmark:
there is only ever one physical machine.

This doc covers running the same `benchmark` CLI subcommand for real, on a
Dataproc cluster with a fixed worker count, so `reports/tables/benchmark_runs.csv`
gets rows with a genuine `workers` value. Once those rows exist, no code
changes are needed anywhere else:

- `metrics.worker_scaling()` already computes `speedup = serial_runtime /
  runtime_mean_s` and `parallel_efficiency = speedup / workers` from any row
  with a non-null `workers` column.
- `report.py`'s `make_figures()` already branches on `workers` having more
  than one distinct value and emits `speedup_vs_workers.png` and
  `scaling_efficiency.png`.
- `results.dataproc_cluster_metadata()` auto-populates
  `reports/run_metadata/*.json` with the real cluster name/worker
  count/machine type when the job actually runs on a Dataproc node.

None of this requires a Python client library for Dataproc -- job submission
goes through the `gcloud` CLI, same as the university coursework this
pipeline is descended from (`notebooks/coursework_original.py`).

## Prerequisites

- `gcloud` CLI authenticated against your GCP project (Cloud Shell has this
  already; if running locally, `gcloud auth login`).
- The Dataproc API enabled on the project (`gcloud services enable
  dataproc.googleapis.com` -- `dataproc_benchmark.sh` will also prompt you if
  it isn't).
- The default Compute Engine service account needs the `roles/dataproc.worker`
  role, or cluster creation will fail with a permissions error:
  ```
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com" \
    --role="roles/dataproc.worker"
  ```
- A GCS bucket you can write to, in (or close to) the region you'll run the
  cluster in. The public `gs://flowers-public/*/*.jpg` dataset is read
  directly -- no need to upload it yourself.

## Run one worker count

```bash
export PROJECT=your-gcp-project-id
export BUCKET=gs://your-bucket
export REGION=us-central1   # match your bucket's region if it's not multi-region

./scripts/dataproc_benchmark.sh 1   # then 2, then 4
```

Each invocation: builds this repo as a wheel, uploads it to
`$BUCKET/wheels/`, creates a cluster named `image-pipeline-bench-<n>w` with
`<n>` fixed workers (autoscaling is off -- `--num-workers` is a static
count, not min/max), submits `benchmark --modes spark --num-workers <n>` as a
PySpark job via `--py-files`, and deletes the cluster when the job finishes.

Do this **once per worker count you want to compare** (1, 2, 4 is the
default plan). Each run is independent, so they don't need to be
back-to-back, but don't create a cluster and then wait around -- a cluster
that sits idle for `DELETE_MAX_IDLE` (default 30 minutes) auto-deletes
itself as a cost safety net, which is a good thing to have, but the job
should still be submitted right after cluster creation finishes, not left
for later.

### Cost

Default machine type is `e2-standard-4` (4 vCPU, 16 GB) for both master and
workers -- enough for image preprocessing, an order of magnitude cheaper
than the Console's oversized 16-vCPU default. At published US on-demand
pricing this is roughly $0.13-0.15/hr per node, so a 4-worker run (5 nodes
total including master) is on the order of $0.70/hr, and each benchmark run
should only take a few minutes end to end. Override `MACHINE_TYPE`,
`WORKER_HOURLY_COST`, `MASTER_HOURLY_COST` as env vars if you use a
different shape or region.

## Bring the results back into the repo

Each run writes its CSV to `$BUCKET/dataproc-benchmark-results/`. Pull each
one down and merge its rows into the repo's own results file (they share the
same `RUN_COLUMNS` schema from `benchmark.py`):

```bash
gsutil cp gs://your-bucket/dataproc-benchmark-results/benchmark_runs_1w.csv .
gsutil cp gs://your-bucket/dataproc-benchmark-results/benchmark_runs_2w.csv .
gsutil cp gs://your-bucket/dataproc-benchmark-results/benchmark_runs_4w.csv .

# Append (not overwrite) into reports/tables/benchmark_runs.csv, keeping the
# existing local-mode rows for comparison. A quick way, since all four CSVs
# share the same header:
python3 - <<'PY'
import pandas as pd
frames = [pd.read_csv("reports/tables/benchmark_runs.csv")]
for n in (1, 2, 4):
    frames.append(pd.read_csv(f"benchmark_runs_{n}w.csv"))
pd.concat(frames, ignore_index=True).to_csv("reports/tables/benchmark_runs.csv", index=False)
PY
```

Then regenerate the report and figures:

```bash
python -m distributed_image_pipeline.cli report
```

`reports/benchmark_report.md`'s "Does increasing workers produce near-linear
scaling?" section, and `reports/figures/speedup_vs_workers.png` /
`scaling_efficiency.png`, should now be populated with real numbers instead
of "Not yet measured." Only then should the README's caveat about
partition-scaling-vs-multi-worker-scaling be updated -- with the real
numbers, not invented ones.
