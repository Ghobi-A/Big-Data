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

Each invocation now performs the full environment lifecycle:

1. Build the repository wheel and upload it to `$BUCKET/wheels/`.
2. Upload the repository-owned `scripts/dataproc_init.sh` bootstrap to the
   bucket. The bootstrap installs the pinned TensorFlow runtime plus Pillow,
   NumPy and pandas on every node.
3. Create the Dataproc topology for the requested benchmark point. The default
   image is pinned to `2.2.86-debian12`; override `IMAGE_VERSION` explicitly
   if you intentionally want another image.
4. Run `scripts/dataproc_runtime_check.py` to verify imports on the Spark
   driver and executors before the benchmark grid starts.
5. Submit `benchmark --modes spark --num-workers <n>` through `--py-files`,
   persist the result CSV to GCS, and delete the cluster.

### 1-worker baseline versus 2/4-worker clusters

Dataproc standard-mode clusters require at least two worker VMs, so the
`1` benchmark point cannot be created with `--num-workers=1`. The runner uses
Dataproc **single-node mode** for that baseline: one `e2-standard-4` VM hosts
both master and worker roles. The benchmark metadata still records
`workers=1`, which is the correct scaling baseline, but cost estimation charges
only that one VM rather than double-counting it as a master plus a worker.

The `2` and `4` benchmark points use normal Dataproc clusters with one master
plus the requested number of worker VMs. All topologies share the same pinned
image, initialization action, runtime preflight, Spark configuration, and
benchmark command.

Do this **once per worker count you want to compare** (1, 2, 4 is the
default plan). Each run is independent, so they do not need to be
back-to-back.

Cluster deletion is registered as an exit trap, so a failed preflight or
benchmark job is cleaned up instead of leaving a billable cluster running
until the idle timeout. For debugging only, set `KEEP_CLUSTER_ON_FAILURE=1`
to leave a failed cluster in place. `DELETE_MAX_IDLE` still defaults to 30
minutes as a second safety net.

### Cost

Default machine type is `e2-standard-4` (4 vCPU, 16 GB). The 1-worker baseline
is one single-node VM. The 2-worker and 4-worker points use one master plus
two or four workers respectively. Override `MACHINE_TYPE`,
`WORKER_HOURLY_COST`, and `MASTER_HOURLY_COST` as environment variables if
you use a different shape or region.

## Bring the results back into the repo

Each run writes its CSV to `$BUCKET/dataproc-benchmark-results/`. The
benchmark writer handles `gs://` paths directly, so the CSV survives cluster
teardown rather than being written to the master's local filesystem.

Pull each file down and merge its rows into the repo's own results file (they
share the same `RUN_COLUMNS` schema from `benchmark.py`):

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
