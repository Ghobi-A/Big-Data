#!/usr/bin/env bash
# Real multi-worker Dataproc benchmark runner.
#
# Builds this repo as a wheel, creates a fixed-size Dataproc cluster, submits
# the repo's own `benchmark` CLI subcommand as a PySpark job against it, and
# tears the cluster down. Run once per worker count you want to compare
# (e.g. 1, 2, 4) so `reports/tables/benchmark_runs.csv` ends up with real
# `workers`-tagged rows -- at that point `metrics.worker_scaling()` and the
# `speedup_vs_workers.png` / `scaling_efficiency.png` figures in
# `report.py` activate automatically; no pipeline code changes needed.
#
# Usage:
#   PROJECT=my-project BUCKET=gs://my-bucket REGION=us-central1 \
#     ./scripts/dataproc_benchmark.sh <num_workers>
#
# Example, run once per worker count:
#   ./scripts/dataproc_benchmark.sh 1
#   ./scripts/dataproc_benchmark.sh 2
#   ./scripts/dataproc_benchmark.sh 4
#
# Each run creates and deletes its own cluster, so runs don't have to be
# back-to-back -- but don't leave a cluster idle: it auto-deletes after
# DELETE_MAX_IDLE anyway, which is a safety net, not a substitute for
# submitting the job promptly after cluster creation finishes.
set -euo pipefail

NUM_WORKERS="${1:?usage: dataproc_benchmark.sh <num_workers>}"

PROJECT="${PROJECT:?set PROJECT to your GCP project ID}"
BUCKET="${BUCKET:?set BUCKET to a gs:// bucket you can write to, e.g. gs://my-bucket}"
REGION="${REGION:-us-central1}"
CLUSTER="image-pipeline-bench-${NUM_WORKERS}w"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
IMAGE_VERSION="${IMAGE_VERSION:-2.2-debian12}"
DELETE_MAX_IDLE="${DELETE_MAX_IDLE:-30m}"

# Rough US on-demand hourly prices for the default machine type, used only
# for the benchmark's own cost/1k-images estimate (estimate_run_cost() in
# benchmark.py). Override if you pick a different machine type or region.
WORKER_HOURLY_COST="${WORKER_HOURLY_COST:-0.134}"
MASTER_HOURLY_COST="${MASTER_HOURLY_COST:-0.134}"

INPUT_URI="${INPUT_URI:-gs://flowers-public/*/*.jpg}"
RESULTS_DIR="${BUCKET}/dataproc-benchmark-results"

echo "== 1/4: building wheel =="
rm -rf dist build ./*.egg-info
python3 -m pip install --quiet --upgrade build
python3 -m build --wheel
WHEEL="$(ls dist/*.whl | head -n1)"
echo "wheel: ${WHEEL}"
echo "uploading wheel to ${BUCKET}/wheels/"
gsutil cp "${WHEEL}" "${BUCKET}/wheels/"
WHEEL_GCS="${BUCKET}/wheels/$(basename "${WHEEL}")"

echo "== 2/4: creating cluster ${CLUSTER} (${NUM_WORKERS} x ${MACHINE_TYPE}) =="
gcloud dataproc clusters create "${CLUSTER}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --image-version="${IMAGE_VERSION}" \
  --master-machine-type="${MACHINE_TYPE}" \
  --master-boot-disk-type=pd-balanced \
  --master-boot-disk-size=50 \
  --num-workers="${NUM_WORKERS}" \
  --worker-machine-type="${MACHINE_TYPE}" \
  --worker-boot-disk-type=pd-balanced \
  --worker-boot-disk-size=50 \
  --enable-component-gateway \
  --delete-max-idle="${DELETE_MAX_IDLE}" \
  --properties="spark:spark.dynamicAllocation.enabled=false"

echo "== 3/4: submitting benchmark job =="
# --py-files installs this repo's package on the cluster; the entrypoint
# module (distributed_image_pipeline.cli) is resolved from that wheel, so
# no separate driver script needs to be uploaded.
gcloud dataproc jobs submit pyspark \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --cluster="${CLUSTER}" \
  --py-files="${WHEEL_GCS}" \
  "$(dirname "$0")/dataproc_job_entrypoint.py" \
  -- \
  benchmark \
  --input "${INPUT_URI}" \
  --output "${RESULTS_DIR}/output-${NUM_WORKERS}w" \
  --modes spark \
  --partitions 8 16 \
  --repeats 3 \
  --num-workers "${NUM_WORKERS}" \
  --worker-hourly-cost "${WORKER_HOURLY_COST}" \
  --master-hourly-cost "${MASTER_HOURLY_COST}" \
  --results "${RESULTS_DIR}/benchmark_runs_${NUM_WORKERS}w.csv"

echo "== 4/4: deleting cluster =="
gcloud dataproc clusters delete "${CLUSTER}" --project="${PROJECT}" --region="${REGION}" --quiet

echo "Done. Pull results down with:"
echo "  gsutil cp ${RESULTS_DIR}/benchmark_runs_${NUM_WORKERS}w.csv reports/tables/"
echo "Then append/merge that file's rows into reports/tables/benchmark_runs.csv and re-run:"
echo "  python -m distributed_image_pipeline.cli report"
