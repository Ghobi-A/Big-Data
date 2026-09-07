#!/usr/bin/env bash
# Real multi-worker Dataproc benchmark runner.
#
# Builds this repo as a wheel, provisions the Python runtime on every Dataproc
# node, validates the driver/executor environments, submits the repo's own
# benchmark CLI, and tears the cluster down even when a job fails.
#
# Usage:
#   PROJECT=my-project BUCKET=gs://my-bucket REGION=us-central1 \
#     ./scripts/dataproc_benchmark.sh <num_workers>
#
# Example:
#   ./scripts/dataproc_benchmark.sh 1
#   ./scripts/dataproc_benchmark.sh 2
#   ./scripts/dataproc_benchmark.sh 4
set -euo pipefail

NUM_WORKERS="${1:?usage: dataproc_benchmark.sh <num_workers>}"

PROJECT="${PROJECT:?set PROJECT to your GCP project ID}"
BUCKET="${BUCKET:?set BUCKET to a gs:// bucket you can write to, e.g. gs://my-bucket}"
REGION="${REGION:-us-central1}"
CLUSTER="image-pipeline-bench-${NUM_WORKERS}w"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
IMAGE_VERSION="${IMAGE_VERSION:-2.2.86-debian12}"
DELETE_MAX_IDLE="${DELETE_MAX_IDLE:-30m}"
KEEP_CLUSTER_ON_FAILURE="${KEEP_CLUSTER_ON_FAILURE:-0}"

# Rough US on-demand hourly prices for the default machine type, used only
# for the benchmark's own cost/1k-images estimate. Override when using a
# different machine type or region.
WORKER_HOURLY_COST="${WORKER_HOURLY_COST:-0.134}"
MASTER_HOURLY_COST="${MASTER_HOURLY_COST:-0.134}"

INPUT_URI="${INPUT_URI:-gs://flowers-public/*/*.jpg}"
RESULTS_DIR="${BUCKET}/dataproc-benchmark-results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
INIT_GCS="${BUCKET}/init-actions/dataproc_init-${GIT_SHA}.sh"
CLUSTER_CREATED=0
RUN_SUCCEEDED=0

cleanup() {
  local exit_code=$?
  if [[ "${CLUSTER_CREATED}" -eq 1 ]]; then
    if [[ "${RUN_SUCCEEDED}" -ne 1 && "${KEEP_CLUSTER_ON_FAILURE}" == "1" ]]; then
      echo "Benchmark failed; KEEP_CLUSTER_ON_FAILURE=1, leaving ${CLUSTER} running for debugging."
    else
      echo "Deleting cluster ${CLUSTER}..."
      gcloud dataproc clusters delete "${CLUSTER}" \
        --project="${PROJECT}" \
        --region="${REGION}" \
        --quiet || true
      CLUSTER_CREATED=0
    fi
  fi
  return "${exit_code}"
}
trap cleanup EXIT INT TERM

echo "== 1/5: building wheel =="
cd "${REPO_ROOT}"
rm -rf dist build ./*.egg-info
python3 -m pip install --quiet --upgrade build
python3 -m build --wheel
WHEEL="$(ls dist/*.whl | head -n1)"
echo "wheel: ${WHEEL}"
echo "uploading wheel to ${BUCKET}/wheels/"
gsutil cp "${WHEEL}" "${BUCKET}/wheels/"
WHEEL_GCS="${BUCKET}/wheels/$(basename "${WHEEL}")"

echo "== 2/5: uploading pinned Dataproc bootstrap =="
gsutil cp "${SCRIPT_DIR}/dataproc_init.sh" "${INIT_GCS}"

echo "== 3/5: creating cluster ${CLUSTER} (${NUM_WORKERS} x ${MACHINE_TYPE}) =="
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
  --initialization-actions="${INIT_GCS}" \
  --initialization-action-timeout=15m \
  --properties="spark:spark.dynamicAllocation.enabled=false"
CLUSTER_CREATED=1

echo "== 4/5: validating runtime and running benchmark =="
# Fail before the expensive benchmark grid if TensorFlow/Pillow/etc. cannot be
# imported on the driver or Spark executors.
gcloud dataproc jobs submit pyspark \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --cluster="${CLUSTER}" \
  "${SCRIPT_DIR}/dataproc_runtime_check.py"

# --py-files makes this repo's package importable on the driver and executors;
# dataproc_init.sh installs its binary/runtime dependencies on every node.
gcloud dataproc jobs submit pyspark \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --cluster="${CLUSTER}" \
  --py-files="${WHEEL_GCS}" \
  "${SCRIPT_DIR}/dataproc_job_entrypoint.py" \
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

RUN_SUCCEEDED=1

echo "== 5/5: deleting cluster =="
cleanup
trap - EXIT INT TERM

echo "Done. Pull results down with:"
echo "  gsutil cp ${RESULTS_DIR}/benchmark_runs_${NUM_WORKERS}w.csv reports/tables/"
echo "Then append/merge that file's rows into reports/tables/benchmark_runs.csv and re-run:"
echo "  python -m distributed_image_pipeline.cli report"
