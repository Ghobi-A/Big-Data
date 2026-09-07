#!/usr/bin/env bash
# Install the Python runtime required by the distributed image benchmark.
#
# This file is uploaded to GCS by dataproc_benchmark.sh and executed by
# Dataproc on every node before the cluster becomes ready. Keeping the
# bootstrap in-repo makes the benchmark environment reproducible from the
# Git commit instead of depending on a moving public initialization action.
set -euo pipefail

DEFAULT_DATAPROC_PYTHON="/opt/conda/default/bin/python"
TENSORFLOW_PACKAGE="${TENSORFLOW_PACKAGE:-tensorflow-cpu==2.19.1}"

# Dataproc 2.x uses the Miniconda interpreter under /opt/conda/default as its
# default Python runtime for Spark jobs. /usr/bin/python3 is the OS Python and
# may be externally managed on Debian 12, so installing there can fail during
# cluster initialization and would not populate Spark's actual environment.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # Explicit override supplied by the caller.
elif [[ -x "${DEFAULT_DATAPROC_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_DATAPROC_PYTHON}"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "No usable Python interpreter found for Dataproc bootstrap." >&2
  exit 1
fi

echo "Dataproc bootstrap Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version

PIP_DISABLE_PIP_VERSION_CHECK=1 "${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
  "${TENSORFLOW_PACKAGE}" \
  "pillow>=10,<12" \
  "numpy>=1.24,<2.2" \
  "pandas>=2,<3"

"${PYTHON_BIN}" - <<'PY'
import sys

import numpy
import pandas
import PIL
import tensorflow as tf

print(
    "Dataproc Python runtime ready:",
    f"python={sys.executable}",
    f"tensorflow={tf.__version__}",
    f"pillow={PIL.__version__}",
    f"numpy={numpy.__version__}",
    f"pandas={pandas.__version__}",
)
PY
