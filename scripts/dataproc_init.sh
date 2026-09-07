#!/usr/bin/env bash
# Install the Python runtime required by the distributed image benchmark.
#
# This file is uploaded to GCS by dataproc_benchmark.sh and executed by
# Dataproc on every node before the cluster becomes ready. Keeping the
# bootstrap in-repo makes the benchmark environment reproducible from the
# Git commit instead of depending on a moving public initialization action.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
TENSORFLOW_PACKAGE="${TENSORFLOW_PACKAGE:-tensorflow-cpu==2.19.1}"

"${PYTHON_BIN}" -m pip install --no-cache-dir --upgrade \
  "${TENSORFLOW_PACKAGE}" \
  "pillow>=10,<12" \
  "numpy>=1.24,<2.2" \
  "pandas>=2,<3"

"${PYTHON_BIN}" - <<'PY'
import numpy
import pandas
import PIL
import tensorflow as tf

print(
    "Dataproc Python runtime ready:",
    f"tensorflow={tf.__version__}",
    f"pillow={PIL.__version__}",
    f"numpy={numpy.__version__}",
    f"pandas={pandas.__version__}",
)
PY
