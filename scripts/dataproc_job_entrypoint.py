"""Thin Dataproc job entrypoint.

`gcloud dataproc jobs submit pyspark` requires a driver .py file; the actual
logic lives in the distributed_image_pipeline package, installed on the
cluster via `--py-files` (see dataproc_benchmark.sh). This just forwards
argv to the repo's own CLI, so cluster-side behaviour is identical to running
`python -m distributed_image_pipeline.cli benchmark ...` locally.
"""

import sys

from distributed_image_pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
