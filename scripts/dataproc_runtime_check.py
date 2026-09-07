"""Fail fast if the Dataproc driver or executors lack benchmark dependencies."""

from __future__ import annotations

import json
import socket

from pyspark import SparkContext


def _versions(_: int) -> dict[str, str]:
    import numpy
    import pandas
    import PIL
    import tensorflow as tf

    return {
        "host": socket.gethostname(),
        "tensorflow": tf.__version__,
        "pillow": PIL.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }


def main() -> None:
    sc = SparkContext.getOrCreate()

    # Import once on the driver as well as inside executor tasks. A small number
    # of partitions is enough to exercise the Python environment before the
    # real benchmark allocates work.
    driver = _versions(0)
    executor_checks = sc.parallelize(
        range(max(1, sc.defaultParallelism)),
        max(1, sc.defaultParallelism),
    ).map(_versions).collect()

    unique_hosts = {item["host"]: item for item in executor_checks}
    print("DATAPROC_RUNTIME_CHECK_OK")
    print(json.dumps({"driver": driver, "executors": unique_hosts}, sort_keys=True))


if __name__ == "__main__":
    main()
