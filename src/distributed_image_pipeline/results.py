"""Run-result and reproducibility metadata structures."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PipelineRunResult:
    """Execution metadata for one preprocessing run."""

    mode: str  # "local" or "spark"
    input_files: int
    processed_files: int
    failed_files: int
    skipped_files: int
    partitions: int
    runtime_seconds: float
    throughput_images_per_second: float
    output_files: int
    output_size_mb: float | None = None
    sample_rate: float = 1.0
    seed: int | None = None
    sampled: bool = False
    output_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def git_commit_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _optional_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def dataproc_cluster_metadata() -> dict[str, Any] | None:
    """Best-effort read of Dataproc/GCE metadata. Returns None off-cloud.

    Never raises: local runs must not fail when cloud metadata is unavailable.
    """
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/?recursive=true",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            data = json.loads(resp.read().decode())
        attrs = data.get("attributes", {})
        return {
            "machine_type": data.get("machineType", "").split("/")[-1] or None,
            "dataproc_cluster_name": attrs.get("dataproc-cluster-name"),
            "dataproc_role": attrs.get("dataproc-role"),
            "worker_count": attrs.get("dataproc-worker-count"),
        }
    except Exception:
        return None


def collect_environment_metadata() -> dict[str, Any]:
    """Reproducibility metadata recorded alongside every benchmark run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_sha(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pyspark_version": _optional_version("pyspark"),
        "tensorflow_version": _optional_version("tensorflow"),
        "numpy_version": _optional_version("numpy"),
        "pandas_version": _optional_version("pandas"),
        "dataproc": dataproc_cluster_metadata(),
    }


def write_run_metadata(directory: str | Path, run_id: str, extra: dict | None = None) -> Path:
    """Write reproducibility metadata JSON for a run into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = collect_environment_metadata()
    if extra:
        meta.update(extra)
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path
