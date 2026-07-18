"""Validated pipeline configuration.

Configuration values are resolved with the following precedence (highest wins):
explicit arguments (e.g. CLI flags) > environment variables > config file.

No personal GCP project or bucket name is ever used as a default: cloud-related
settings must be supplied explicitly via one of the mechanisms above.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ENV_PREFIX_MAP = {
    "GCP_PROJECT_ID": "gcp_project_id",
    "GCP_REGION": "gcp_region",
    "GCS_INPUT_URI": "input_uri",
    "GCS_OUTPUT_URI": "output_uri",
    "DATAPROC_CLUSTER": "dataproc_cluster",
}

VALID_ERROR_POLICIES = ("raise", "skip", "log")


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a preprocessing run.

    ``sample_rate`` defaults to 1.0: full-dataset processing is the default.
    Sampled runs must set it explicitly and always carry a seed so that the
    sample is deterministic and reportable.
    """

    input_uri: str
    output_uri: str
    partitions: int = 1
    sample_rate: float = 1.0
    target_height: int = 192
    target_width: int = 192
    seed: int = 42
    error_policy: str = "log"
    gcp_project_id: str | None = None
    gcp_region: str | None = None
    dataproc_cluster: str | None = None

    def __post_init__(self) -> None:
        if not self.input_uri or not str(self.input_uri).strip():
            raise ConfigError("input_uri must be a non-empty path or URI")
        if not self.output_uri or not str(self.output_uri).strip():
            raise ConfigError("output_uri must be a non-empty path or URI")
        if not isinstance(self.partitions, int) or self.partitions <= 0:
            raise ConfigError(f"partitions must be a positive integer, got {self.partitions!r}")
        if not (0 < self.sample_rate <= 1):
            raise ConfigError(f"sample_rate must satisfy 0 < rate <= 1, got {self.sample_rate!r}")
        if self.target_height <= 0 or self.target_width <= 0:
            raise ConfigError(
                f"target dimensions must be positive, got "
                f"{self.target_height}x{self.target_width}"
            )
        if not isinstance(self.seed, int):
            raise ConfigError(f"seed must be an integer, got {self.seed!r}")
        if self.error_policy not in VALID_ERROR_POLICIES:
            raise ConfigError(
                f"error_policy must be one of {VALID_ERROR_POLICIES}, got {self.error_policy!r}"
            )

    @property
    def target_size(self) -> tuple[int, int]:
        return (self.target_height, self.target_width)

    @property
    def is_sampled(self) -> bool:
        return self.sample_rate < 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClusterCost:
    """Hourly cluster cost inputs for cost *estimation*.

    Prices are never hardcoded in this package; they must be supplied by the
    user (CLI flags or config file), and all derived figures are estimates.
    """

    worker_hourly_cost: float = 0.0
    master_hourly_cost: float = 0.0
    num_workers: int = 0
    num_masters: int = 1

    def __post_init__(self) -> None:
        if self.worker_hourly_cost < 0 or self.master_hourly_cost < 0:
            raise ConfigError("hourly costs must be non-negative")
        if self.num_workers < 0 or self.num_masters < 0:
            raise ConfigError("node counts must be non-negative")

    @property
    def total_hourly_cost(self) -> float:
        return (
            self.worker_hourly_cost * self.num_workers
            + self.master_hourly_cost * self.num_masters
        )


def _read_config_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file {p} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file {p} must contain a JSON object")
    return data


def load_config(
    config_file: str | Path | None = None,
    env: dict[str, str] | None = None,
    **overrides: Any,
) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from file, environment and overrides.

    Precedence: overrides > environment variables > config file.
    """
    env = os.environ if env is None else env
    values: dict[str, Any] = {}
    if config_file is not None:
        values.update(_read_config_file(config_file))
    for env_name, field_name in ENV_PREFIX_MAP.items():
        if env.get(env_name):
            values[field_name] = env[env_name]
    values.update({k: v for k, v in overrides.items() if v is not None})
    allowed = {f.name for f in PipelineConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(values) - set(allowed)
    if unknown:
        raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")
    try:
        return PipelineConfig(**values)
    except TypeError as exc:
        raise ConfigError(str(exc)) from exc
