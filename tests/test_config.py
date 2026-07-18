import json

import pytest

from distributed_image_pipeline.config import (
    ClusterCost,
    ConfigError,
    PipelineConfig,
    load_config,
)


def valid_kwargs(**over):
    kw = dict(input_uri="data/raw", output_uri="data/processed")
    kw.update(over)
    return kw


def test_valid_config_defaults():
    cfg = PipelineConfig(**valid_kwargs())
    assert cfg.sample_rate == 1.0  # full processing is the default
    assert cfg.partitions == 1
    assert cfg.target_size == (192, 192)
    assert not cfg.is_sampled


@pytest.mark.parametrize("partitions", [0, -1, -8])
def test_invalid_partitions(partitions):
    with pytest.raises(ConfigError):
        PipelineConfig(**valid_kwargs(partitions=partitions))


@pytest.mark.parametrize("rate", [0.0, -0.5, 1.5, 2.0])
def test_invalid_sample_rate(rate):
    with pytest.raises(ConfigError):
        PipelineConfig(**valid_kwargs(sample_rate=rate))


@pytest.mark.parametrize("h,w", [(0, 192), (192, 0), (-1, -1)])
def test_invalid_target_dimensions(h, w):
    with pytest.raises(ConfigError):
        PipelineConfig(**valid_kwargs(target_height=h, target_width=w))


@pytest.mark.parametrize("uri_field", ["input_uri", "output_uri"])
@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_paths_rejected(uri_field, bad):
    with pytest.raises(ConfigError):
        PipelineConfig(**valid_kwargs(**{uri_field: bad}))


def test_invalid_error_policy():
    with pytest.raises(ConfigError):
        PipelineConfig(**valid_kwargs(error_policy="ignore"))


def test_env_variables_resolved():
    env = {
        "GCS_INPUT_URI": "gs://bucket/in",
        "GCS_OUTPUT_URI": "gs://bucket/out",
        "GCP_PROJECT_ID": "example-project",
    }
    cfg = load_config(env=env)
    assert cfg.input_uri == "gs://bucket/in"
    assert cfg.gcp_project_id == "example-project"


def test_override_beats_env_and_file(tmp_path):
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps({"input_uri": "file-in", "output_uri": "file-out", "partitions": 2}))
    env = {"GCS_INPUT_URI": "env-in"}
    cfg = load_config(config_file=f, env=env, input_uri="cli-in")
    assert cfg.input_uri == "cli-in"
    assert cfg.output_uri == "file-out"
    assert cfg.partitions == 2


def test_no_default_gcp_project():
    cfg = load_config(env={}, input_uri="a", output_uri="b")
    assert cfg.gcp_project_id is None  # never silently uses a personal project


def test_unknown_config_keys_rejected(tmp_path):
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps({"input_uri": "a", "output_uri": "b", "bogus": 1}))
    with pytest.raises(ConfigError, match="bogus"):
        load_config(config_file=f, env={})


def test_missing_config_file():
    with pytest.raises(ConfigError):
        load_config(config_file="/nonexistent/cfg.json", env={})


def test_cluster_cost():
    cost = ClusterCost(worker_hourly_cost=0.10, master_hourly_cost=0.12, num_workers=4)
    assert cost.total_hourly_cost == pytest.approx(0.52)
    with pytest.raises(ConfigError):
        ClusterCost(worker_hourly_cost=-1.0)
