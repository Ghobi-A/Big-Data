SCRIPT = "scripts/dataproc_benchmark.sh"
INIT_SCRIPT = "scripts/dataproc_init.sh"


def _script_text() -> str:
    with open(SCRIPT, encoding="utf-8") as handle:
        return handle.read()


def _init_script_text() -> str:
    with open(INIT_SCRIPT, encoding="utf-8") as handle:
        return handle.read()


def test_dataproc_runner_preserves_single_node_baseline():
    text = _script_text()

    assert 'if [[ "${NUM_WORKERS}" -eq 1 ]]' in text
    assert "--single-node" in text
    assert '--num-workers="${NUM_WORKERS}"' in text


def test_dataproc_runner_keeps_runtime_bootstrap_in_common_cluster_args():
    text = _script_text()
    common_start = text.index("COMMON_CLUSTER_ARGS=(")
    single_node_start = text.index('if [[ "${NUM_WORKERS}" -eq 1 ]]')
    common_block = text[common_start:single_node_start]

    assert '--initialization-actions="${INIT_GCS}"' in common_block
    assert "--initialization-action-timeout=15m" in common_block
    assert '--image-version="${IMAGE_VERSION}"' in common_block


def test_single_node_cost_is_not_double_counted():
    text = _script_text()

    assert "BENCHMARK_WORKER_HOURLY_COST=0" in text
    assert '--worker-hourly-cost "${BENCHMARK_WORKER_HOURLY_COST}"' in text


def test_default_image_uses_supported_minor_alias_not_expiring_subminor():
    text = _script_text()

    assert 'IMAGE_VERSION="${IMAGE_VERSION:-2.2-debian12}"' in text
    assert "2.2.86-debian12" not in text
    assert "Resolved Dataproc image:" in text


def test_bootstrap_uses_dataproc_default_python_environment():
    text = _init_script_text()

    assert 'DEFAULT_DATAPROC_PYTHON="/opt/conda/default/bin/python"' in text
    assert 'elif [[ -x "${DEFAULT_DATAPROC_PYTHON}" ]]' in text
    assert 'PYTHON_BIN="${DEFAULT_DATAPROC_PYTHON}"' in text
    assert 'PYTHON_BIN="$(command -v python3)"' in text
    assert 'f"python={sys.executable}"' in text


def test_cluster_cleanup_is_armed_before_cluster_creation():
    text = _script_text()

    armed = text.index("CLUSTER_CREATED=1", text.index("COMMON_CLUSTER_ARGS=("))
    create = text.index('gcloud dataproc clusters create "${CLUSTER}"', armed)
    assert armed < create
