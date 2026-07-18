import json

import pytest

from distributed_image_pipeline.cli import build_parser, main


def test_help_available(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0


def test_subcommand_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_invalid_config_returns_error_code(tmp_path):
    rc = main([
        "preprocess-local",
        "--input", str(tmp_path),
        "--output", str(tmp_path / "out"),
        "--partitions", "0",
    ])
    assert rc == 2


def test_missing_input_returns_error_code(tmp_path):
    rc = main([
        "preprocess-local",
        "--input", str(tmp_path / "nope"),
        "--output", str(tmp_path / "out"),
        "--output-format", "jpeg",
    ])
    assert rc == 2


def test_preprocess_local_end_to_end(flowers_dir, tmp_path, capsys):
    rc = main([
        "preprocess-local",
        "--input", str(flowers_dir),
        "--output", str(tmp_path / "out"),
        "--target-size", "32", "32",
        "--output-format", "jpeg",
    ])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["processed_files"] == 20
    assert result["mode"] == "local"


@pytest.mark.tf
def test_report_cli(flowers_dir, tmp_path, capsys):
    pytest.importorskip("tensorflow")
    pytest.importorskip("matplotlib")
    rc = main([
        "benchmark",
        "--input", str(flowers_dir),
        "--output", str(tmp_path / "scratch"),
        "--modes", "local",
        "--partitions", "1",
        "--repeats", "2",
        "--target-size", "32", "32",
        "--results", str(tmp_path / "runs.csv"),
        "--summary", str(tmp_path / "summary.csv"),
        "--metadata-dir", str(tmp_path / "meta"),
    ])
    assert rc == 0
    rc = main([
        "report",
        "--input", str(tmp_path / "runs.csv"),
        "--output", str(tmp_path / "report.md"),
        "--figures-dir", str(tmp_path / "figures"),
    ])
    assert rc == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "summary.csv").exists()
