from __future__ import annotations

import json

from typer.testing import CliRunner

from tab_audit import cli
from tab_audit.cli import app
from tab_audit.io.dataset_inputs import (
    parse_kaggle_dataset_specs,
    parse_local_dir_paths,
    parse_local_file_paths,
    parse_openml_dataset_ids,
)


def test_parse_openml_dataset_ids_from_mixed_sources(tmp_path):
    ids_txt = tmp_path / "ids.txt"
    ids_txt.write_text("61, 15\n# comment\n37 61\n")

    ids_json = tmp_path / "ids.json"
    ids_json.write_text(json.dumps({"dataset_ids": [151, 61]}))

    ids = parse_openml_dataset_ids(
        dataset_ids=[3],
        dataset_id_lists=["1,2", "2 3 4"],
        dataset_ids_file=str(ids_txt),
    )
    assert ids == [3, 1, 2, 4, 61, 15, 37]

    ids2 = parse_openml_dataset_ids(dataset_ids_file=str(ids_json))
    assert ids2 == [151, 61]


def test_parse_kaggle_dataset_specs_from_file_and_inline(tmp_path):
    specs_yaml = tmp_path / "kaggle.yaml"
    specs_yaml.write_text(
        """
datasets:
  - "owner/a:file1.csv"
  - slug: owner/b
    file: train.csv
"""
    )

    specs = parse_kaggle_dataset_specs(
        datasets=["owner/a:file1.csv", "owner/c:file2.csv"],
        datasets_file=str(specs_yaml),
    )
    assert specs == [
        ("owner/a", "file1.csv"),
        ("owner/c", "file2.csv"),
        ("owner/b", "train.csv"),
    ]


def test_batch_openml_accepts_dataset_ids_file(tmp_path, monkeypatch):
    ids_file = tmp_path / "openml_ids.txt"
    ids_file.write_text("61\n15\n37\n")

    captured: dict[str, list[int]] = {}

    def fake_run_batch(specs, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        captured["ids"] = [s.dataset_id for s in specs]

    monkeypatch.setattr(cli, "_run_batch", fake_run_batch)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "batch",
            "openml",
            "--dataset-ids-file",
            str(ids_file),
            "--out",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["ids"] == [61, 15, 37]


def test_parse_local_paths_inputs(tmp_path):
    file_list = tmp_path / "files.txt"
    file_list.write_text("/tmp/a.csv\n/tmp/b.csv\n")
    dir_list = tmp_path / "dirs.yaml"
    dir_list.write_text("local_dirs: [\"/tmp/d1\", \"/tmp/d2\"]\n")

    files = parse_local_file_paths(
        local_files=["/tmp/x.csv"],
        local_file_lists=["/tmp/y.csv,/tmp/z.csv"],
        local_files_file=str(file_list),
    )
    assert files == ["/tmp/x.csv", "/tmp/y.csv", "/tmp/z.csv", "/tmp/a.csv", "/tmp/b.csv"]

    dirs = parse_local_dir_paths(local_dirs=["/tmp/dd"], local_dirs_file=str(dir_list))
    assert dirs == ["/tmp/dd", "/tmp/d1", "/tmp/d2"]


def test_batch_mixed_combines_sources(tmp_path, monkeypatch):
    local_file = tmp_path / "one.csv"
    local_file.write_text("a,b\n1,2\n")
    local_dir = tmp_path / "dir"
    local_dir.mkdir()
    (local_dir / "two.csv").write_text("x,y\n3,4\n")

    openml_file = tmp_path / "openml.txt"
    openml_file.write_text("61\n")
    kaggle_file = tmp_path / "kaggle.txt"
    kaggle_file.write_text("owner/ds:file.csv\n")

    captured: dict[str, list[str]] = {}

    def fake_run_batch(specs, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        captured["sources"] = [s.source for s in specs]
        captured["names"] = [s.name for s in specs]

    monkeypatch.setattr(cli, "_run_batch", fake_run_batch)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "batch",
            "mixed",
            "--openml-dataset-ids-file",
            str(openml_file),
            "--kaggle-datasets-file",
            str(kaggle_file),
            "--local-files",
            str(local_file),
            "--local-dirs",
            str(local_dir),
            "--out",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["sources"] == ["openml", "kaggle", "local", "local"]
    assert "openml_61" in captured["names"]
