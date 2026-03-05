from __future__ import annotations

import pandas as pd

from tab_audit import cli
from tab_audit.config import DatasetSpec, GlobalConfig


def test_run_batch_continues_after_failure(monkeypatch, tmp_path):
    out_dir = tmp_path / "reports"
    cfg = GlobalConfig(batch_concurrency=1)
    specs = [
        DatasetSpec(name="good_ds", source="local", path="good.csv", target="y"),
        DatasetSpec(name="bad_ds", source="local", path="bad.csv", target="y"),
    ]

    def fake_evaluate_one(spec, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if spec.name == "bad_ds":
            raise RuntimeError("boom")
        report = {
            "dataset_slug": "good_ds",
            "dataset_name": "good_ds",
            "target": "y",
            "basic_stats": {"n_rows": 10, "n_cols": 3},
            "scores": {
                "quality_score": 70.0,
                "cleanliness_score": 70.0,
                "structure_score": 70.0,
                "learnability_score": 70.0,
                "label_quality_score": 70.0,
            },
            "checks": {"baseline_model": {"backend": "sklearn", "device": "cpu", "training_time_sec": 0.1}},
            "warnings": [],
            "errors": [],
            "status": "OK",
        }
        row = {
            "dataset_slug": "good_ds",
            "dataset_name": "good_ds",
            "source": "local",
            "target": "y",
            "status": "OK",
            "n_rows": 10,
            "n_cols": 3,
            "quality_score": 70.0,
            "cleanliness_score": 70.0,
            "structure_score": 70.0,
            "learnability_score": 70.0,
            "label_quality_score": 70.0,
            "warnings_count": 0,
            "errors_count": 0,
            "backend": "sklearn",
            "device": "cpu",
            "runtime_sec": 0.1,
            "error_message": None,
        }
        return report, row

    monkeypatch.setattr(cli, "_evaluate_one", fake_evaluate_one)
    monkeypatch.setattr(cli, "_print_result", lambda report: None)
    monkeypatch.setattr(cli, "build_leaderboard", lambda out: pd.read_csv(out_dir / "summary.csv"))

    cli._run_batch(
        specs=specs,
        cfg=cfg,
        out=str(out_dir),
        save_row_ids=False,
        enable_profiling=False,
    )

    summary = pd.read_csv(out_dir / "summary.csv")
    assert len(summary) == 2
    assert set(summary["status"].astype(str)) == {"OK", "FAILED"}


def test_run_batch_continues_if_failure_report_write_fails(monkeypatch, tmp_path):
    out_dir = tmp_path / "reports"
    cfg = GlobalConfig(batch_concurrency=1)
    specs = [
        DatasetSpec(name="good_ds2", source="local", path="good.csv", target="y"),
        DatasetSpec(name="bad_ds2", source="local", path="bad.csv", target="y"),
    ]

    def fake_evaluate_one(spec, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if spec.name == "bad_ds2":
            raise RuntimeError("dataset fail")
        report = {
            "dataset_slug": "good_ds2",
            "dataset_name": "good_ds2",
            "target": "y",
            "basic_stats": {"n_rows": 10, "n_cols": 3},
            "scores": {
                "quality_score": 80.0,
                "cleanliness_score": 80.0,
                "structure_score": 80.0,
                "learnability_score": 80.0,
                "label_quality_score": 80.0,
            },
            "checks": {"baseline_model": {"backend": "sklearn", "device": "cpu", "training_time_sec": 0.1}},
            "warnings": [],
            "errors": [],
            "status": "OK",
        }
        row = {
            "dataset_slug": "good_ds2",
            "dataset_name": "good_ds2",
            "source": "local",
            "target": "y",
            "status": "OK",
            "n_rows": 10,
            "n_cols": 3,
            "quality_score": 80.0,
            "cleanliness_score": 80.0,
            "structure_score": 80.0,
            "learnability_score": 80.0,
            "label_quality_score": 80.0,
            "warnings_count": 0,
            "errors_count": 0,
            "backend": "sklearn",
            "device": "cpu",
            "runtime_sec": 0.1,
            "error_message": None,
        }
        return report, row

    monkeypatch.setattr(cli, "_evaluate_one", fake_evaluate_one)
    monkeypatch.setattr(
        cli,
        "_write_failure_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("io fail")),
    )
    monkeypatch.setattr(cli, "_print_result", lambda report: None)
    monkeypatch.setattr(cli, "build_leaderboard", lambda out: pd.read_csv(out_dir / "summary.csv"))

    cli._run_batch(
        specs=specs,
        cfg=cfg,
        out=str(out_dir),
        save_row_ids=False,
        enable_profiling=False,
    )

    summary = pd.read_csv(out_dir / "summary.csv")
    assert len(summary) == 2
    assert set(summary["status"].astype(str)) == {"OK", "FAILED"}
