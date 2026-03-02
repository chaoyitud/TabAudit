from __future__ import annotations

import json

from tab_audit.leaderboard.build import build_leaderboard


def test_leaderboard_build(tmp_path):
    reports = tmp_path / "reports"
    (reports / "ds_a").mkdir(parents=True)
    (reports / "ds_b").mkdir(parents=True)

    report_a = {
        "dataset_name": "ds_a",
        "dataset_slug": "ds_a",
        "target": "y",
        "metadata": {"source": "local"},
        "basic_stats": {"n_rows": 100, "n_cols": 10},
        "scores": {
            "quality_score": 80.0,
            "cleanliness_score": 85.0,
            "structure_score": 75.0,
            "learnability_score": 78.0,
            "label_quality_score": 70.0,
        },
        "checks": {
            "baseline_model": {
                "backend": "sklearn",
                "device": "cpu",
                "training_time_sec": 1.2,
                "primary_metric_cv": 0.78,
            }
        },
        "warnings": ["x"],
        "errors": [],
        "status": "OK",
    }
    report_b = {
        "dataset_name": "ds_b",
        "dataset_slug": "ds_b",
        "target": None,
        "metadata": {"source": "openml"},
        "basic_stats": {"n_rows": 50, "n_cols": 4},
        "scores": {
            "quality_score": 40.0,
            "cleanliness_score": 50.0,
            "structure_score": 30.0,
            "learnability_score": 20.0,
            "label_quality_score": 60.0,
        },
        "checks": {
            "baseline_model": {
                "backend": "none",
                "device": "cpu",
                "training_time_sec": 0.0,
                "primary_metric_cv": None,
            }
        },
        "warnings": [],
        "errors": ["bad"],
        "status": "FAILED",
        "error_message": "bad file",
    }

    (reports / "ds_a" / "report.json").write_text(json.dumps(report_a))
    (reports / "ds_b" / "report.json").write_text(json.dumps(report_b))

    df = build_leaderboard(reports)
    assert len(df) == 2
    assert df.iloc[0]["dataset_slug"] == "ds_a"
    assert "quality_score" in df.columns
    assert (reports / "leaderboard.csv").exists()
    assert (reports / "leaderboard.html").exists()
