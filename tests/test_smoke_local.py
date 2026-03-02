from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from tab_audit.cli import app


def test_smoke_local_evaluate_generates_report(tmp_path):
    data_path = tmp_path / "demo.csv"
    out_dir = tmp_path / "reports"

    rows = 60
    df = pd.DataFrame(
        {
            "x1": list(range(rows)),
            "x2": ["a" if i % 2 == 0 else "b" for i in range(rows)],
            "y": [0 if i % 2 == 0 else 1 for i in range(rows)],
        }
    )
    df.to_csv(data_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "local",
            "--path",
            str(data_path),
            "--target",
            "y",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    report_json = out_dir / "demo" / "report.json"
    assert report_json.exists()
    payload = json.loads(report_json.read_text())
    assert "scores" in payload
    assert "quality_score" in payload["scores"]
