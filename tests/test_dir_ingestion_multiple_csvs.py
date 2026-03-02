from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from tab_audit.cli import app


def test_dir_ingestion_multiple_csvs(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "reports"
    data_dir.mkdir()

    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(data_dir / "one.csv", index=False)
    pd.DataFrame({"c": [10, 11], "d": [0, 1]}).to_csv(data_dir / "two.csv", index=False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "batch",
            "dir",
            "--path",
            str(data_dir),
            "--pattern",
            "*.csv",
            "--target",
            "none",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "one" / "report.json").exists()
    assert (out_dir / "two" / "report.json").exists()
    assert (out_dir / "leaderboard.csv").exists()

    payload = json.loads((out_dir / "one" / "report.json").read_text())
    assert payload["status"] in {"OK", "FAILED"}
