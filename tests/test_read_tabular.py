from __future__ import annotations

import pandas as pd

from tab_audit.checks.schema_checks import run_schema_checks
from tab_audit.io.read_tabular import read_tabular


def test_read_tabular_mixed_types_and_target(tmp_path):
    path = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "num": [1, 2, 3],
            "cat": ["a", "2", "c"],
            "target": [0, 1, 0],
        }
    ).to_csv(path, index=False)

    df = read_tabular(path)
    assert list(df.columns) == ["num", "cat", "target"]

    schema = run_schema_checks(df, target="target", max_columns=500, min_rows_warn=2)
    assert schema["errors"] == []
