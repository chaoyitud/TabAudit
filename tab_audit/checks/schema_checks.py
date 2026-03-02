from __future__ import annotations

from collections import Counter

import pandas as pd


def run_schema_checks(
    df: pd.DataFrame,
    target: str | None,
    max_columns: int,
    min_rows_warn: int,
) -> dict:
    warnings: list[str] = []
    errors: list[str] = []

    counts = Counter(df.columns)
    dup_cols = [col for col, cnt in counts.items() if cnt > 1]
    if dup_cols:
        errors.append(f"duplicate column names found: {dup_cols[:10]}")

    if target is not None and target not in df.columns:
        errors.append(f"target column '{target}' does not exist")

    if df.shape[1] > max_columns:
        warnings.append(f"column count {df.shape[1]} exceeds max_columns={max_columns}")

    if df.shape[0] < min_rows_warn:
        warnings.append(f"row count {df.shape[0]} is below recommended minimum {min_rows_warn}")

    return {
        "warnings": warnings,
        "errors": errors,
    }
