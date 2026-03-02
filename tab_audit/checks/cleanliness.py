from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _outlier_fraction_iqr(s: pd.Series) -> float:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if len(clean) < 20:
        return 0.0
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    low, high = q1 - 3 * iqr, q3 + 3 * iqr
    return float(((clean < low) | (clean > high)).mean())


def run_cleanliness_checks(df: pd.DataFrame) -> dict:
    invalid_numeric_count = 0
    mixed_type_columns: list[str] = []
    whitespace_only_count = 0
    outlier_columns: dict[str, float] = {}
    datetime_parse_issue_columns: dict[str, float] = {}

    for col in df.columns:
        s = df[col]
        if is_numeric_dtype(s):
            arr = pd.to_numeric(s, errors="coerce")
            invalid_numeric_count += int(np.isinf(arr).sum())
            outlier_frac = _outlier_fraction_iqr(s)
            if outlier_frac > 0.05:
                outlier_columns[col] = outlier_frac
        else:
            non_null = s.dropna()
            if not non_null.empty:
                as_str = non_null.astype(str)
                whitespace_only_count += int(as_str.str.fullmatch(r"\s*").sum())
                numeric_like = pd.to_numeric(as_str, errors="coerce").notna().mean()
                if 0.1 < numeric_like < 0.9:
                    mixed_type_columns.append(col)
                parsed_dt = pd.to_datetime(as_str, errors="coerce", utc=True)
                parse_fail_rate = float(parsed_dt.isna().mean())
                if 0.2 < parse_fail_rate < 0.95 and as_str.str.contains(r"[-/:]", regex=True).mean() > 0.3:
                    datetime_parse_issue_columns[col] = parse_fail_rate

    return {
        "invalid_numeric_count": int(invalid_numeric_count),
        "whitespace_only_count": int(whitespace_only_count),
        "mixed_type_columns_count": len(mixed_type_columns),
        "mixed_type_columns": mixed_type_columns[:50],
        "extreme_outlier_columns": outlier_columns,
        "datetime_parse_issue_columns": datetime_parse_issue_columns,
    }
