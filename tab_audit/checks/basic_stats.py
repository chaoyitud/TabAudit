from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from tab_audit.io.read_tabular import infer_textish


def _type_bucket(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "bool"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return "numeric"
    if is_object_dtype(series) and infer_textish(series):
        return "textish"
    return "categorical"


def compute_basic_stats(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    type_summary: dict[str, int] = {"numeric": 0, "categorical": 0, "bool": 0, "datetime": 0, "textish": 0}

    missing_per_col = df.isna().mean().sort_values(ascending=False)
    duplicate_fraction = float(df.duplicated().mean()) if n_rows > 0 else 0.0
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    all_missing_cols = [c for c in df.columns if df[c].isna().all()]

    near_unique_cols = []
    high_card_cat = []
    for col in df.columns:
        s = df[col]
        bucket = _type_bucket(s)
        type_summary[bucket] = type_summary.get(bucket, 0) + 1
        nunique = s.nunique(dropna=True)
        ratio = nunique / max(1, n_rows)
        if ratio >= 0.95 and nunique > 20:
            near_unique_cols.append(col)
        if bucket in {"categorical", "textish"} and nunique > min(200, max(20, int(0.5 * n_rows))):
            high_card_cat.append(col)

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "type_summary": type_summary,
        "missing_fraction_overall": float(df.isna().mean().mean()) if n_cols else 0.0,
        "missing_top_columns": missing_per_col.head(10).to_dict(),
        "duplicate_rows_fraction": duplicate_fraction,
        "constant_columns_count": len(constant_cols),
        "constant_columns": constant_cols[:50],
        "all_missing_columns_count": len(all_missing_cols),
        "all_missing_columns": all_missing_cols[:50],
        "unique_id_like_columns_count": len(near_unique_cols),
        "unique_id_like_columns": near_unique_cols[:50],
        "high_cardinality_categorical_count": len(high_card_cat),
        "high_cardinality_categorical": high_card_cat[:50],
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
    }
