from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def run_label_quality(
    df: pd.DataFrame,
    target: str | None,
    task: str | None,
    random_seed: int,
    save_row_ids: bool = False,
) -> dict:
    if target is None or target not in df.columns or task != "classification":
        return {
            "available": False,
            "issue_fraction": None,
            "issues_count": None,
            "warnings": ["label quality skipped (requires supervised classification target)"],
        }

    try:
        from cleanlab.filter import find_label_issues
    except ImportError:
        return {
            "available": False,
            "issue_fraction": None,
            "issues_count": None,
            "warnings": ["cleanlab not installed; label_quality_score fallback to 50"],
        }

    y = df[target]
    X = df.drop(columns=[target])
    if y.nunique(dropna=True) < 2:
        return {
            "available": False,
            "issue_fraction": None,
            "issues_count": None,
            "warnings": ["target has <2 classes; label quality skipped"],
        }
    min_class = int(y.value_counts(dropna=True).min())
    if min_class < 5:
        return {
            "available": False,
            "issue_fraction": None,
            "issues_count": None,
            "warnings": ["not enough samples per class for label quality CV; fallback to 50"],
        }

    y_filled = y.astype(str).fillna("__nan__")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_filled)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    model = Pipeline([("prep", prep), ("est", LogisticRegression(max_iter=1000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    pred_probs = cross_val_predict(model, X, y_encoded, cv=cv, method="predict_proba")
    issue_idx = find_label_issues(labels=y_encoded, pred_probs=pred_probs)

    payload = {
        "available": True,
        "issue_fraction": float(len(issue_idx) / len(df)),
        "issues_count": int(len(issue_idx)),
        "warnings": [],
    }
    if save_row_ids:
        payload["suspicious_row_indices"] = [int(i) for i in issue_idx[:5000]]
    return payload
