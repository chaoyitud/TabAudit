from __future__ import annotations

import pandas as pd
from sklearn.utils.multiclass import type_of_target

from tab_audit.modeling.backend import fit_predict_cv, result_to_dict
from tab_audit.utils.sampling import maybe_sample_rows


def _detect_task(y: pd.Series) -> str:
    non_null = y.dropna()
    if non_null.empty:
        return "regression"

    target_kind = type_of_target(non_null)
    if target_kind in {"binary", "multiclass"}:
        return "classification"
    if target_kind.startswith("continuous"):
        return "regression"

    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return "classification"
    return "regression"


def run_baseline_model(
    df: pd.DataFrame,
    target: str | None,
    max_rows_model: int,
    random_seed: int,
    device: str = "auto",
    model_backend_preference: list[str] | None = None,
    cv_folds: int = 5,
    cv_n_jobs: int = 1,
) -> dict:
    if target is None or target not in df.columns:
        return {"available": False, "reason": "no_target", "backend": "none", "device": device}

    work = maybe_sample_rows(df, max_rows_model, random_seed).copy()
    y = work[target]
    X = work.drop(columns=[target])

    if len(work) < 25:
        return {"available": False, "reason": "insufficient_rows_for_cv", "backend": "none", "device": device}

    if y.dropna().nunique() <= 1:
        return {"available": False, "reason": "target_not_variable", "backend": "none", "device": device}

    task = _detect_task(y)

    if task == "classification":
        # Safety fallback: if target looks continuous, do not use stratified CV.
        if type_of_target(y.dropna()) not in {"binary", "multiclass"}:
            task = "regression"

    if task == "classification":
        min_class_size = int(y.value_counts(dropna=True).min())
        if min_class_size < max(2, int(cv_folds)):
            return {
                "available": False,
                "reason": "insufficient_samples_per_class_for_cv",
                "backend": "none",
                "device": device,
            }

    result = fit_predict_cv(
        task_type=task,
        X=X,
        y=y,
        random_seed=random_seed,
        device=device,
        model_backend_preference=model_backend_preference or ["xgboost", "lightgbm", "catboost", "sklearn"],
        cv_folds=cv_folds,
        cv_n_jobs=cv_n_jobs,
    )
    payload = result_to_dict(result)
    payload["n_rows_modeled"] = int(len(work))
    return payload
