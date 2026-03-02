from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tab_audit.modeling.types import BaselineResult


def fit_predict_cv(
    task_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int,
    device: str,
    cv_folds: int = 5,
    cv_n_jobs: int = 1,
) -> BaselineResult:
    from catboost import CatBoostClassifier, CatBoostRegressor

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    preprocessor = ColumnTransformer(
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

    task_device = "GPU" if device == "cuda" else "CPU"
    if task_type == "classification":
        estimator = CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            random_seed=random_seed,
            task_type=task_device,
            thread_count=1,
            verbose=False,
        )
        if y.nunique(dropna=True) == 2:
            scoring = {"primary": "roc_auc", "accuracy": "accuracy"}
        else:
            scoring = {"primary": "f1_macro", "accuracy": "accuracy"}
        min_class = int(y.value_counts(dropna=True).min())
        n_splits = max(2, min(int(cv_folds), min_class, len(X)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    else:
        estimator = CatBoostRegressor(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            random_seed=random_seed,
            task_type=task_device,
            thread_count=1,
            verbose=False,
        )
        scoring = {"primary": "r2", "rmse": "neg_root_mean_squared_error"}
        n_splits = max(2, min(int(cv_folds), len(X)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    model = Pipeline([("prep", preprocessor), ("est", estimator)])
    started = time.time()
    if int(cv_n_jobs) != 1:
        with parallel_backend("threading"):
            results = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=int(cv_n_jobs),
                error_score="raise",
            )
    else:
        results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1,
            error_score="raise",
        )
    elapsed = time.time() - started

    primary_cv = float(np.mean(results["test_primary"]))
    primary_cv_std = float(np.std(results["test_primary"]))
    overfit_gap = float(np.mean(results["train_primary"]) - primary_cv)

    if task_type == "classification":
        return BaselineResult(
            available=True,
            task=task_type,
            backend="catboost",
            device=device,
            primary_metric_cv=primary_cv,
            primary_metric_cv_std=primary_cv_std,
            overfit_gap=overfit_gap,
            training_time_sec=elapsed,
            baseline_score_norm=max(0.0, min(1.0, primary_cv)),
            class_imbalance=float(1.0 - y.value_counts(normalize=True).max()),
            accuracy_cv=float(np.mean(results["test_accuracy"])),
            warnings=[],
        )

    return BaselineResult(
        available=True,
        task=task_type,
        backend="catboost",
        device=device,
        primary_metric_cv=primary_cv,
        primary_metric_cv_std=primary_cv_std,
        overfit_gap=overfit_gap,
        training_time_sec=elapsed,
        baseline_score_norm=max(0.0, min(1.0, (primary_cv + 1.0) / 2.0)),
        class_imbalance=None,
        rmse_cv=float(-np.mean(results["test_rmse"])),
        warnings=[],
    )
