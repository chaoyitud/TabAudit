from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

    if task_type == "classification":
        estimator = LogisticRegression(max_iter=1000)
        if y.nunique(dropna=True) == 2:
            scoring = {"primary": "roc_auc", "accuracy": "accuracy"}
        else:
            scoring = {"primary": "f1_macro", "accuracy": "accuracy"}
        min_class = int(y.value_counts(dropna=True).min())
        n_splits = max(2, min(int(cv_folds), min_class, len(X)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    else:
        estimator = RandomForestRegressor(n_estimators=200, random_state=random_seed, n_jobs=1)
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
    primary_train = float(np.mean(results["train_primary"]))
    overfit_gap = float(primary_train - primary_cv)

    if task_type == "classification":
        norm = max(0.0, min(1.0, primary_cv))
        class_imbalance = float(1.0 - y.value_counts(normalize=True).max())
        return BaselineResult(
            available=True,
            task=task_type,
            backend="sklearn",
            device=device,
            primary_metric_cv=primary_cv,
            primary_metric_cv_std=primary_cv_std,
            overfit_gap=overfit_gap,
            training_time_sec=elapsed,
            baseline_score_norm=norm,
            class_imbalance=class_imbalance,
            accuracy_cv=float(np.mean(results["test_accuracy"])),
            warnings=[],
        )

    norm = max(0.0, min(1.0, (primary_cv + 1.0) / 2.0))
    return BaselineResult(
        available=True,
        task=task_type,
        backend="sklearn",
        device=device,
        primary_metric_cv=primary_cv,
        primary_metric_cv_std=primary_cv_std,
        overfit_gap=overfit_gap,
        training_time_sec=elapsed,
        baseline_score_norm=norm,
        class_imbalance=None,
        rmse_cv=float(-np.mean(results["test_rmse"])),
        warnings=[],
    )
