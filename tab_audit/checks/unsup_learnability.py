from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from tab_audit.checks.baseline_model import run_baseline_model
from tab_audit.utils.sampling import maybe_sample_rows


def _eligible_target_columns(
    df: pd.DataFrame,
    skip_id_threshold: float,
    max_cat_card: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    valid: list[str] = []
    skipped: list[dict[str, Any]] = []

    n_rows = max(1, len(df))
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        nunique = int(non_null.nunique())
        unique_ratio = float(nunique / n_rows)

        if nunique <= 1:
            skipped.append({"column": col, "reason": "constant_or_single_value"})
            continue
        if unique_ratio >= skip_id_threshold:
            skipped.append({"column": col, "reason": f"id_like_unique_ratio_{unique_ratio:.3f}"})
            continue

        is_numeric = pd.api.types.is_numeric_dtype(s)
        if not is_numeric and nunique > max_cat_card:
            skipped.append({"column": col, "reason": f"high_cardinality_{nunique}"})
            continue

        valid.append(col)

    return valid, skipped


def run_rcpl_learnability(
    df: pd.DataFrame,
    *,
    rcpl_target_cols: int,
    rcpl_repeats: int,
    rcpl_max_rows: int,
    rcpl_seed: int,
    rcpl_max_cat_card: int,
    rcpl_skip_id_threshold: float,
    rcpl_cv_folds: int,
    device: str,
    model_backend_preference: list[str],
) -> dict[str, Any]:
    import os

    started = time.time()
    warnings: list[str] = []

    sampled = maybe_sample_rows(df, rcpl_max_rows, rcpl_seed)
    valid_cols, skipped_cols = _eligible_target_columns(
        sampled,
        skip_id_threshold=rcpl_skip_id_threshold,
        max_cat_card=rcpl_max_cat_card,
    )

    if not valid_cols:
        return {
            "available": False,
            "reason": "no_eligible_rcpl_target_columns",
            "rcpl_score_norm": None,
            "rcpl_score": None,
            "stability_std": None,
            "per_target": [],
            "skipped_targets": skipped_cols,
            "n_valid_targets": 0,
            "warnings": ["No eligible RCPL targets after filtering; fallback learnability scoring will be used"],
            "training_time_sec": time.time() - started,
        }

    repeats = max(1, rcpl_repeats)
    target_cols = min(rcpl_target_cols, len(valid_cols))

    # Parallelize CV folds on CPU to reduce RCPL wall time; backend layer will force 1 on CUDA.
    cv_n_jobs = min(4, (os.cpu_count() or 1))

    rng = np.random.default_rng(rcpl_seed)
    repeat_scores: list[float] = []
    per_target: list[dict[str, Any]] = []

    for r in range(repeats):
        chosen = list(rng.choice(valid_cols, size=target_cols, replace=False))
        this_repeat_scores: list[float] = []

        for col in chosen:
            # Predict chosen target column from all remaining columns.
            res = run_baseline_model(
                sampled,
                target=col,
                # `sampled` already respects `rcpl_max_rows`; avoid re-sampling each proxy task.
                max_rows_model=int(len(sampled)),
                random_seed=rcpl_seed + r,
                device=device,
                model_backend_preference=model_backend_preference,
                cv_folds=rcpl_cv_folds,
                cv_n_jobs=cv_n_jobs,
            )

            norm = res.get("baseline_score_norm")
            available = bool(res.get("available"))
            if available and norm is not None:
                this_repeat_scores.append(float(norm))
            else:
                skipped_cols.append({"column": col, "reason": f"rcpl_unavailable_{res.get('reason', 'unknown')}"})

            per_target.append(
                {
                    "repeat": r,
                    "target_column": col,
                    "available": available,
                    "backend": res.get("backend"),
                    "device": res.get("device"),
                    "task": res.get("task"),
                    "baseline_score_norm": norm,
                    "primary_metric_cv": res.get("primary_metric_cv"),
                    "primary_metric_cv_std": res.get("primary_metric_cv_std"),
                    "overfit_gap": res.get("overfit_gap"),
                    "reason": res.get("reason"),
                }
            )

        if this_repeat_scores:
            repeat_scores.append(float(np.mean(this_repeat_scores)))

    if not repeat_scores:
        warnings.append("RCPL could not train any valid proxy targets")
        return {
            "available": False,
            "reason": "rcpl_all_proxy_tasks_failed",
            "rcpl_score_norm": None,
            "rcpl_score": None,
            "stability_std": None,
            "per_target": per_target,
            "skipped_targets": skipped_cols,
            "n_valid_targets": len(valid_cols),
            "warnings": warnings,
            "training_time_sec": time.time() - started,
        }

    base_norm = float(np.mean(repeat_scores))
    stability_std = float(np.std(repeat_scores)) if len(repeat_scores) > 1 else 0.0

    # Penalize unstable RCPL runs to reward reliable signal.
    stability_penalty = min(0.3, stability_std)
    rcpl_norm = max(0.0, min(1.0, base_norm * (1.0 - stability_penalty)))

    return {
        "available": True,
        "reason": None,
        "rcpl_score_norm": rcpl_norm,
        "rcpl_score": rcpl_norm * 100.0,
        "stability_std": stability_std,
        "repeat_scores": repeat_scores,
        "per_target": per_target,
        "skipped_targets": skipped_cols,
        "n_valid_targets": len(valid_cols),
        "warnings": warnings,
        "training_time_sec": time.time() - started,
        "settings": {
            "rcpl_target_cols": rcpl_target_cols,
            "rcpl_repeats": rcpl_repeats,
            "rcpl_max_rows": rcpl_max_rows,
            "rcpl_cv_folds": rcpl_cv_folds,
            "rcpl_seed": rcpl_seed,
            "rcpl_max_cat_card": rcpl_max_cat_card,
            "rcpl_skip_id_threshold": rcpl_skip_id_threshold,
        },
    }
