from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineResult:
    available: bool
    task: str | None
    backend: str
    device: str
    primary_metric_cv: float | None
    primary_metric_cv_std: float | None
    overfit_gap: float | None
    training_time_sec: float
    baseline_score_norm: float | None
    class_imbalance: float | None
    accuracy_cv: float | None = None
    rmse_cv: float | None = None
    warnings: list[str] | None = None
