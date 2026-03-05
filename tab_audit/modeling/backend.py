from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import asdict

import numpy as np
import pandas as pd

from tab_audit.modeling import catboost_backend, lightgbm_backend, sklearn_backend, xgboost_backend
from tab_audit.modeling.types import BaselineResult


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


_CUDA_AVAILABLE: bool | None = None
_GPU_BACKEND_USABLE: dict[str, bool] = {}


def detect_cuda_available() -> bool:
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=2)
        _CUDA_AVAILABLE = proc.returncode == 0 and "GPU" in proc.stdout
        return _CUDA_AVAILABLE
    except Exception:
        _CUDA_AVAILABLE = False
        return False


def _probe_gpu_backend(name: str) -> bool:
    cached = _GPU_BACKEND_USABLE.get(name)
    if cached is not None:
        return cached

    try:
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y_cls = np.array([0, 1, 0, 1], dtype=int)

        if name == "xgboost":
            import xgboost as xgb

            model = xgb.XGBClassifier(
                n_estimators=1,
                max_depth=1,
                learning_rate=0.3,
                tree_method="hist",
                device="cuda",
                n_jobs=1,
                verbosity=0,
            )
            model.fit(X, y_cls)
            _GPU_BACKEND_USABLE[name] = True
            return True

        if name == "lightgbm":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=1,
                num_leaves=7,
                learning_rate=0.3,
                device_type="gpu",
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(X, y_cls)
            _GPU_BACKEND_USABLE[name] = True
            return True

        if name == "catboost":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                iterations=1,
                depth=2,
                learning_rate=0.3,
                task_type="GPU",
                thread_count=1,
                verbose=False,
            )
            model.fit(X, y_cls)
            _GPU_BACKEND_USABLE[name] = True
            return True
    except Exception:
        _GPU_BACKEND_USABLE[name] = False
        return False

    _GPU_BACKEND_USABLE[name] = False
    return False


def select_backend(device: str, preferences: list[str]) -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    wants_cuda = device == "cuda" or (device == "auto" and detect_cuda_available())

    if wants_cuda:
        for pref in preferences:
            if pref == "xgboost" and _module_available("xgboost"):
                if not _probe_gpu_backend("xgboost"):
                    warnings.append("xgboost installed but CUDA backend unusable; trying next backend")
                    continue
                return "xgboost", "cuda", warnings
            if pref == "lightgbm" and _module_available("lightgbm"):
                if not _probe_gpu_backend("lightgbm"):
                    warnings.append("lightgbm installed but CUDA backend unusable; trying next backend")
                    continue
                return "lightgbm", "cuda", warnings
            if pref == "catboost" and _module_available("catboost"):
                if not _probe_gpu_backend("catboost"):
                    warnings.append("catboost installed but CUDA backend unusable; trying next backend")
                    continue
                return "catboost", "cuda", warnings
        warnings.append("No GPU backend available; falling back to sklearn CPU backend")
        return "sklearn", "cpu", warnings

    if "sklearn" in preferences:
        return "sklearn", "cpu", warnings

    for pref in preferences:
        if pref == "xgboost" and _module_available("xgboost"):
            return "xgboost", "cpu", warnings
        if pref == "lightgbm" and _module_available("lightgbm"):
            return "lightgbm", "cpu", warnings
        if pref == "catboost" and _module_available("catboost"):
            return "catboost", "cpu", warnings

    return "sklearn", "cpu", warnings


def fit_predict_cv(
    task_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_seed: int,
    device: str,
    model_backend_preference: list[str],
    cv_folds: int = 5,
    cv_n_jobs: int = 1,
) -> BaselineResult:
    backend, resolved_device, warnings = select_backend(device, model_backend_preference)

    # Avoid running multiple GPU trainings concurrently (joblib parallel CV).
    if resolved_device == "cuda":
        cv_n_jobs = 1

    runner_map = {
        "sklearn": sklearn_backend.fit_predict_cv,
        "xgboost": xgboost_backend.fit_predict_cv,
        "lightgbm": lightgbm_backend.fit_predict_cv,
        "catboost": catboost_backend.fit_predict_cv,
    }
    runner = runner_map[backend]

    try:
        result = runner(
            task_type=task_type,
            X=X,
            y=y,
            random_seed=random_seed,
            device=resolved_device,
            cv_folds=cv_folds,
            cv_n_jobs=cv_n_jobs,
        )
        result.backend = backend
        result.device = resolved_device
        result.warnings = (result.warnings or []) + warnings
        return result
    except Exception as exc:
        fallback = sklearn_backend.fit_predict_cv(
            task_type=task_type,
            X=X,
            y=y,
            random_seed=random_seed,
            device="cpu",
            cv_folds=cv_folds,
            cv_n_jobs=cv_n_jobs,
        )
        fallback.backend = "sklearn"
        fallback.device = "cpu"
        fallback.warnings = (fallback.warnings or []) + warnings + [
            f"backend {backend} failed on {resolved_device}; fell back to sklearn ({exc.__class__.__name__})"
        ]
        return fallback


def result_to_dict(result: BaselineResult) -> dict:
    return asdict(result)
