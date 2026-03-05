from __future__ import annotations

import numpy as np
import pandas as pd

from tab_audit.checks.baseline_model import run_baseline_model


def test_baseline_continuous_target_not_forced_to_classification():
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.integers(0, 5, size=n).astype(str),
            # Continuous target with limited unique values can be mis-detected by naive heuristics.
            "y": np.round(rng.normal(size=n), 1),
        }
    )

    result = run_baseline_model(
        df=df,
        target="y",
        max_rows_model=200_000,
        random_seed=42,
        device="cpu",
        model_backend_preference=["sklearn"],
        cv_folds=5,
    )

    assert result["available"] is True
    assert result["task"] == "regression"
