from __future__ import annotations

import numpy as np
import pandas as pd

from tab_audit.checks.unsup_learnability import run_rcpl_learnability


def test_rcpl_learnability_runs_without_supervised_target():
    n = 200
    rng = np.random.default_rng(42)
    a = rng.normal(size=n)
    b = a * 0.8 + rng.normal(scale=0.1, size=n)
    c = (a > 0).astype(int)
    d = np.where(c == 1, "yes", "no")

    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})

    res = run_rcpl_learnability(
        df,
        rcpl_target_cols=3,
        rcpl_repeats=2,
        rcpl_max_rows=500,
        rcpl_seed=42,
        rcpl_max_cat_card=200,
        rcpl_skip_id_threshold=0.98,
        rcpl_cv_folds=5,
        device="cpu",
        model_backend_preference=["sklearn"],
    )

    assert res["available"] is True
    assert 0.0 <= res["rcpl_score"] <= 100.0
    assert len(res["per_target"]) > 0
