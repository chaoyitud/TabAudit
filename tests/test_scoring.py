from __future__ import annotations

import pytest

from tab_audit.config import ScoringWeights
from tab_audit.scoring.score import compute_scores


def test_weights_sum_to_one():
    with pytest.raises(ValueError):
        ScoringWeights(cleanliness=0.5, structure=0.5, learnability=0.5, label_quality=0.1)


def test_score_bounds():
    basic = {
        "n_rows": 100,
        "n_cols": 10,
        "missing_fraction_overall": 0.1,
        "duplicate_rows_fraction": 0.05,
        "constant_columns_count": 1,
        "all_missing_columns_count": 0,
        "high_cardinality_categorical_count": 0,
        "unique_id_like_columns_count": 0,
        "type_summary": {"numeric": 5, "categorical": 5, "bool": 0, "datetime": 0, "textish": 0},
    }
    clean = {"invalid_numeric_count": 0, "mixed_type_columns_count": 0}
    baseline = {
        "available": True,
        "baseline_score_norm": 0.8,
        "primary_metric_cv_std": 0.05,
        "overfit_gap": 0.1,
        "class_imbalance": 0.2,
    }
    label_q = {"available": True, "issue_fraction": 0.1}
    weights = {"cleanliness": 0.35, "structure": 0.25, "learnability": 0.25, "label_quality": 0.15}

    scores = compute_scores(basic, clean, baseline, None, label_q, weights)
    for k in ["cleanliness_score", "structure_score", "learnability_score", "label_quality_score", "quality_score"]:
        assert 0 <= scores[k] <= 100
