from __future__ import annotations

from typing import Any


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def clamp100(x: float) -> float:
    return max(0.0, min(100.0, x))


def cleanliness_score(basic: dict[str, Any], clean: dict[str, Any]) -> float:
    missing = float(basic.get("missing_fraction_overall", 0.0))
    dups = float(basic.get("duplicate_rows_fraction", 0.0))
    invalids = min(1.0, float(clean.get("invalid_numeric_count", 0)) / max(1, basic.get("n_rows", 1)))
    const_frac = float(basic.get("constant_columns_count", 0)) / max(1, basic.get("n_cols", 1))
    mixed_frac = float(clean.get("mixed_type_columns_count", 0)) / max(1, basic.get("n_cols", 1))
    all_missing_frac = float(basic.get("all_missing_columns_count", 0)) / max(1, basic.get("n_cols", 1))

    penalty = (
        0.35 * missing
        + 0.2 * dups
        + 0.15 * invalids
        + 0.15 * const_frac
        + 0.1 * mixed_frac
        + 0.05 * all_missing_frac
    )
    return clamp100((1.0 - clamp01(penalty)) * 100)


def structure_score(basic: dict[str, Any]) -> float:
    n_cols = basic.get("n_cols", 0)
    high_card = basic.get("high_cardinality_categorical_count", 0)
    id_like = basic.get("unique_id_like_columns_count", 0)
    types = basic.get("type_summary", {})
    diversity = sum(1 for v in types.values() if v > 0) / max(1, len(types))

    if n_cols == 0:
        return 0.0

    feature_penalty = 0.0
    if n_cols < 3:
        feature_penalty += 0.2
    if n_cols > 500:
        feature_penalty += min(0.4, (n_cols - 500) / 1000)

    card_penalty = min(0.3, high_card / max(1, n_cols))
    id_penalty = min(0.3, id_like / max(1, n_cols))

    score = (0.7 * diversity + 0.3 * (1 - clamp01(feature_penalty + card_penalty + id_penalty))) * 100
    return clamp100(score)


def learnability_score(baseline: dict[str, Any], unsup: dict[str, Any] | None, warnings: list[str]) -> float:
    if baseline.get("available"):
        norm = float(baseline.get("baseline_score_norm", 0.5))
        variance = float(baseline.get("primary_metric_cv_std", 0.0))
        overfit_gap = max(0.0, float(baseline.get("overfit_gap", 0.0)))
        imbalance = baseline.get("class_imbalance")
        imbalance_penalty = 0.0 if imbalance is None else max(0.0, 0.3 - float(imbalance))

        penalty = min(0.7, 0.4 * variance + 0.3 * overfit_gap + 0.3 * imbalance_penalty)
        return clamp100((clamp01(norm) * (1 - penalty)) * 100)

    if unsup and unsup.get("available"):
        warnings.extend(unsup.get("warnings", []))
        return clamp100(float(unsup.get("rcpl_score", 50.0)))

    if unsup and unsup.get("reason"):
        warnings.append(f"learnability fallback to 50 ({unsup.get('reason')})")
    else:
        warnings.append("learnability fallback to 50 (no supervised target or RCPL disabled)")
    return 50.0


def label_quality_score(label_q: dict[str, Any], warnings: list[str]) -> float:
    if not label_q.get("available"):
        warnings.extend(label_q.get("warnings", []))
        return 50.0

    issue_fraction = float(label_q.get("issue_fraction", 0.0))
    return clamp100((1.0 - clamp01(issue_fraction)) * 100)


def compute_scores(
    basic: dict[str, Any],
    clean: dict[str, Any],
    baseline: dict[str, Any],
    unsup: dict[str, Any] | None,
    label_q: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, Any]:
    warnings: list[str] = []
    c = cleanliness_score(basic, clean)
    s = structure_score(basic)
    learn = learnability_score(baseline, unsup, warnings)
    q = label_quality_score(label_q, warnings)

    total = (
        weights["cleanliness"] * c
        + weights["structure"] * s
        + weights["learnability"] * learn
        + weights["label_quality"] * q
    )

    return {
        "cleanliness_score": round(c, 4),
        "structure_score": round(s, 4),
        "learnability_score": round(learn, 4),
        "label_quality_score": round(q, 4),
        "quality_score": round(clamp100(total), 4),
        "warnings": warnings,
    }
