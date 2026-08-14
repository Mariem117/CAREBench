"""Aggregation, CIs, composites (metrics_version 2.0)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .metrics import METRICS_VERSION

# ---------------------------------------------------------------------------
# Aggregation weight tables (must sum to 1.0 — tested in test_metrics_v2.py)
# ---------------------------------------------------------------------------
L1_WEIGHTS = {"parse_pass": 0.40, "schema_pass": 0.35, "structure_coverage_hardened": 0.25}

L2_EXTRACTION_WEIGHTS = {
    "value_accuracy_hardened": 0.35,
    "faithfulness_hardened": 0.15,
    "path_recall_hardened": 0.20,
    "type_safety": 0.10,
    "perfect_response": 0.20,
}

L2_REPAIR_WEIGHTS = {
    "value_accuracy_hardened": 0.20,
    "faithfulness_hardened": 0.10,
    "path_recall_hardened": 0.15,
    "type_safety": 0.10,
    "perfect_response": 0.10,
    "detection_f1": 0.15,
    "localization_accuracy": 0.10,
    "repair_exact_match": 0.10,
}

L3_WEIGHTS = {"nesting_retention_gated_by_l2": 1.0}

L3_REPAIR_WEIGHTS = L3_WEIGHTS

CELL_SCORE_WEIGHTS = {"L1": 0.30, "L2": 0.50, "L3": 0.20}
REPAIR_SCORE_WEIGHTS = {"L1": 0.20, "L2": 0.55, "L3": 0.25}
OVERALL_WEIGHTS = {
    "cell_score_prompt_only": 0.30,
    "cell_score_reasoning_assisted": 0.25,
    "repair_score": 0.45,
}


def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> dict:
    if total == 0:
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
    p = successes / total
    z = _norm_ppf(1 - alpha / 2)
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return {
        "estimate": round(float(p), 4),
        "ci_lower": round(float(max(0.0, centre - margin)), 4),
        "ci_upper": round(float(min(1.0, centre + margin)), 4),
        "n": total,
    }


def bootstrap_ci(
    scores,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    scores = np.asarray(list(scores), dtype=float)
    if not len(scores):
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
    estimate = round(float(np.mean(scores)), 4)
    if n_bootstrap <= 0:
        return {
            "estimate": estimate,
            "ci_lower": estimate,
            "ci_upper": estimate,
            "n": int(len(scores)),
        }
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(scores, len(scores), replace=True)))
        for _ in range(n_bootstrap)
    ]
    return {
        "estimate": estimate,
        "ci_lower": round(float(np.percentile(means, 100 * alpha / 2)), 4),
        "ci_upper": round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4),
        "n": int(len(scores)),
    }


def aggregate(item_scores: list[dict], n_bootstrap: int = 2000) -> dict:
    valid = [s for s in item_scores if s is not None]
    if not valid:
        return {"metrics_version": METRICS_VERSION}
    metrics: dict = {"metrics_version": METRICS_VERSION}
    binary_keys = (
        "json_parse_success",
        "parse_pass",
        "schema_pass",
        "json_pass_rate",
        "perfect_response",
        "prr_pass",
        "type_coercion",
    )
    for key in binary_keys:
        metrics[key] = wilson_ci(sum(bool(s.get(key)) for s in valid), len(valid))
    continuous_keys = (
        "value_accuracy",
        "faithfulness",
        "path_recall",
        "structure_coverage",
        "type_safety",
        "field_accuracy",
        "missing_rate",
        "nested_field_accuracy",
        "value_accuracy_hardened",
        "faithfulness_hardened",
        "path_recall_hardened",
        "structure_coverage_hardened",
        "coverage_gate",
    )
    for key in continuous_keys:
        metrics[key] = bootstrap_ci([float(s.get(key, 0.0)) for s in valid], n_bootstrap=n_bootstrap)

    metrics["nesting_degradation"] = nesting_degradation(valid)
    metrics["n_scored"] = len(valid)
    return metrics


def nesting_degradation(item_scores: list[dict]) -> dict:
    shallow, deep = [], []
    for s in item_scores:
        by = s.get("acc_by_depth") or {}
        if "1" in by:
            shallow.append(float(by["1"]))
        deeper = [float(v) for k, v in by.items() if int(k) >= 2]
        if deeper:
            deep.append(float(np.mean(deeper)))

    if not shallow:
        return {
            "estimate": 0.0,
            "shallow_acc": 0.0,
            "deep_acc": 0.0,
            "retention": 0.0,
            "n_shallow": 0,
            "n_deep": 0,
        }
    sh = float(np.mean(shallow))
    dp = float(np.mean(deep)) if deep else sh
    deg = sh - dp
    retention = (dp / sh) if sh > 1e-9 else 0.0
    return {
        "estimate": round(deg, 4),
        "shallow_acc": round(sh, 4),
        "deep_acc": round(dp, 4),
        "retention": round(retention, 4),
        "n_shallow": len(shallow),
        "n_deep": len(deep),
    }


def robustness_ratio(
    hard_metric: float,
    easy_metric: float,
    eps: float = 1e-9,
) -> float:
    if easy_metric < eps:
        return 0.0 if hard_metric < eps else 1.5
    return float(min(1.5, hard_metric / easy_metric))


def _e(metrics: dict, key: str) -> float:
    return float(metrics.get(key, {}).get("estimate", 0.0))


def _clamp_unit(x: float) -> float:
    """Keep composite layer scores on the 0–1 scale shown in the dashboard."""
    return float(min(1.0, max(0.0, x)))


def composite_l1(metrics: dict) -> float:
    return (
        _e(metrics, "parse_pass") * L1_WEIGHTS["parse_pass"]
        + _e(metrics, "schema_pass") * L1_WEIGHTS["schema_pass"]
        + _e(metrics, "structure_coverage_hardened") * L1_WEIGHTS["structure_coverage_hardened"]
    )


def composite_l2_extraction(metrics: dict) -> float:
    return sum(_e(metrics, key) * weight for key, weight in L2_EXTRACTION_WEIGHTS.items())


def composite_l2_repair(
    metrics: dict,
    *,
    detection_f1: float = 0.0,
    localization_accuracy: float = 0.0,
    repair_exact_match: float = 0.0,
) -> float:
    total = 0.0
    for key, weight in L2_REPAIR_WEIGHTS.items():
        if key == "detection_f1":
            total += detection_f1 * weight
        elif key == "localization_accuracy":
            total += localization_accuracy * weight
        elif key == "repair_exact_match":
            total += repair_exact_match * weight
        else:
            total += _e(metrics, key) * weight
    return total


def _effective_retention(metrics: dict) -> float:
    """Retention capped at 1 — deep cannot boost L3 above L2."""
    retention = float(metrics.get("nesting_degradation", {}).get("retention", 0.0))
    return float(min(1.0, max(0.0, retention)))


def composite_l3(metrics: dict, *, l2: float) -> float:
    """Depth robustness gated by accuracy: L3 ≤ L2 always."""
    return _clamp_unit(_effective_retention(metrics) * l2)


def composite_l3_repair(
    metrics: dict,
    *,
    l2: float,
    over_correction_rate: float = 0.0,
) -> float:
    """Repair L3: capped retention × L2 × (1 − over-correction)."""
    ocr = float(max(0.0, min(1.0, over_correction_rate)))
    return _clamp_unit(_effective_retention(metrics) * l2 * (1.0 - ocr))


def composite_l1_l2(metrics: dict) -> dict:
    """Legacy notebook helper — kept for backward compatibility."""
    l1 = composite_l1(metrics)
    l2 = composite_l2_extraction(metrics)
    return {
        "L1_compliance": round(l1, 4),
        "L2_accuracy": round(l2, 4),
        "composite_legacy_fixed": round(0.40 * l1 + 0.60 * l2, 4),
    }


def cell_score(metrics: dict) -> dict:
    """Extraction / reasoning-assisted cell score (metrics v2)."""
    l1 = composite_l1(metrics)
    l2 = composite_l2_extraction(metrics)
    l3 = composite_l3(metrics, l2=l2)
    score = (
        CELL_SCORE_WEIGHTS["L1"] * l1
        + CELL_SCORE_WEIGHTS["L2"] * l2
        + CELL_SCORE_WEIGHTS["L3"] * l3
    )
    return {
        "metrics_version": METRICS_VERSION,
        "L1_compliance": round(l1, 4),
        "L2_accuracy": round(l2, 4),
        "L3_structure": round(l3, 4),
        "cell_score": round(score, 4),
    }


def repair_score(
    metrics: dict,
    *,
    detection_f1: float = 0.0,
    localization_accuracy: float = 0.0,
    over_correction_rate: float = 0.0,
    repair_exact_match: float = 0.0,
) -> dict:
    l1 = composite_l1(metrics)
    l2 = composite_l2_repair(
        metrics,
        detection_f1=detection_f1,
        localization_accuracy=localization_accuracy,
        repair_exact_match=repair_exact_match,
    )
    l3 = composite_l3_repair(
        metrics,
        l2=l2,
        over_correction_rate=over_correction_rate,
    )
    score = (
        REPAIR_SCORE_WEIGHTS["L1"] * l1
        + REPAIR_SCORE_WEIGHTS["L2"] * l2
        + REPAIR_SCORE_WEIGHTS["L3"] * l3
    )
    return {
        "metrics_version": METRICS_VERSION,
        "L1_compliance": round(l1, 4),
        "L2_accuracy": round(l2, 4),
        "L3_structure": round(l3, 4),
        "repair_score": round(score, 4),
    }


def overall_benchmark_score(
    *,
    cell_score_prompt_only: float,
    cell_score_reasoning_assisted: float,
    repair_score_value: float,
) -> float:
    return (
        OVERALL_WEIGHTS["cell_score_prompt_only"] * cell_score_prompt_only
        + OVERALL_WEIGHTS["cell_score_reasoning_assisted"] * cell_score_reasoning_assisted
        + OVERALL_WEIGHTS["repair_score"] * repair_score_value
    )
