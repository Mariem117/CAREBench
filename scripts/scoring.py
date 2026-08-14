"""Per-file scoring for extraction and repair tasks."""

from __future__ import annotations

import json
from pathlib import Path

from sob_eval.aggregate import aggregate, cell_score, repair_score
from sob_eval.metrics import METRICS_VERSION, score_item

from scripts.repair_utils import compute_live_repair_metrics, extract_corrected_payload


def _estimate(metrics: dict, key: str, default: float = 0.0) -> float:
    return float(metrics.get(key, {}).get("estimate", default))


def score_extraction_file(path: Path, gold: dict[str, dict]) -> dict:
    """Score one prompt_only or reasoning_assisted JSONL dump."""
    item_scores = []
    nulls = 0
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        record_id = str(row.get("record_id", ""))
        example = gold.get(record_id)
        if example is None:
            continue
        raw = row.get("raw_output")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            nulls += 1
            item_scores.append(score_item(None, example))
        else:
            item_scores.append(score_item(raw, example))

    metrics = aggregate(item_scores, n_bootstrap=0)
    composites = cell_score(metrics)
    n = len(item_scores)
    return {
        "metrics_version": METRICS_VERSION,
        "n": n,
        "nulls": nulls,
        "coverage": 1.0 - (nulls / n if n else 0.0),
        "json_parse_success": _estimate(metrics, "json_parse_success"),
        "parse_pass": _estimate(metrics, "parse_pass"),
        "schema_pass": _estimate(metrics, "schema_pass"),
        "json_pass_rate": _estimate(metrics, "json_pass_rate"),
        "perfect_response": _estimate(metrics, "perfect_response"),
        "prr_pass": _estimate(metrics, "prr_pass"),
        "value_accuracy": _estimate(metrics, "value_accuracy"),
        "value_accuracy_hardened": _estimate(metrics, "value_accuracy_hardened"),
        "nested_field_accuracy": _estimate(metrics, "nested_field_accuracy"),
        "field_accuracy": _estimate(metrics, "field_accuracy"),
        "faithfulness": _estimate(metrics, "faithfulness"),
        "faithfulness_hardened": _estimate(metrics, "faithfulness_hardened"),
        "path_recall": _estimate(metrics, "path_recall"),
        "path_recall_hardened": _estimate(metrics, "path_recall_hardened"),
        "structure_coverage": _estimate(metrics, "structure_coverage"),
        "structure_coverage_hardened": _estimate(metrics, "structure_coverage_hardened"),
        "type_safety": _estimate(metrics, "type_safety"),
        "missing_rate": _estimate(metrics, "missing_rate"),
        "type_coercion": _estimate(metrics, "type_coercion"),
        "nesting_degradation": float(
            metrics.get("nesting_degradation", {}).get("estimate", 0.0)
        ),
        "nesting_retention": float(
            metrics.get("nesting_degradation", {}).get("retention", 0.0)
        ),
        "L1": composites["L1_compliance"],
        "L2": composites["L2_accuracy"],
        "L3": composites["L3_structure"],
        "cell_score": composites["cell_score"],
    }


def score_repair_file(path: Path, gold: dict[str, dict]) -> dict:
    """Score one zero_shot repair JSONL dump."""
    item_scores = []
    repair_exact_values: list[float] = []
    row_count = 0
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row_count += 1
        record = json.loads(line)
        record_id = str(record.get("record_id", ""))
        example = gold.get(record_id)
        if example is None:
            continue

        repair_exact = record.get("repair_exact")
        if repair_exact is not None and str(repair_exact).strip() != "":
            try:
                repair_exact_values.append(float(repair_exact))
            except (TypeError, ValueError):
                pass

        corrected = extract_corrected_payload(record.get("raw_output"))
        item_scores.append(score_item(None if corrected is None else corrected, example))

    metrics = aggregate([score for score in item_scores if score is not None], n_bootstrap=0)
    live = compute_live_repair_metrics(path)
    repair_em = live.get("repair_exact_match", 0.0)
    over_correction = live.get("over_correction_rate", 0.0)
    if not repair_em and repair_exact_values:
        repair_em = sum(repair_exact_values) / len(repair_exact_values)

    composites = repair_score(
        metrics,
        detection_f1=live.get("detection_f1", 0.0),
        localization_accuracy=live.get("localization_accuracy", 0.0),
        over_correction_rate=over_correction,
        repair_exact_match=repair_em,
    )
    return {
        "metrics_version": METRICS_VERSION,
        "rows": row_count,
        "parse_pass": _estimate(metrics, "parse_pass"),
        "schema_pass": _estimate(metrics, "schema_pass"),
        "prr_pass": _estimate(metrics, "prr_pass"),
        "field_accuracy": _estimate(metrics, "field_accuracy"),
        "nested_field_accuracy": _estimate(metrics, "nested_field_accuracy"),
        "path_recall": _estimate(metrics, "path_recall"),
        "missing_rate": _estimate(metrics, "missing_rate"),
        "type_safety": _estimate(metrics, "type_safety"),
        "type_coercion": _estimate(metrics, "type_coercion"),
        "repair_exact_match": round(repair_em, 6),
        "detection_f1": round(live.get("detection_f1", 0.0), 6),
        "over_correction_rate": round(over_correction, 6),
        "faithfulness": _estimate(metrics, "faithfulness"),
        "structure_coverage": _estimate(metrics, "structure_coverage"),
        "nesting_degradation": float(
            metrics.get("nesting_degradation", {}).get("estimate", 0.0)
        ),
        "nesting_retention": float(
            metrics.get("nesting_degradation", {}).get("retention", 0.0)
        ),
        "L1": composites["L1_compliance"],
        "L2": composites["L2_accuracy"],
        "L3": composites["L3_structure"],
        "cell_score": composites["repair_score"],
    }
