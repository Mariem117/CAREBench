"""Helpers for JSON repair (zero_shot) task scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from sob_eval.metrics import extract_json


def extract_corrected_payload(raw: Any) -> Any:
    """Return corrected_json from a repair wrapper, or the parsed object."""
    extracted = extract_json(raw)
    if extracted is None:
        return None
    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and "corrected_json" in parsed:
        return parsed["corrected_json"]
    return parsed


def compute_live_repair_metrics(path: Path) -> dict[str, float]:
    """Detection, localization, over-correction, and repair-EM from zero_shot JSONL."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return {}

    frame = pd.DataFrame(records)
    true_error = (
        frame.get("true_has_error", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
    )
    errored = frame[true_error]
    clean = frame[~true_error]

    tp = int((true_error & frame.get("pred_has_error", pd.Series()).eq(1)).sum())
    fp = int((~true_error & frame.get("pred_has_error", pd.Series()).eq(1)).sum())
    fn = int((true_error & frame.get("pred_has_error", pd.Series()).eq(0)).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if pd.notna(precision) and pd.notna(recall) and precision + recall
        else math.nan
    )

    def rate(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.mean()) if len(values) else 0.0

    return {
        "detection_accuracy": rate(frame["detection_correct"])
        if "detection_correct" in frame
        else 0.0,
        "detection_f1": float(f1) if pd.notna(f1) else 0.0,
        "localization_accuracy": rate(errored["localization_ok"])
        if "localization_ok" in errored
        else 0.0,
        "over_correction_rate": rate(clean["over_correction"])
        if "over_correction" in clean
        else 0.0,
        "repair_exact_match": rate(errored["repair_exact"])
        if "repair_exact" in errored
        else 0.0,
    }
