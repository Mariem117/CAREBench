#!/usr/bin/env python3
"""Score extraction and repair model outputs → results/*.csv.

Usage (from repo root, after pip install -e .):
    python -m scripts.score_all --all
    python -m scripts.score_all --task extraction --models qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.csv_utils import load_csv, upsert_rows, write_csv
from scripts.gold import load_extraction_gold
from scripts.paths import (
    EXTRACT_MODES,
    EXTRACTION_SUMMARY,
    PREDICTIONS_DIR,
    REPAIR_MODE,
    REPAIR_SUMMARY,
    RESULTS_DIR,
)
from scripts.scoring import score_extraction_file, score_repair_file


def discover_models() -> list[str]:
    models = {
        path.name.split("__", 1)[0]
        for path in PREDICTIONS_DIR.glob("*__prompt_only.jsonl")
    }
    return sorted(models)


def resolve_prediction(model: str, mode: str) -> Path | None:
    path = PREDICTIONS_DIR / f"{model}__{mode}.jsonl"
    return path if path.exists() else None


def score_extraction(
    gold: dict,
    models: list[str],
    rescore_all: bool,
) -> list[dict]:
    rows: list[dict] = []
    summary_path = EXTRACTION_SUMMARY
    existing = {(row["model"], row["mode"]) for row in load_csv(summary_path)}
    all_json_path = RESULTS_DIR / "all_layered.json"
    by_model: dict[str, dict] = {}
    if all_json_path.exists():
        by_model = json.loads(all_json_path.read_text(encoding="utf-8"))

    for model in models:
        for mode in EXTRACT_MODES:
            if not rescore_all and (model, mode) in existing:
                continue
            path = resolve_prediction(model, mode)
            if path is None:
                print(f"SKIP missing {model}__{mode}.jsonl", flush=True)
                continue
            print(f"\n>>> EXTRACTION  {model} / {mode}", flush=True)
            started = time.time()
            result = score_extraction_file(path, gold)
            print(
                f"    VA={result['value_accuracy']:.4f}  "
                f"JSONPass={result['json_pass_rate']:.4f}  "
                f"L3={result['L3']:.4f}  "
                f"Cell={result['cell_score']:.4f}  ({time.time() - started:.1f}s)",
                flush=True,
            )
            by_model.setdefault(model, {})[mode] = result
            rows.append({"model": model, "mode": mode, **result})

    if by_model:
        all_json_path.write_text(json.dumps(by_model, indent=2), encoding="utf-8")
    return rows


def score_repair(
    gold: dict,
    models: list[str],
    rescore_all: bool,
) -> list[dict]:
    rows: list[dict] = []
    summary_path = REPAIR_SUMMARY
    existing = {row["model"] for row in load_csv(summary_path)}

    for model in models:
        if not rescore_all and model in existing:
            continue
        path = resolve_prediction(model, REPAIR_MODE)
        if path is None:
            print(f"SKIP missing {model}__{REPAIR_MODE}.jsonl", flush=True)
            continue
        print(f"\n>>> REPAIR  {model} / {REPAIR_MODE}", flush=True)
        started = time.time()
        result = score_repair_file(path, gold)
        print(
            f"    n={result['rows']}  RepairEM={result['repair_exact_match']:.4f}  "
            f"OCR={result['over_correction_rate']:.4f}  "
            f"L3={result['L3']:.4f}  "
            f"Score={result['cell_score']:.4f}  ({time.time() - started:.1f}s)",
            flush=True,
        )
        rows.append({"model": model, "mode": REPAIR_MODE, **result})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CAREBench model predictions")
    parser.add_argument("--all", action="store_true", help="Rescore all models")
    parser.add_argument(
        "--task",
        choices=("extraction", "repair", "both"),
        default="both",
        help="Which task(s) to score (default: both)",
    )
    parser.add_argument("--models", nargs="*", help="Optional model id subset")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PREDICTIONS_DIR.exists():
        raise SystemExit(f"Predictions directory not found: {PREDICTIONS_DIR}")

    models = args.models or discover_models()
    print(f"Models: {', '.join(models)}", flush=True)
    gold = load_extraction_gold()

    if args.task in ("extraction", "both"):
        extract_rows = score_extraction(gold, models, args.all)
        merged = upsert_rows(load_csv(EXTRACTION_SUMMARY), extract_rows, ("model", "mode"))
        write_csv(
            EXTRACTION_SUMMARY,
            merged,
            preferred=[
                "model", "mode", "n", "nulls", "coverage",
                "parse_pass", "schema_pass", "json_pass_rate",
                "value_accuracy", "faithfulness", "path_recall",
                "structure_coverage", "type_safety", "perfect_response",
                "nesting_retention", "nesting_degradation",
                "L1", "L2", "L3", "cell_score", "metrics_version",
            ],
        )
        print(f"\nWrote {EXTRACTION_SUMMARY} ({len(merged)} rows)", flush=True)

    if args.task in ("repair", "both"):
        repair_rows = score_repair(gold, models, args.all)
        merged = upsert_rows(load_csv(REPAIR_SUMMARY), repair_rows, ("model", "mode"))
        write_csv(
            REPAIR_SUMMARY,
            merged,
            preferred=[
                "model", "mode", "rows",
                "L1", "L2", "L3", "cell_score", "metrics_version",
                "repair_exact_match", "detection_f1", "over_correction_rate",
                "path_recall", "faithfulness", "structure_coverage",
                "nesting_retention", "nesting_degradation",
            ],
        )
        print(f"Wrote {REPAIR_SUMMARY} ({len(merged)} rows)", flush=True)


if __name__ == "__main__":
    main()
