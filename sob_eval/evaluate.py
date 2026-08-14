"""
CLI: score SOB raw_predictions.json against a test set.

Examples
--------
  # Score an existing hard-question run against question_hard:
  uv run python -m sob_eval.evaluate \\
      --predictions benchmark_outputs/hard_google_gemma_3_4b_it_prompt_only_raw_predictions.json \\
      --testset question_hard \\
      --scenario prompt_only

  # Score Medium×Hard (build filter on the fly; preds must align by record_id or row):
  uv run python -m sob_eval.evaluate \\
      --predictions path/to/preds.json \\
      --testset medium_q_hard_s

  # Multi-cell summary with weighting:
  uv run python -m sob_eval.evaluate --manifest eval_manifest.json --scheme difficulty_tilted
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sob_eval.aggregate import aggregate, cell_score, composite_l1_l2, robustness_ratio
from sob_eval.metrics import METRICS_VERSION
from sob_eval.metrics import score_item
from sob_eval.test_sets import CORE_SUITE, TEST_SETS, load_testset
from sob_eval.weighting import WEIGHTING_SCHEMES, justify_weights, overall_sob_score


def _load_predictions(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list in {path}")
    return data


def _align(
    predictions: list[dict],
    dataset,
) -> list[tuple[Any, dict]]:
    """Align preds to dataset rows by record_id when possible, else by row index."""
    by_id = {}
    by_row = {}
    for p in predictions:
        if p.get("record_id") is not None:
            by_id[str(p["record_id"])] = p
        if "row" in p:
            by_row[int(p["row"])] = p

    paired = []
    for i, ex in enumerate(dataset):
        rid = str(ex.get("record_id", i))
        pred_row = by_id.get(rid) or by_row.get(i)
        raw = None if pred_row is None else pred_row.get("prediction")
        paired.append((raw, ex))
    return paired


def evaluate_predictions(
    predictions_path: str,
    testset_name: str,
    n_bootstrap: int = 2000,
    limit: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> dict:
    preds = _load_predictions(predictions_path)
    ds = load_testset(testset_name, hf_token=hf_token, limit=limit)
    paired = _align(preds, ds)

    item_scores = []
    null_preds = 0
    for raw, ex in paired:
        if raw is None:
            null_preds += 1
        item_scores.append(score_item(raw, ex))

    metrics = aggregate(item_scores, n_bootstrap=n_bootstrap)
    out = {
        "metrics_version": METRICS_VERSION,
        "testset": testset_name,
        "predictions": os.path.abspath(predictions_path),
        "n_dataset": len(ds),
        "n_predictions_file": len(preds),
        "n_null_predictions": null_preds,
        "metrics": metrics,
        "composites": {
            **composite_l1_l2(metrics),
            **cell_score(metrics),
        },
    }
    return out


def evaluate_manifest(manifest_path: str, scheme: str) -> dict:
    """
    Manifest JSON:
      {
        "cells": {
          "medium|medium": {"predictions": "...", "testset": "medium_q_medium_s"},
          "medium|hard":   {"predictions": "...", "testset": "medium_q_hard_s"},
          ...
        }
      }
    """
    with open(manifest_path) as f:
        man = json.load(f)
    cell_scores = {}
    details = {}
    for key, cfg in man["cells"].items():
        q, s = key.split("|", 1)
        result = evaluate_predictions(cfg["predictions"], cfg["testset"])
        details[key] = result
        cell_scores[(q, s)] = result["composites"]["cell_score"]

    overall = overall_sob_score(cell_scores, scheme=scheme)

    # Robustness ratios if both cells present
    rr = {}
    if ("medium", "medium") in cell_scores and ("medium", "hard") in cell_scores:
        fa_mm = details["medium|medium"]["metrics"]["field_accuracy"]["estimate"]
        fa_mh = details["medium|hard"]["metrics"]["field_accuracy"]["estimate"]
        rr["FA_mediumQ_hardS_over_mediumQ_mediumS"] = round(
            robustness_ratio(fa_mh, fa_mm), 4
        )
    if ("hard", "medium") in cell_scores and ("hard", "hard") in cell_scores:
        fa_hm = details["hard|medium"]["metrics"]["field_accuracy"]["estimate"]
        fa_hh = details["hard|hard"]["metrics"]["field_accuracy"]["estimate"]
        rr["FA_hardQ_hardS_over_hardQ_mediumS"] = round(
            robustness_ratio(fa_hh, fa_hm), 4
        )

    return {"overall": overall, "robustness_ratios": rr, "cells": details}


def _print_result(result: dict) -> None:
    m = result["metrics"]
    c = result["composites"]
    print("\n=== SOB evaluation ===")
    print(f"metrics_version: {result.get('metrics_version', '—')}")
    print(f"testset:     {result['testset']}")
    print(f"n:           {result['n_dataset']}  (null preds: {result['n_null_predictions']})")
    for k in (
        "parse_pass",
        "schema_pass",
        "json_pass_rate",
        "value_accuracy",
        "faithfulness",
        "path_recall",
        "structure_coverage",
        "type_safety",
        "perfect_response",
        "missing_rate",
    ):
        est = m.get(k, {})
        print(f"  {k:24s} {est.get('estimate', '—')}")
    nd = m.get("nesting_degradation", {})
    print(f"  {'nesting_degradation':24s} {nd.get('estimate', '—')}  "
          f"(retention={nd.get('retention', '—')})")
    print(f"  L1={c['L1_compliance']}  L2={c['L2_accuracy']}  "
          f"L3={c['L3_structure']}  cell_score={c['cell_score']}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="SOB evaluation pipeline")
    ap.add_argument("--predictions", help="raw_predictions.json path")
    ap.add_argument("--testset", choices=sorted(TEST_SETS), help="test set name")
    ap.add_argument("--manifest", help="multi-cell manifest JSON")
    ap.add_argument("--scheme", default="difficulty_tilted",
                    choices=sorted(WEIGHTING_SCHEMES))
    ap.add_argument("--out", help="write JSON results here")
    ap.add_argument("--csv", help="append one summary row to CSV")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--list-testsets", action="store_true")
    ap.add_argument("--show-weights", action="store_true")
    ap.add_argument("--scenario", default="", help="label only (prompt_only / reasoning_assisted)")
    args = ap.parse_args(argv)

    if args.list_testsets:
        for name in CORE_SUITE:
            s = TEST_SETS[name]
            print(f"{name:22s}  {s.dataset_id} / {s.config} / {s.split}")
            print(f"  filter q={s.question_difficulty} s={s.schema_complexity}")
            print(f"  {s.description}")
        return

    if args.show_weights:
        print(justify_weights())
        print("\nSchemes:")
        for name, w in WEIGHTING_SCHEMES.items():
            print(f"\n[{name}]")
            for (q, s), v in sorted(w.items()):
                print(f"  {q:6s} × {s:6s}  {v:.4f}")
        return

    if args.manifest:
        result = evaluate_manifest(args.manifest, args.scheme)
        print(json.dumps(result["overall"], indent=2))
        print("robustness_ratios:", result["robustness_ratios"])
    else:
        if not args.predictions or not args.testset:
            ap.error("--predictions and --testset required (or --manifest / --list-testsets)")
        result = evaluate_predictions(args.predictions, args.testset, limit=args.limit)
        result["scenario"] = args.scenario
        _print_result(result)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")

    if args.csv and not args.manifest:
        c = result["composites"]
        m = result["metrics"]
        row = {
            "testset": result["testset"],
            "scenario": args.scenario,
            "predictions": result["predictions"],
            "n": result["n_dataset"],
            "nulls": result["n_null_predictions"],
            "parse_pass": m["parse_pass"]["estimate"],
            "schema_pass": m["schema_pass"]["estimate"],
            "prr_pass": m["prr_pass"]["estimate"],
            "field_accuracy": m["field_accuracy"]["estimate"],
            "nested_field_accuracy": m["nested_field_accuracy"]["estimate"],
            "nesting_degradation": m["nesting_degradation"]["estimate"],
            "L1": c["L1_compliance"],
            "L2": c["L2_accuracy"],
            "L3": c["L3_structure"],
            "cell_score": c["cell_score"],
        }
        write_header = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"appended {args.csv}")


if __name__ == "__main__":
    main()
