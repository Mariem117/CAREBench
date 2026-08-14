"""
SOB metrics aligned with the Structured Output Benchmark paper
(arXiv:2604.25359), metrics_version 2.0.

Leaf paths come from a single shared ``flatten_paths`` helper so Path Recall,
Structure Coverage, Faithfulness, and Perfect Response agree on what counts
as a leaf.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import jsonschema

METRICS_VERSION = "2.3"

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def extract_json(predicted: Any) -> Optional[str]:
    """Normalize model output to a JSON string (handles CoT / fences / objects)."""
    if predicted is None:
        return None
    if isinstance(predicted, (dict, list)):
        return json.dumps(predicted)
    text = str(predicted).strip()
    if not text:
        return None
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
    return match.group(0) if match else text


def flatten_paths(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts/lists to dot-paths with concrete array indices."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            if prefix:
                out[prefix] = obj
            return out
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_paths(v, path))
    elif isinstance(obj, list):
        if not obj:
            if prefix:
                out[prefix] = obj
            return out
        for i, v in enumerate(obj):
            path = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_paths(v, path))
    else:
        out[prefix or "$"] = obj
    return out


def path_depth(path: str) -> int:
    if path in ("", "$"):
        return 0
    return path.count(".") + 1


def canonicalize(obj: Any) -> Any:
    """Recursively key-sort dicts for Perfect Response comparison."""
    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [canonicalize(x) for x in obj]
    return obj


def _tokenize(text: Any) -> list[str]:
    s = str(text).lower()
    s = _ARTICLES.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s.split() if s else []


def token_f1(a: Any, b: Any) -> float:
    """Token-level F1 after normalization (SOB Faithfulness)."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(cb.values())
    rec = overlap / sum(ca.values())
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _json_type_name(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, int) and not isinstance(val, bool):
        return "integer"
    if isinstance(val, float):
        return "number"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return "array"
    if isinstance(val, dict):
        return "object"
    return type(val).__name__


def _normalize_value(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val).strip().lower()


def _schema_type_at_path(schema: dict, path: str) -> Optional[str]:
    """Resolve JSON Schema type for a dotted path (array indices wildcarded)."""
    node: Any = schema
    parts = path.split(".") if path and path != "$" else []
    for part in parts:
        if not isinstance(node, dict):
            return None
        if part.isdigit():
            node = node.get("items", {})
            continue
        props = node.get("properties") or {}
        if part in props:
            node = props[part]
            continue
        add = node.get("additionalProperties")
        if isinstance(add, dict):
            node = add
            continue
        return None
    if not isinstance(node, dict):
        return None
    t = node.get("type")
    if isinstance(t, list):
        for x in t:
            if x != "null":
                return str(x)
        return str(t[0]) if t else None
    return str(t) if t else None


def _types_compatible(pred_type: str, schema_type: Optional[str]) -> bool:
    if schema_type is None:
        return True
    if pred_type == schema_type:
        return True
    if schema_type == "number" and pred_type == "integer":
        return True
    return False


def resolve_source(example: dict) -> str:
    """Return modality/source label for coverage-gate selection."""
    for key in ("source", "modality", "input_modality", "media_type", "input_type"):
        val = example.get(key)
        if val is None:
            continue
        text = str(val).strip().lower()
        if text in {"text", "image", "audio"}:
            return text
        if "image" in text:
            return "image"
        if "audio" in text:
            return "audio"
    return "text"


def coverage_gate(f1_raw: float, source: str) -> float:
    """SOB coverage gate applied on raw Structure Coverage (F1 over paths)."""
    if f1_raw <= 0:
        return 0.0
    src = (source or "text").lower()
    if src in {"image", "audio"}:
        return min(1.0, (f1_raw / 0.90) ** 2)
    return 1.0 if f1_raw >= 0.95 else 0.0


def path_recall(gold_paths: set[str], pred_paths: set[str]) -> float:
    if not gold_paths:
        return 0.0
    return len(gold_paths & pred_paths) / len(gold_paths)


def structure_coverage(gold_paths: set[str], pred_paths: set[str]) -> float:
    if not gold_paths and not pred_paths:
        return 1.0
    overlap = gold_paths & pred_paths
    if not overlap:
        return 0.0
    prec = len(overlap) / len(pred_paths) if pred_paths else 0.0
    rec = len(overlap) / len(gold_paths) if gold_paths else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def faithfulness(gold_leaves: dict[str, Any], pred_leaves: dict[str, Any]) -> float:
    if not gold_leaves:
        return 0.0
    total = 0.0
    for path, ref_val in gold_leaves.items():
        if path in pred_leaves and pred_leaves[path] == ref_val:
            total += 1.0
        else:
            total += token_f1(ref_val, pred_leaves[path]) if path in pred_leaves else 0.0
    return total / len(gold_leaves)


def value_accuracy(gold_leaves: dict[str, Any], pred_leaves: dict[str, Any]) -> float:
    if not gold_leaves:
        return 0.0
    correct = sum(1 for path, ref_val in gold_leaves.items() if pred_leaves.get(path) == ref_val)
    return correct / len(gold_leaves)


def missing_rate_top_level(
    predicted: dict,
    schema: dict,
    reference: dict,
) -> float:
    req = schema.get("required") if isinstance(schema, dict) else None
    required = list(req) if isinstance(req, list) and req else list(reference.keys())
    if not required:
        return 0.0
    pred = predicted if isinstance(predicted, dict) else {}
    missing = sum(
        1 for field in required if field not in pred or pred.get(field) is None
    )
    return missing / len(required)


def type_coercion_binary(
    predicted: Any,
    reference: Any,
    schema: dict,
) -> bool:
    """True if any leaf is textually equal to gold but has the wrong JSON type."""
    pref = flatten_paths(reference)
    ppred = flatten_paths(predicted)
    for path, pred_val in ppred.items():
        if path not in pref:
            continue
        if isinstance(pred_val, (dict, list)):
            continue
        ref_val = pref[path]
        if isinstance(ref_val, (dict, list)):
            continue
        schema_t = _schema_type_at_path(schema, path)
        pred_t = _json_type_name(pred_val)
        if (
            _normalize_value(pred_val) == _normalize_value(ref_val)
            and not _types_compatible(pred_t, schema_t)
        ):
            return True
    return False


def type_safety(predicted: Any, schema: dict) -> float:
    leaves = flatten_paths(predicted)
    if not leaves:
        return 1.0
    ok = 0
    n = 0
    for path, val in leaves.items():
        if isinstance(val, (dict, list)):
            continue
        n += 1
        schema_t = _schema_type_at_path(schema, path)
        if _types_compatible(_json_type_name(val), schema_t):
            ok += 1
    return 1.0 if n == 0 else ok / n


def field_accuracy_top_level(predicted: dict, reference: dict, schema: dict) -> float:
    req = schema.get("required") if isinstance(schema, dict) else None
    required = list(req) if isinstance(req, list) and req else list(reference.keys())
    if not required:
        return 0.0
    pred = predicted if isinstance(predicted, dict) else {}
    ref = reference if isinstance(reference, dict) else {}
    correct = sum(1 for field in required if pred.get(field) == ref.get(field))
    return correct / len(required)


def nesting_stats(predicted: Any, reference: Any) -> dict:
    pref = flatten_paths(reference)
    ppred = flatten_paths(predicted)
    g_paths = set(pref)
    p_paths = set(ppred)
    pr = path_recall(g_paths, p_paths)
    sc = structure_coverage(g_paths, p_paths)
    va = value_accuracy(pref, ppred)
    faith = faithfulness(pref, ppred)

    by_depth: dict[int, list[int]] = {}
    for path, ref_val in pref.items():
        d = path_depth(path)
        ok = 1 if path in ppred and ppred[path] == ref_val else 0
        by_depth.setdefault(d, []).append(ok)

    return {
        "value_accuracy": va,
        "nested_field_accuracy": va,
        "nested_n": len(pref),
        "acc_by_depth": {
            str(d): round(sum(v) / len(v), 4) for d, v in sorted(by_depth.items()) if v
        },
        "path_recall": pr,
        "structure_coverage": sc,
        "faithfulness": faith,
    }


def apply_hardening(raw: dict, *, h: float, gate: float) -> dict:
    """Apply parse/schema hardening and coverage gate to semantic metrics."""
    return {
        "value_accuracy_hardened": raw["value_accuracy"] * h * gate,
        "faithfulness_hardened": raw["faithfulness"] * h * gate,
        "path_recall_hardened": raw["path_recall"] * h,
        "structure_coverage_hardened": raw["structure_coverage"] * h,
    }


def score_item(predicted_raw: Any, example: dict) -> Optional[dict]:
    """Score one prediction; returns None if gold/schema unusable."""
    try:
        schema = example["json_schema"]
        if isinstance(schema, str):
            schema = json.loads(schema)
        reference = example.get("validated_output") or example.get("ground_truth")
        if isinstance(reference, str):
            reference = json.loads(reference)
    except Exception:
        return None

    if not isinstance(reference, (dict, list)):
        return None

    source = resolve_source(example)
    extracted = extract_json(predicted_raw)
    if extracted is None:
        return _fail_score()

    try:
        predicted = json.loads(extracted)
        json_parse_success = True
    except Exception:
        return _fail_score()

    structured_root = isinstance(predicted, (dict, list))
    schema_pass = False
    if structured_root:
        try:
            jsonschema.validate(predicted, schema)
            schema_pass = True
        except jsonschema.ValidationError:
            schema_pass = False

    parse_pass = bool(json_parse_success)
    json_pass_rate = bool(parse_pass and structured_root and schema_pass)
    h = float(json_pass_rate)

    pred_obj = predicted if isinstance(predicted, dict) else {}
    ref_obj = reference if isinstance(reference, dict) else {}
    nest = nesting_stats(predicted if structured_root else {}, reference)
    gate = coverage_gate(nest["structure_coverage"], source)
    hardened = apply_hardening(nest, h=h, gate=gate)

    perfect = (
        canonicalize(predicted) == canonicalize(reference) if structured_root else False
    )
    missing = missing_rate_top_level(pred_obj, schema, ref_obj)
    coercion = type_coercion_binary(predicted if structured_root else {}, reference, schema)
    ts = type_safety(predicted, schema) if structured_root else 0.0
    fa = field_accuracy_top_level(pred_obj, ref_obj, schema)

    out = {
        "metrics_version": METRICS_VERSION,
        "json_parse_success": parse_pass,
        "parse_pass": parse_pass,
        "schema_pass": schema_pass,
        "json_pass_rate": json_pass_rate,
        "value_accuracy": nest["value_accuracy"],
        "faithfulness": nest["faithfulness"],
        "path_recall": nest["path_recall"],
        "structure_coverage": nest["structure_coverage"],
        "type_safety": ts,
        "perfect_response": perfect,
        "prr_pass": perfect,
        "field_accuracy": fa,
        "missing_rate": missing,
        "type_coercion": coercion,
        "nested_field_accuracy": nest["nested_field_accuracy"],
        "nested_n": nest["nested_n"],
        "acc_by_depth": nest["acc_by_depth"],
        "schema_depth": 0,
        "coverage_gate": gate,
        "source_modality": source,
        **hardened,
    }

    if not json_pass_rate:
        out.update(
            {
                "value_accuracy": 0.0,
                "faithfulness": 0.0,
                "path_recall": 0.0,
                "structure_coverage": 0.0,
                "perfect_response": False,
                "prr_pass": False,
                "value_accuracy_hardened": 0.0,
                "faithfulness_hardened": 0.0,
                "path_recall_hardened": 0.0,
                "structure_coverage_hardened": 0.0,
            }
        )

    return out


def _fail_score() -> dict:
    return {
        "metrics_version": METRICS_VERSION,
        "json_parse_success": False,
        "parse_pass": False,
        "schema_pass": False,
        "json_pass_rate": False,
        "value_accuracy": 0.0,
        "faithfulness": 0.0,
        "path_recall": 0.0,
        "structure_coverage": 0.0,
        "type_safety": 0.0,
        "perfect_response": False,
        "prr_pass": False,
        "field_accuracy": 0.0,
        "missing_rate": 1.0,
        "type_coercion": False,
        "nested_field_accuracy": 0.0,
        "nested_n": 0,
        "acc_by_depth": {},
        "schema_depth": 0,
        "coverage_gate": 0.0,
        "source_modality": "text",
        "value_accuracy_hardened": 0.0,
        "faithfulness_hardened": 0.0,
        "path_recall_hardened": 0.0,
        "structure_coverage_hardened": 0.0,
    }
