"""
Combined Ollama benchmark over TWO datasets, one model at a time.

For every model in MODEL_CONFIGS the script runs, in order, then unloads:

  DATASET 1  (generation task)   interfaze-ai/sob  (single parquet in data/)
      scenario "prompt_only"        -> plain JSON extraction
      scenario "reasoning_assisted" -> manual chain-of-thought then JSON

  DATASET 2  (verify / repair task) MoetezF/SOB-with-errors-injection
      scenario "zero_shot"          -> detect / locate / repair, with scoring
                                       (detection, localization, repair_exact,
                                        schema_ok, over_correction)

Dataset-2 prompts, eval-record shape, per-row JSONL fields, and summary files
match run_medschema_benchmark.py so outputs are interchangeable for analysis.
Only the inference backend differs (Ollama HTTP).

Design points:
  * Ollama HTTP backend (no transformers / no vLLM).
  * think = false on EVERY request (fastest path).
  * 4 concurrent requests per model (asyncio + aiohttp), tunable via --concurrency.
  * Model-by-model: a model finishes BOTH datasets before the next model loads.
  * Explicit unload between models via keep_alive=0.
  * Both datasets cached locally on first download.
  * JSONL checkpointing + resume by eval_id (same as medschema).

Usage:
    python test_models.py                       # all models, both datasets
    python test_models.py --model qwen3-8b      # single model alias
    python test_models.py --limit 20            # first 20 rows (smoke)
    python test_models.py --datasets ds1        # only dataset 1
    python test_models.py --datasets ds2        # only dataset 2
    python test_models.py --concurrency 4

Outputs JSONL under data/predictions/ ({alias}__{mode}.jsonl).
Score them with: python -m scripts.score_all --all
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

# Disable the Xet download backend by default — it can fail with
# "Unable to parse string as hex hash value" on some datasets. The classic
# HTTPS path is slower but reliable. Override by exporting HF_HUB_DISABLE_XET=0.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import aiohttp
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# All 10 CAREBench models. `tag` = Ollama model name (`ollama list`); `alias` = output filename prefix.
MODEL_CONFIGS = [
    {"tag": "deepseek-r1:1.5b", "alias": "deepseek-r1-1.5b"},
    {"tag": "gemma3:4b", "alias": "gemma3-4b"},
    {"tag": "granite4.1:8b", "alias": "granite-4.1-8b"},
    {"tag": "llama3.2:3b", "alias": "llama-3.2-3b"},
    {"tag": "llama3.1:8b", "alias": "llama3.1-8b"},
    {"tag": "ministral-3:3b-instruct-2512-q4_K_M", "alias": "ministral-3-3b"},
    {"tag": "numind/nuextract3:Q4_K_M", "alias": "nuextract-3-4b"},
    {"tag": "phi4-mini", "alias": "phi-4-mini"},
    {"tag": "qwen3:8b", "alias": "qwen3-8b"},
    {"tag": "qwen3.5:9b", "alias": "qwen3.5-9b"},
]

# ---- Dataset 1: generation task -----------------------------------------
# interfaze-ai/sob — a single parquet file in the data/ folder. Rows carry
# records / images / data columns; we ONLY use the `data` payload.
DS1_ID = os.environ.get("DS1_ID", "interfaze-ai/sob")
DS1_PARQUET = os.environ.get("DS1_PARQUET", "data/test-00000-of-00001.parquet")
DS1_SNAPSHOT = Path(os.environ.get("DS1_SNAPSHOT", "./ds1_snapshot"))

# ---- Dataset 2: verify / repair task ------------------------------------
DS2_ID = os.environ.get("HF_DATASET_ID", "MoetezF/SOB-with-errors-injection")
DS2_SNAPSHOT = Path(os.environ.get("DS2_SNAPSHOT", "./SOB-with-errors-injection"))
DS2_CLEAN_FRACTION = 0.25
DS2_MAX_CONTEXT_CHARS = 2500

DEFAULT_CONCURRENCY = 4       # must be <= server OLLAMA_NUM_PARALLEL
DEFAULT_BATCH_SIZE = 8        # rows per checkpoint window

MAX_NEW_TOKENS_PROMPT_ONLY = 200
MAX_NEW_TOKENS_REASONING = 600
MAX_NEW_TOKENS_VERIFY = 512

SEED = 42
REQUEST_TIMEOUT_S = 300
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "data/predictions"))

# Identical to run_medschema_benchmark.SYSTEM
SYSTEM = (
    "You verify and repair JSON answers. Given a CONTEXT, QUESTION, JSON SCHEMA, and a "
    "candidate JSON, decide whether the candidate JSON is fully correct and grounded in "
    "the context. If it is wrong, return the corrected JSON. If it is already correct, "
    "return it unchanged and report no error. Respond with ONLY a single JSON object of "
    'the form: {"has_error": <bool>, "location": <string or null>, "corrected_json": <object>}. '
    "No prose, no markdown, no code fences."
)


# ─────────────────────────────────────────────────────────────
# HF token
# ─────────────────────────────────────────────────────────────
def resolve_hf_token() -> str | None:
    try:
        from huggingface_hub import get_token
        tok = get_token()
    except ImportError:
        from huggingface_hub import HfFolder
        tok = HfFolder.get_token()
    tok = tok or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        try:
            from huggingface_hub import login
            login(token=tok)
        except Exception:
            pass
        print("HF token found.")
    else:
        print("No HF token — only ungated repos will work.")
    return tok


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# ─────────────────────────────────────────────────────────────
# Dataset 1 — generation task
# ─────────────────────────────────────────────────────────────
def ds1_build_prompt(example: dict) -> str:
    return (
        f"Schema:\n{example['json_schema']}\n\n"
        f"Context:\n{example['context']}\n\n"
        f"Question: {example['question']}\n\n"
        "Return ONLY valid JSON that matches the schema, using information from the context. No explanation."
    )


def ds1_build_prompt_cot(example: dict) -> str:
    return (
        f"Schema:\n{example['json_schema']}\n\n"
        f"Context:\n{example['context']}\n\n"
        f"Question: {example['question']}\n\n"
        "First, briefly reason step-by-step using the context to determine each schema field's value. "
        "Then, on a new line, output ONLY the final strictly valid JSON conforming to the schema, "
        "with no additional text after it."
    )


def _ds1_extract_payload(row: dict) -> dict:
    """Pull the QA payload out of the interfaze-ai/sob `data` column.

    `data` may be a dict already, or a JSON string of one. We ignore the
    `records` and `images` columns entirely, per requirements.
    """
    payload = row.get("data", None)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = None
    if isinstance(payload, dict):
        return payload
    # Fall back to the top-level row if `data` isn't present/parseable.
    return row


def ds1_load_records(token: str | None, n: int | None = None) -> list[dict]:
    """Download the single parquet from data/, read it, normalize to records.

    Returns list of dicts: eval_id, record_id, json_schema, context, question.
    Cached locally as parquet so later runs need no network.
    """
    from huggingface_hub import hf_hub_download

    DS1_SNAPSHOT.mkdir(parents=True, exist_ok=True)
    local_parquet = DS1_SNAPSHOT / "ds1.parquet"

    if local_parquet.exists():
        df = pd.read_parquet(local_parquet)
        print(f"[ds1] loaded local snapshot {local_parquet} ({len(df):,} rows)")
    else:
        print(f"[ds1] downloading {DS1_ID}:{DS1_PARQUET} from HF (first time only) ...")
        path = hf_hub_download(
            repo_id=DS1_ID, filename=DS1_PARQUET,
            repo_type="dataset", token=token,
        )
        df = pd.read_parquet(path)
        df.to_parquet(local_parquet)
        print(f"[ds1] saved snapshot {local_parquet} ({len(df):,} rows). No network next time.")
        print(f"[ds1] parquet columns: {list(df.columns)}")

    records = []
    for i, row in enumerate(df.to_dict(orient="records")):
        payload = _ds1_extract_payload(row)

        def pick(*names):
            for src in (payload, row):
                if isinstance(src, dict):
                    for nm in names:
                        if src.get(nm) not in (None, ""):
                            return src[nm]
            return ""

        rid = pick("record_id", "id", "uid") or f"ds1_{i}"
        records.append({
            "eval_id": str(rid),
            "record_id": str(rid),
            "json_schema": pick("json_schema", "schema"),
            "context": pick("context", "passage", "text"),
            "question": pick("question", "query"),
        })

    if n is not None:
        records = records[:n]
    return records


# ─────────────────────────────────────────────────────────────
# Dataset 2 — verify / repair (same shapes as run_medschema_benchmark.py)
# ─────────────────────────────────────────────────────────────
def _parses(s) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _trim_ctx(text: str, max_chars: int) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"


def _find_table_file(root: Path) -> Path | None:
    names = (
        "sob_text_test_with_errors.parquet",
        "sob_text_test_with_errors.jsonl",
        "data.parquet",
        "train.parquet",
        "test.parquet",
    )
    for name in names:
        p = root / name
        if p.is_file():
            return p
    parquets = sorted(root.rglob("*.parquet"))
    if parquets:
        return parquets[0]
    jsonls = sorted(root.rglob("*.jsonl"))
    if jsonls:
        return jsonls[0]
    return None


def ds2_download(token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    DS2_SNAPSHOT.mkdir(parents=True, exist_ok=True)
    existing = _find_table_file(DS2_SNAPSHOT)
    if existing:
        print(f"[ds2] using cached table {existing}")
        return existing

    print(f"[ds2] downloading {DS2_ID} from HF (first time only) ...")
    snapshot_download(
        repo_id=DS2_ID, repo_type="dataset",
        local_dir=str(DS2_SNAPSHOT), token=token,
    )
    found = _find_table_file(DS2_SNAPSHOT)
    if not found:
        raise FileNotFoundError(f"No parquet/jsonl table found in {DS2_SNAPSHOT}")
    print(f"[ds2] snapshot ready: {found}")
    return found


def ds2_load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported data file: {path}")


def build_eval_set(
    raw: pd.DataFrame,
    n_eval: int | None,
    clean_fraction: float,
    seed: int,
    max_context_chars: int,
) -> list[dict]:
    """Identical record shape to run_medschema_benchmark.build_eval_set."""
    required = {"record_id", "context", "question", "json_schema", "validated_output", "errored_json"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    valid = raw[raw["validated_output"].map(_parses) & raw["errored_json"].map(_parses)].copy()
    valid = valid.reset_index(drop=True)
    print(f"[ds2] structurally valid rows: {len(valid)} / {len(raw)}")

    if n_eval is None or n_eval >= len(valid):
        errored_df = valid.copy()
        n_clean = int(round(len(errored_df) * clean_fraction / max(1e-9, (1 - clean_fraction))))
        n_clean = min(n_clean, len(valid))
        clean_df = valid.sample(n_clean, random_state=seed).reset_index(drop=True)
    else:
        frac = n_eval / len(valid)
        if "error_difficulty" in valid.columns:
            errored_df = (
                valid.groupby("error_difficulty", group_keys=False)
                .sample(frac=frac, random_state=seed)
                .reset_index(drop=True)
            )
        else:
            errored_df = valid.sample(frac=frac, random_state=seed).reset_index(drop=True)
        errored_ids = set(errored_df["record_id"])
        remaining = valid[~valid["record_id"].isin(errored_ids)].reset_index(drop=True)
        n_clean = int(round(len(errored_df) * clean_fraction / max(1e-9, (1 - clean_fraction))))
        n_clean = min(n_clean, len(remaining))
        clean_df = remaining.sample(max(0, n_clean), random_state=seed).reset_index(drop=True)

    def make_rec(row, is_clean: bool) -> dict:
        return {
            "eval_id": row["record_id"] + ("__clean" if is_clean else "__err"),
            "record_id": row["record_id"],
            "context": _trim_ctx(row["context"], max_context_chars),
            "question": str(row["question"]),
            "json_schema": row["json_schema"],
            "gold": row["validated_output"],
            "candidate": row["validated_output"] if is_clean else row["errored_json"],
            "true_has_error": (not is_clean),
            "true_location": None if is_clean else row.get("error_location"),
            "error_difficulty": "clean" if is_clean else row.get("error_difficulty", "unknown"),
            "error_source": "clean" if is_clean else row.get("error_source", "unknown"),
        }

    eval_records = [make_rec(r, False) for _, r in errored_df.iterrows()]
    eval_records += [make_rec(r, True) for _, r in clean_df.iterrows()]
    random.Random(seed).shuffle(eval_records)
    print(
        f"[ds2] eval set: {len(eval_records)} rows "
        f"({len(errored_df)} errored + {len(clean_df)} clean)"
    )
    return eval_records


def user_block(rec: dict) -> str:
    return (
        f"CONTEXT:\n{rec['context']}\n\n"
        f"QUESTION:\n{rec['question']}\n\n"
        f"JSON SCHEMA:\n{rec['json_schema']}\n\n"
        f"CANDIDATE JSON:\n{rec['candidate']}"
    )


def build_messages(rec: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_block(rec)},
    ]


def parse_output(text: str):
    if not text:
        return None
    t = text.strip()
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1].strip()
    else:
        t = t.replace("<think>", "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _norm(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _schema_of(rec):
    s = rec["json_schema"]
    return json.loads(s) if isinstance(s, str) else s


def _path_tokens(loc):
    if loc is None:
        return None
    if isinstance(loc, str):
        try:
            loc = json.loads(loc)
        except Exception:
            pass
    if isinstance(loc, (list, tuple)):
        toks = [str(x) for x in loc]
    else:
        toks = re.split(r"[.\[\]'\"\s]+", str(loc))
    return tuple(t for t in toks if t not in ("", "$", "root"))


def score_row(rec, raw_text):
    """Same signals dict as run_medschema_benchmark.score_row."""
    from jsonschema import Draft7Validator

    pred = parse_output(raw_text)
    out = {
        "valid_json": int(isinstance(pred, dict)),
        "pred_has_error": None,
        "detection_correct": None,
        "schema_ok": None,
        "repair_exact": None,
        "localization_ok": None,
        "over_correction": None,
    }
    if not isinstance(pred, dict):
        out["detection_correct"] = 0
        if rec["true_has_error"]:
            out["schema_ok"] = 0
            out["repair_exact"] = 0
        else:
            out["over_correction"] = 0
        return out, pred

    pred_has_error = bool(pred.get("has_error"))
    out["pred_has_error"] = int(pred_has_error)
    out["detection_correct"] = int(pred_has_error == rec["true_has_error"])
    corrected = pred.get("corrected_json")
    gold = json.loads(rec["gold"])

    if corrected is not None:
        try:
            Draft7Validator(_schema_of(rec)).validate(corrected)
            out["schema_ok"] = 1
        except Exception:
            out["schema_ok"] = 0

    if rec["true_has_error"]:
        out["repair_exact"] = int(corrected is not None and _norm(corrected) == _norm(gold))
        if pred_has_error:
            pt, tt = _path_tokens(pred.get("location")), _path_tokens(rec["true_location"])
            if pt is not None and tt is not None:
                out["localization_ok"] = int(pt == tt or pt[-1:] == tt[-1:])
            else:
                out["localization_ok"] = 0
    else:
        changed = corrected is not None and _norm(corrected) != _norm(gold)
        out["over_correction"] = int(pred_has_error or changed)
    return out, pred


# ─────────────────────────────────────────────────────────────
# JSONL I/O (same layout as run_medschema_benchmark.py)
# ─────────────────────────────────────────────────────────────
def result_path(alias: str, mode: str) -> Path:
    return OUTPUT_DIR / f"{safe_name(alias)}__{mode}.jsonl"


def done_marker_path(alias: str, mode: str) -> Path:
    return OUTPUT_DIR / "_done" / f"{safe_name(alias)}__{mode}.done"


def rewrite_rows(path: Path, rows: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_saved_rows(path: Path) -> dict:
    rows = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("raw_output") is not None and row.get("eval_id") is not None:
                    rows[row["eval_id"]] = row
    return rows


def write_eval_manifest(eval_records: list[dict]) -> None:
    manifest = OUTPUT_DIR / "eval_manifest.jsonl"
    if manifest.exists():
        return
    with manifest.open("w") as f:
        for rec in eval_records:
            f.write(
                json.dumps(
                    {
                        k: rec[k]
                        for k in (
                            "eval_id",
                            "record_id",
                            "true_has_error",
                            "error_difficulty",
                            "error_source",
                        )
                    }
                )
                + "\n"
            )
    print(f"[ds2] wrote {manifest.name}")


def write_model_summary(alias: str, rows: dict) -> None:
    vals = list(rows.values())
    summary = {
        "zero_shot": {
            "n": len(vals),
            "valid_json": float(pd.Series([r.get("valid_json") for r in vals]).mean()) if vals else None,
            "repair_exact": float(
                pd.Series([r.get("repair_exact") for r in vals if r.get("repair_exact") is not None]).mean()
            )
            if vals
            else None,
            "detection_correct": float(
                pd.Series(
                    [r.get("detection_correct") for r in vals if r.get("detection_correct") is not None]
                ).mean()
            )
            if vals
            else None,
        }
    }
    out = OUTPUT_DIR / f"{safe_name(alias)}__summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out.name}: {summary}")


def write_summary_csv() -> pd.DataFrame:
    rows = []
    for p in sorted(OUTPUT_DIR.glob("*__zero_shot.jsonl")):
        alias = p.stem.replace("__zero_shot", "")
        df = pd.read_json(p, lines=True)
        if df.empty:
            continue

        def rate(col):
            s = df[col].dropna()
            return float(s.mean()) if len(s) else float("nan")

        rows.append(
            {
                "model": alias,
                "mode": "zero_shot",
                "n": len(df),
                "json_validity": rate("valid_json"),
                "schema_conformance": rate("schema_ok"),
                "repair_exact_match": rate("repair_exact"),
                "detection_acc": rate("detection_correct"),
                "localization_acc": rate("localization_ok"),
                "over_correction_rate": rate("over_correction"),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["model"]) if rows else pd.DataFrame()
    if len(summary):
        out = OUTPUT_DIR / "summary_metrics.csv"
        summary.to_csv(out, index=False)
        print(summary.to_string(index=False))
        print("Saved", out)
    else:
        print("No zero_shot result files yet.")
    return summary


# ─────────────────────────────────────────────────────────────
# Ollama HTTP (async, think=false)
# ─────────────────────────────────────────────────────────────
async def ollama_chat(session, model, messages, max_new_tokens) -> dict:
    """`messages` is a full chat list (system+user) OR a single user string."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": max_new_tokens},
        "keep_alive": "1h",
    }
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.post(
                f"{OLLAMA_URL}/api/chat", json=payload,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_err}")


async def warm_up(session, model) -> None:
    print(f"  warming up {model} ...", flush=True)
    t0 = time.perf_counter()
    await ollama_chat(session, model, "ok", max_new_tokens=1)
    print(f"  warm-up done in {time.perf_counter() - t0:.1f}s")


async def unload(session, model) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "bye"}],
        "stream": False,
        "options": {"num_predict": 1},
        "keep_alive": 0,
    }
    try:
        async with session.post(
            f"{OLLAMA_URL}/api/chat", json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            await resp.read()
        print(f"  unloaded {model} (keep_alive=0)")
    except Exception as exc:
        print(f"  WARN: unload {model} failed: {exc}")


# ─────────────────────────────────────────────────────────────
# Scenario runner — JSONL + eval_id resume (medschema-compatible)
# ─────────────────────────────────────────────────────────────
async def run_scenario(
    session,
    cfg: dict,
    mode: str,
    records: list[dict],
    build_msg,
    make_row,
    max_new_tokens: int,
    concurrency: int,
    batch_size: int,
) -> dict:
    alias, tag = cfg["alias"], cfg["tag"]
    path = result_path(alias, mode)
    marker = done_marker_path(alias, mode)
    (OUTPUT_DIR / "_done").mkdir(parents=True, exist_ok=True)

    saved = load_saved_rows(path)
    if marker.exists() and len(saved) >= len(records):
        print(f"  skip {alias}|{mode} — done marker present ({len(saved)} rows)")
        return saved

    todo = [r for r in records if r["eval_id"] not in saved]
    print(f"\n=== {alias} | {mode} === {len(saved)} done, {len(todo)} todo -> {path.name}")
    if not todo:
        marker.write_text(f"complete {len(saved)} rows\n")
        return saved

    rewrite_rows(path, saved)
    start_time = time.perf_counter()
    processed = 0

    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        sem = asyncio.Semaphore(concurrency)

        async def one(rec):
            async with sem:
                try:
                    return await ollama_chat(session, tag, build_msg(rec), max_new_tokens)
                except Exception as exc:
                    print(f"  [infer error] {alias}|{rec['eval_id']}: {str(exc)[:200]}")
                    return None

        responses = await asyncio.gather(*(one(r) for r in chunk))

        for rec, resp in zip(chunk, responses):
            raw = None
            if resp is not None:
                raw = resp.get("message", {}).get("content", "")
                if raw is not None:
                    raw = raw.strip() if isinstance(raw, str) else str(raw)
            saved[rec["eval_id"]] = make_row(cfg, mode, rec, raw)

        processed += len(chunk)
        rewrite_rows(path, saved)
        el = time.perf_counter() - start_time
        rate = processed / el if el > 0 else 0
        eta = (len(todo) - processed) / rate / 60 if rate > 0 else 0
        print(f"  ckpt ({len(saved)}/{len(records)})  {rate:.2f} row/s  eta={eta:.1f} min")

    ok = {k: v for k, v in saved.items() if v.get("raw_output") is not None}
    rewrite_rows(path, ok)
    marker.write_text(f"complete {len(ok)} rows at {time.time()}\n")
    print(f"  finished {alias}|{mode} -> {path.name} ({len(ok)} successful)")
    return ok


def ds1_make_row(cfg: dict, mode: str, rec: dict, raw) -> dict:
    """Generation outputs use the same envelope fields as medschema rows."""
    return {
        "eval_id": rec["eval_id"],
        "record_id": rec["record_id"],
        "backend": "ollama",
        "model": cfg["alias"],
        "hf_id": cfg["tag"],
        "mode": mode,
        "raw_output": raw,
    }


def ds2_make_row(cfg: dict, mode: str, rec: dict, raw) -> dict:
    """Exact field set written by run_medschema_benchmark.run_zero_shot."""
    signals, _ = score_row(rec, raw)
    return {
        "eval_id": rec["eval_id"],
        "record_id": rec["record_id"],
        "backend": "ollama",
        "model": cfg["alias"],
        "hf_id": cfg["tag"],
        "mode": mode,
        "error_difficulty": rec["error_difficulty"],
        "error_source": rec["error_source"],
        "true_has_error": rec["true_has_error"],
        "raw_output": raw,
        **signals,
    }


# ─────────────────────────────────────────────────────────────
# Per-model orchestration
# ─────────────────────────────────────────────────────────────
async def run_model(
    session, cfg, ds1_records, ds2_records, which_datasets,
    concurrency, batch_size,
) -> None:
    tag, alias = cfg["tag"], cfg["alias"]
    skip = set(cfg.get("skip_datasets") or [])
    active = [d for d in which_datasets if d not in skip]
    print(f"\n{'=' * 64}\nMODEL: {alias} ({tag})\n{'=' * 64}")
    if skip:
        print(f"  skip_datasets={sorted(skip)} -> running {active or '(nothing)'}")

    if not active:
        print(f"  nothing to run for {alias} with --datasets {which_datasets}, skipping.")
        return

    modes_needed = []
    if "ds1" in active and ds1_records is not None:
        modes_needed += ["prompt_only", "reasoning_assisted"]
    if "ds2" in active and ds2_records is not None:
        modes_needed += ["zero_shot"]

    def _complete(mode: str, n_expected: int) -> bool:
        marker = done_marker_path(alias, mode)
        path = result_path(alias, mode)
        return marker.exists() and len(load_saved_rows(path)) >= n_expected

    all_done = True
    for mode in modes_needed:
        n = len(ds2_records) if mode == "zero_shot" else len(ds1_records)
        if not _complete(mode, n):
            all_done = False
            break
    if all_done and modes_needed:
        print(f"  all scenarios already complete for {alias}, skipping model.")
        return

    await warm_up(session, tag)

    async def _safe_scenario(*sc_args, **sc_kwargs):
        """Run one scenario; on any error, log and continue (never abort the run)."""
        mode = sc_args[0] if sc_args else sc_kwargs.get("mode", "?")
        try:
            return await run_scenario(session, cfg, *sc_args, **sc_kwargs)
        except Exception as exc:
            print(f"!!! {alias}|{mode} scenario failed, skipping it: {exc}",
                  file=sys.stderr)
            return {}

    if "ds1" in active and ds1_records is not None:
        await _safe_scenario(
            "prompt_only", ds1_records,
            build_msg=ds1_build_prompt, make_row=ds1_make_row,
            max_new_tokens=MAX_NEW_TOKENS_PROMPT_ONLY,
            concurrency=concurrency, batch_size=batch_size,
        )
        await _safe_scenario(
            "reasoning_assisted", ds1_records,
            build_msg=ds1_build_prompt_cot, make_row=ds1_make_row,
            max_new_tokens=MAX_NEW_TOKENS_REASONING,
            concurrency=concurrency, batch_size=batch_size,
        )

    if "ds2" in active and ds2_records is not None:
        rows = await _safe_scenario(
            "zero_shot", ds2_records,
            build_msg=build_messages, make_row=ds2_make_row,
            max_new_tokens=MAX_NEW_TOKENS_VERIFY,
            concurrency=concurrency, batch_size=batch_size,
        )
        if rows:
            write_model_summary(alias, rows)

    await unload(session, tag)
    gc.collect()


# ─────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default=None,
        help="Single model alias from MODEL_CONFIGS (e.g. qwen3-8b).",
    )
    p.add_argument(
        "--datasets", nargs="+", default=["ds1", "ds2"],
        choices=["ds1", "ds2"],
        help="Which datasets to run (default: both).",
    )
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="Parallel requests per model (default 4).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--limit", type=int, default=None,
                   help="Only first N rows of each dataset (smoke testing).")
    p.add_argument("--n-eval", type=int, default=None,
                   help="Dataset 2: number of errored rows to eval (default: all).")
    p.add_argument("--clean-fraction", type=float, default=DS2_CLEAN_FRACTION)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--summary-only", action="store_true",
        help="Only rebuild summary_metrics.csv from existing *__zero_shot.jsonl files.",
    )
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "_done").mkdir(exist_ok=True)
    random.seed(args.seed)

    if args.summary_only:
        write_summary_csv()
        return

    token = resolve_hf_token()

    configs = MODEL_CONFIGS
    if args.model:
        configs = [c for c in MODEL_CONFIGS if c["alias"] == args.model]
        if not configs:
            valid = ", ".join(c["alias"] for c in MODEL_CONFIGS)
            raise SystemExit(f"Unknown --model '{args.model}'. Valid: {valid}")
        print(f"Single model: {args.model}")

    # Load each dataset defensively: a failure in one must NOT stop the run.
    # Whatever loads successfully still gets benchmarked overnight.
    ds1_records = None
    if "ds1" in args.datasets:
        try:
            ds1_records = ds1_load_records(token, n=args.limit)
            print(f"[ds1] {len(ds1_records)} records ready")
        except Exception as exc:
            print(f"!!! [ds1] load failed, SKIPPING dataset 1: {exc}", file=sys.stderr)
            ds1_records = None

    ds2_records = None
    if "ds2" in args.datasets:
        try:
            table = ds2_download(token)
            raw = ds2_load_frame(table)
            print(f"[ds2] loaded {len(raw)} rows from {table.name}")
            ds2_records = build_eval_set(
                raw,
                n_eval=args.n_eval,
                clean_fraction=args.clean_fraction,
                seed=args.seed,
                max_context_chars=DS2_MAX_CONTEXT_CHARS,
            )
            if args.limit:
                ds2_records = ds2_records[:args.limit]
                print(f"[ds2] limited to first {len(ds2_records)} rows")
            write_eval_manifest(ds2_records)
        except Exception as exc:
            print(f"!!! [ds2] load failed, SKIPPING dataset 2: {exc}", file=sys.stderr)
            ds2_records = None

    if ds1_records is None and ds2_records is None:
        print("!!! Both datasets failed to load — nothing to run.", file=sys.stderr)
        return

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        for cfg in configs:
            try:
                await run_model(
                    session, cfg, ds1_records, ds2_records, args.datasets,
                    args.concurrency, args.batch_size,
                )
            except Exception as exc:
                print(f"\n!!! model {cfg['alias']} failed: {exc}", file=sys.stderr)
                print("!!! continuing with next model; checkpoint preserved.",
                      file=sys.stderr)

    if "ds2" in args.datasets:
        write_summary_csv()
    print("\nAll configured models finished. Outputs in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    a = parse_args()
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoints preserved; rerun to resume.", file=sys.stderr)
        sys.exit(130)