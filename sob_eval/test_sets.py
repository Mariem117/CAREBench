"""
Canonical SOB evaluation test sets.

Primary slices
--------------
1. schema_medium          — sob_by_schema_complexity / data_medium / test   (n≈1924)
2. question_hard          — sob_by_question_difficulty / data_hard / test   (n≈856)
3. medium_q_hard_s        — Medium Question × Hard Schema (built from schema hard + filter)
4. joint_grid cells       — optional full 3×2 grid via record_id merge

Also exposes helpers to slice any loaded HF split by (question_difficulty, schema_complexity).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Expected joint counts from Benchmark/merging-data.py (full repos, not test-only)
EXPECTED_JOINT_FULL = {
    ("easy", "hard"): 640,
    ("easy", "medium"): 321,
    ("medium", "hard"): 1836,
    ("medium", "medium"): 1270,
    ("hard", "hard"): 533,
    ("hard", "medium"): 333,
}


@dataclass(frozen=True)
class TestSetSpec:
    name: str
    dataset_id: str
    config: str
    split: str
    question_difficulty: Optional[str] = None  # filter if set
    schema_complexity: Optional[str] = None    # filter if set
    description: str = ""


# --- Official evaluation suites ------------------------------------------------

SCHEMA_MEDIUM = TestSetSpec(
    name="schema_medium",
    dataset_id="mariem123kfg/sob_by_schema_complexity",
    config="data_medium",
    split="test",
    schema_complexity="medium",
    description="Schema-complexity medium test (mixed question difficulties). Primary medium bench.",
)

SCHEMA_HARD = TestSetSpec(
    name="schema_hard",
    dataset_id="mariem123kfg/sob_by_schema_complexity",
    config="data_hard",
    split="test",
    schema_complexity="hard",
    description="Schema-complexity hard test (mixed question difficulties).",
)

QUESTION_HARD = TestSetSpec(
    name="question_hard",
    dataset_id="mariem123kfg/sob_by_question_difficulty",
    config="data_hard",
    split="test",
    question_difficulty="hard",
    description="Question-difficulty hard test (mixed schema complexities).",
)

MEDIUM_Q_HARD_S = TestSetSpec(
    name="medium_q_hard_s",
    dataset_id="mariem123kfg/sob_by_schema_complexity",
    config="data_hard",
    split="test",
    question_difficulty="medium",
    schema_complexity="hard",
    description=(
        "CRITICAL CELL: Medium Question × Hard Schema. "
        "Isolates schema stress under moderate reasoning load. "
        "Built by filtering schema data_hard/test where question_difficulty==medium."
    ),
)

HARD_Q_HARD_S = TestSetSpec(
    name="hard_q_hard_s",
    dataset_id="mariem123kfg/sob_by_schema_complexity",
    config="data_hard",
    split="test",
    question_difficulty="hard",
    schema_complexity="hard",
    description="Hard Question × Hard Schema — maximum stress cell.",
)

HARD_Q_MEDIUM_S = TestSetSpec(
    name="hard_q_medium_s",
    dataset_id="mariem123kfg/sob_by_schema_complexity",
    config="data_medium",
    split="test",
    question_difficulty="hard",
    schema_complexity="medium",
    description="Hard Question × Medium Schema — reasoning stress with milder schemas.",
)

MEDIUM_Q_MEDIUM_S = TestSetSpec(
    name="medium_q_medium_s",
    dataset_id="mariem123kfg/sob_by_schema_complexity",
    config="data_medium",
    split="test",
    question_difficulty="medium",
    schema_complexity="medium",
    description="Medium × Medium — baseline cell for Robustness Ratio denominators.",
)

TEST_SETS: dict[str, TestSetSpec] = {
    s.name: s
    for s in (
        SCHEMA_MEDIUM,
        SCHEMA_HARD,
        QUESTION_HARD,
        MEDIUM_Q_HARD_S,
        HARD_Q_HARD_S,
        HARD_Q_MEDIUM_S,
        MEDIUM_Q_MEDIUM_S,
    )
}

# Minimal suite for a paper-ready table
CORE_SUITE = [
    "medium_q_medium_s",
    "medium_q_hard_s",   # primary new cell
    "hard_q_medium_s",
    "hard_q_hard_s",
    "schema_medium",     # legacy comparable to prior runs (n=1924)
    "question_hard",     # legacy comparable to hard_* benches (n=856)
]


def load_testset(
    spec: TestSetSpec | str,
    hf_token: Optional[str] = None,
    limit: Optional[int] = None,
):
    """Load a TestSetSpec from Hugging Face, applying axis filters."""
    from datasets import load_dataset

    if isinstance(spec, str):
        spec = TEST_SETS[spec]

    kwargs = {}
    token = hf_token or os.environ.get("HFHUB_TOKEN_1") or os.environ.get("HF_TOKEN")
    if token and str(token).startswith("hf_"):
        kwargs["token"] = token

    print(
        f"[testset] {spec.name}: {spec.dataset_id} / {spec.config} / {spec.split}",
        flush=True,
    )
    ds = load_dataset(spec.dataset_id, spec.config, split=spec.split, **kwargs)

    def _ok(ex):
        if spec.question_difficulty and ex.get("question_difficulty") != spec.question_difficulty:
            return False
        if spec.schema_complexity and ex.get("schema_complexity") != spec.schema_complexity:
            return False
        return True

    if spec.question_difficulty or spec.schema_complexity:
        before = len(ds)
        ds = ds.filter(_ok)
        print(
            f"[testset] filter q={spec.question_difficulty} s={spec.schema_complexity}: "
            f"{before} → {len(ds)}",
            flush=True,
        )

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[testset] {spec.name}: {len(ds)} rows — {spec.description}", flush=True)
    return ds


def cell_key_from_example(ex: dict) -> tuple[str, str]:
    return (
        str(ex.get("question_difficulty", "unknown")),
        str(ex.get("schema_complexity", "unknown")),
    )
