"""
SOB weighting schemes across the Question × Schema evaluation grid.

Axes
----
  question_difficulty ∈ {easy, medium, hard}
  schema_complexity   ∈ {medium, hard}

Primary stress cell for this project: **(medium, hard)** — Medium Question + Hard Schema.
"""

from __future__ import annotations

from typing import Iterable

# Grid cell keys as (question_difficulty, schema_complexity)
Cell = tuple[str, str]

# ---------------------------------------------------------------------------
# Scheme A — Uniform (baseline)
# ---------------------------------------------------------------------------
UNIFORM: dict[Cell, float] = {
    ("easy", "medium"): 1 / 6,
    ("easy", "hard"): 1 / 6,
    ("medium", "medium"): 1 / 6,
    ("medium", "hard"): 1 / 6,
    ("hard", "medium"): 1 / 6,
    ("hard", "hard"): 1 / 6,
}

# ---------------------------------------------------------------------------
# Scheme B — Difficulty-tilted (recommended default)
# Weight ∝ (1 + α·q_rank) · (1 + β·s_rank), renormalized.
# q_rank: easy=0, medium=1, hard=2 ; s_rank: medium=0, hard=1
# α=0.75 (question), β=1.00 (schema) → schema hardness slightly heavier,
# and Medium×Hard gets substantial mass without wiping Easy×Medium.
# ---------------------------------------------------------------------------
def _difficulty_tilted(alpha: float = 0.75, beta: float = 1.0) -> dict[Cell, float]:
    q_rank = {"easy": 0, "medium": 1, "hard": 2}
    s_rank = {"medium": 0, "hard": 1}
    raw: dict[Cell, float] = {}
    for q, qr in q_rank.items():
        for s, sr in s_rank.items():
            raw[(q, s)] = (1.0 + alpha * qr) * (1.0 + beta * sr)
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


DIFFICULTY_TILTED = _difficulty_tilted()

# ---------------------------------------------------------------------------
# Scheme C — Stress-focused
# Puts ~35% on Medium×Hard (critical combo for "reasoning OK + schema hard"),
# ~25% on Hard×Hard, rest spread. Use when reporting "can models survive
# hard schemas without hard questions?" as a first-class claim.
# ---------------------------------------------------------------------------
STRESS_FOCUSED: dict[Cell, float] = {
    ("easy", "medium"): 0.05,
    ("easy", "hard"): 0.10,
    ("medium", "medium"): 0.10,
    ("medium", "hard"): 0.35,  # Medium Q + Hard Schema
    ("hard", "medium"): 0.15,
    ("hard", "hard"): 0.25,
}

# ---------------------------------------------------------------------------
# Scheme D — Separable axes (for ablations)
# Final = λ · Score_Q + (1-λ) · Score_S
# where Score_Q averages over schema within a difficulty, etc.
# Implemented in overall_sob_score_separable().
# ---------------------------------------------------------------------------
DEFAULT_LAMBDA_QUESTION = 0.45  # schema complexity gets 0.55

WEIGHTING_SCHEMES: dict[str, dict[Cell, float]] = {
    "uniform": UNIFORM,
    "difficulty_tilted": DIFFICULTY_TILTED,
    "stress_focused": STRESS_FOCUSED,
}


def normalize_weights(weights: dict[Cell, float], present: Iterable[Cell]) -> dict[Cell, float]:
    """Renormalize over cells that actually have scores (missing cells dropped)."""
    present = list(present)
    sub = {c: weights[c] for c in present if c in weights}
    z = sum(sub.values()) or 1.0
    return {c: w / z for c, w in sub.items()}


def overall_sob_score(
    cell_scores: dict[Cell, float],
    scheme: str = "difficulty_tilted",
) -> dict:
    """
    Weighted mean of per-cell scores.

    cell_scores: {(q, s): cell_score float in [0,1]}
    """
    if scheme not in WEIGHTING_SCHEMES:
        raise KeyError(f"Unknown scheme {scheme!r}; choose from {list(WEIGHTING_SCHEMES)}")
    base = WEIGHTING_SCHEMES[scheme]
    w = normalize_weights(base, cell_scores.keys())
    total = sum(w[c] * cell_scores[c] for c in w)
    return {
        "scheme": scheme,
        "overall": round(total, 4),
        "weights_used": {f"{q}|{s}": round(w[(q, s)], 4) for q, s in w},
        "cells": {f"{q}|{s}": round(cell_scores[(q, s)], 4) for q, s in cell_scores},
    }


def overall_sob_score_separable(
    cell_scores: dict[Cell, float],
    lambda_question: float = DEFAULT_LAMBDA_QUESTION,
) -> dict:
    """
    Separable axis blend:

      Score_Q(d) = mean_s cell(d, s)
      Score_S(c) = mean_d cell(d, c)
      Overall = λ · mean_d Score_Q(d)  +  (1-λ) · mean_c Score_S(c)

    With harder difficulties optionally importance-weighted inside each axis
    via DIFFICULTY_TILTED marginals.
    """
    from collections import defaultdict

    by_q: dict[str, list[float]] = defaultdict(list)
    by_s: dict[str, list[float]] = defaultdict(list)
    for (q, s), v in cell_scores.items():
        by_q[q].append(v)
        by_s[s].append(v)

    score_q = {q: sum(vs) / len(vs) for q, vs in by_q.items()}
    score_s = {s: sum(vs) / len(vs) for s, vs in by_s.items()}

    # marginal weights from difficulty_tilted
    q_w = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
    s_w = {"medium": 0.0, "hard": 0.0}
    for (q, s), w in DIFFICULTY_TILTED.items():
        q_w[q] += w
        s_w[s] += w
    # renormalize to available
    q_w = normalize_weights({(k, "medium"): v for k, v in q_w.items() if k in score_q},
                            [(k, "medium") for k in score_q])
    # simpler:
    q_avail = {k: DIFFICULTY_TILTED.get((k, "medium"), 0) + DIFFICULTY_TILTED.get((k, "hard"), 0)
               for k in score_q}
    s_avail = {k: sum(DIFFICULTY_TILTED.get((q, k), 0) for q in ("easy", "medium", "hard"))
               for k in score_s}
    zq, zs = sum(q_avail.values()) or 1, sum(s_avail.values()) or 1
    q_avail = {k: v / zq for k, v in q_avail.items()}
    s_avail = {k: v / zs for k, v in s_avail.items()}

    q_part = sum(q_avail[q] * score_q[q] for q in score_q)
    s_part = sum(s_avail[s] * score_s[s] for s in score_s)
    overall = lambda_question * q_part + (1 - lambda_question) * s_part
    return {
        "scheme": "separable_axes",
        "lambda_question": lambda_question,
        "score_question_axis": round(q_part, 4),
        "score_schema_axis": round(s_part, 4),
        "overall": round(overall, 4),
        "by_question": {k: round(v, 4) for k, v in score_q.items()},
        "by_schema": {k: round(v, 4) for k, v in score_s.items()},
    }


def justify_weights() -> str:
    return """
Weighting justification
-----------------------
1. Uniform — diagnostic only; under-weights rare hard cells if n-weighted elsewhere.

2. Difficulty-tilted (DEFAULT) — multiplicative ranks:
     w(q,s) ∝ (1 + 0.75·q_rank)·(1 + 1.0·s_rank)
   Schema hardness is slightly heavier than question hardness because SOB is
   schema-oriented; Medium×Hard receives ~22% on a full 6-cell grid.

3. Stress-focused — explicitly allocates 35% to Medium Question × Hard Schema,
   the cell that isolates schema stress without conflating hard multi-hop
   questions. Use when that cell is the paper's primary claim.

4. Separable axes — reports question-axis and schema-axis scores separately,
   then blends with λ≈0.45 on questions / 0.55 on schema. Prefer this for
   ablations and radar plots.

Medium×Hard special handling
----------------------------
Evaluate as its own test set (see test_sets.MEDIUM_Q_HARD_S). Report:
  - cell_score(Medium×Hard)
  - Robustness Ratio vs Medium×Medium: FA(M×H)/FA(M×M)
  - Nesting Degradation within Medium×Hard (schemas are deepest here)
Never bury Medium×Hard only inside a micro-averaged full-test score.
""".strip()
