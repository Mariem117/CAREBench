# CAREBench Methodology

CAREBench (**C**ompliance · **A**ccuracy · **R**obustness **Bench**mark) evaluates small language models on schema-oriented JSON tasks using a three-layer metric stack aligned with the Structured Output Benchmark (SOB) literature, extended for repair and gated robustness scoring.

**Metrics version:** 2.3 (`sob_eval.metrics.METRICS_VERSION`)

---

## 1. Benchmark tasks

| Task | Mode suffix | Input | Output | Rows |
|------|-------------|-------|--------|------|
| **JSON Extraction** | `prompt_only` | Context + question + schema | Structured JSON | 4,951 |
| **Reasoning-assisted extraction** | `reasoning_assisted` | Same | JSON (may follow reasoning) | 4,951 |
| **JSON Repair & error detection** | `zero_shot` | Flawed JSON + schema | `{has_error, corrected_json}` | 6,614–6,615 |

Each task is scored **separately**. Reasoning-assisted outputs are never mixed with prompt-only outputs before aggregation.

---

## 2. Evaluation pipeline

```
data/predictions/{model}__{mode}.jsonl
        │
        ▼
  sob_eval.metrics.score_item()     ← per-row atomic metrics
        │
        ▼
  sob_eval.aggregate.aggregate()    ← micro-means + Wilson/bootstrap CIs
        │
        ▼
  composite L1 / L2 / L3            ← layer formulas (below)
        │
        ▼
  Cell score / Repair score         ← weighted blend
        │
        ▼
  results/extraction_summary.csv
  results/repair_summary.csv
```

**Gold references:** `data/gold/extraction_test.parquet` (`json_schema`, `validated_output`).

**Repair content metrics** are computed on `corrected_json` extracted from the model wrapper, not on the error-detection flags.

---

## 3. Layer 1 — Compliance

Can the model produce usable structured output?

| Metric | Definition |
|--------|------------|
| **Parse Pass** | Valid JSON after extraction (strips fences, chain-of-thought prefixes) |
| **Schema Pass** | `jsonschema.validate(output, schema)` succeeds |
| **Structure Coverage (hardened)** | F1 over leaf path sets (precision × recall on dot-paths) |

**L1 (extraction & repair):**

\[
\textbf{L1} = 0.40 \cdot \text{ParsePass} + 0.35 \cdot \text{SchemaPass} + 0.25 \cdot \text{StructureCoverage}_{\text{hardened}}
\]

**JSON Pass Rate** (reported separately) = Parse Pass × Schema Pass.

---

## 4. Layer 2 — Accuracy

Are field values and structure correct?

### Extraction / reasoning L2

| Component | Weight |
|-----------|--------|
| Value Accuracy (hardened) | 0.35 |
| Faithfulness (hardened) | 0.15 |
| Path Recall (hardened) | 0.20 |
| Type Safety | 0.10 |
| Perfect Response | 0.20 |

**Value Accuracy** = exact leaf-value match rate over gold dot-paths (SOB primary metric).  
**Faithfulness** = token-level F1 soft match (partial credit).  
**Path Recall** = \|gold ∩ pred paths\| / \|gold paths\|.  
**Perfect Response** = full leaf-level exact match on all gold paths.

### Repair L2

Same content metrics with repair-specific additions:

| Component | Weight |
|-----------|--------|
| Value Accuracy (hardened) | 0.20 |
| Faithfulness (hardened) | 0.10 |
| Path Recall (hardened) | 0.15 |
| Type Safety | 0.10 |
| Perfect Response | 0.10 |
| Detection F1 | 0.15 |
| Localization Accuracy | 0.10 |
| Repair Exact Match | 0.10 |

**Detection Accuracy** is diagnostic-only (not in L2).

---

## 5. Layer 3 — Robustness

Does accuracy hold for nested fields?

| Metric | Definition |
|--------|------------|
| **Nesting Retention** | `deep_acc / shallow_acc` (depth ≥ 2 vs depth 1) |
| **Nesting Degradation** | `shallow_acc − deep_acc` (diagnostic) |

### Extraction / reasoning L3 (v2.3)

\[
\textbf{L3} = \text{clamp}\big(\min(\text{Retention}, 1) \times \text{L2},\ 0,\ 1\big)
\]

**Design rationale:** Robustness is depth stability **gated by accuracy**. A model that is wrong everywhere cannot score high L3. Retention is capped at 1.0 so deep fields cannot inflate L3 above L2.

### Repair L3 (v2.3)

\[
\textbf{L3}_{\text{repair}} = \text{clamp}\big(\min(\text{Retention}, 1) \times \text{L2} \times (1 - \text{OverCorrectionRate}),\ 0,\ 1\big)
\]

**Over-correction rate** = on clean (error-free) records, fraction where the model changed values unnecessarily.

---

## 6. Overall scores

| Task | Formula |
|------|---------|
| **Cell score** (extraction / reasoning) | `0.30·L1 + 0.50·L2 + 0.20·L3` |
| **Repair score** | `0.20·L1 + 0.55·L2 + 0.25·L3` |

Weights prioritize **accuracy (L2)** over compliance and robustness for deployment-oriented ranking.

---

## 7. Aggregation

- **Micro-average** over all scored rows (unweighted by schema difficulty).
- Optional **schema-weighted** analysis: `scripts/compute_schema_weighted_metrics.py` (easy=1, medium=2, hard=3) — see `results/*_schema_weighted.csv`.
- Bootstrap 95% CIs available via `n_bootstrap > 0` in `aggregate()` (default 2,000 in CLI evaluate).

---

## 8. References

- SOB paper metrics: [arXiv:2604.25359](https://arxiv.org/abs/2604.25359)
- Nesting depth framing: DeepJSONEval (Zhou et al., 2025)
- Over-correction framing: GLEU / grammatical error correction literature

See also: [METRICS_REPORT.md](METRICS_REPORT.md) for empirical results on the DGX benchmark run.
