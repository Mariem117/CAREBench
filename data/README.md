# Data (not shipped in this repo)

Gold datasets are downloaded from Hugging Face at scoring time. Model predictions are **your local JSONL outputs** — place them under `predictions/` using the naming convention below.

## Extraction gold

| Field | Value |
|-------|--------|
| **Dataset** | [`interfaze-ai/sob`](https://huggingface.co/datasets/interfaze-ai/sob) |
| **Config** | `default` |
| **Split** | `test` |
| **Rows** | ~4,951 |

Used for JSON extraction (`prompt_only`) and reasoning-assisted extraction (`reasoning_assisted`). Each example provides `record_id`, `json_schema`, and `validated_output` (or `ground_truth`).

```python
from datasets import load_dataset
ds = load_dataset("interfaze-ai/sob", "default", split="test")
```

## Repair / error-detection gold

| Field | Value |
|-------|--------|
| **Dataset** | [`MoetezF/SOB-with-errors-injection`](https://huggingface.co/datasets/MoetezF/SOB-with-errors-injection) |
| **Primary file** | `sob_text_test_with_errors.parquet` (or `.jsonl`) |
| **Rows** | ~6,667 eval cells (clean + injected errors) |

Used for the `zero_shot` repair task. Repair JSONL outputs should include per-row labels such as `true_has_error`, `pred_has_error`, `repair_exact`, and `over_correction` when available.

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="MoetezF/SOB-with-errors-injection",
    filename="sob_text_test_with_errors.parquet",
)
```

Extraction gold (`interfaze-ai/sob`) is also used to score the **content** of `corrected_json` against `validated_output`.

## Model predictions (`predictions/`)

Not included in git. Run `python test_models.py` to generate JSONL here, or add your own files:

| Suffix | Task |
|--------|------|
| `__prompt_only.jsonl` | JSON extraction |
| `__reasoning_assisted.jsonl` | Reasoning-assisted extraction |
| `__zero_shot.jsonl` | JSON repair & error detection |

Example: `predictions/qwen3-8b__prompt_only.jsonl`

Each line must include at least `record_id` and `raw_output`.

## Models scored in CAREBench

`deepseek-r1-1.5b`, `gemma3-4b`, `granite-4.1-8b`, `llama-3.2-3b`, `llama3.1-8b`, `ministral-3-3b`, `nuextract-3-4b`, `phi-4-mini`, `qwen3-8b`, `qwen3.5-9b`
