"""Load gold references from Hugging Face for extraction and repair scoring."""

from __future__ import annotations

from scripts.paths import EXTRACTION_HF


def load_extraction_gold() -> dict[str, dict]:
    """Return record_id → {json_schema, validated_output} from interfaze-ai/sob test split."""
    from datasets import load_dataset

    dataset_id, config, split = EXTRACTION_HF
    print(f"Loading extraction gold from {dataset_id} / {config} / {split} …", flush=True)
    kwargs = {"split": split}
    if config:
        ds = load_dataset(dataset_id, config, **kwargs)
    else:
        ds = load_dataset(dataset_id, **kwargs)

    gold: dict[str, dict] = {}
    for example in ds:
        record_id = str(example["record_id"])
        gold[record_id] = {
            "record_id": record_id,
            "json_schema": example["json_schema"],
            "validated_output": example.get("validated_output") or example.get("ground_truth"),
        }
    print(f"  {len(gold):,} gold records", flush=True)
    return gold
