"""Repository path constants for CAREBench scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
PREDICTIONS_DIR = DATA_DIR / "predictions"
RESULTS_DIR = REPO_ROOT / "results"

EXTRACTION_SUMMARY = RESULTS_DIR / "extraction_summary.csv"
REPAIR_SUMMARY = RESULTS_DIR / "repair_summary.csv"

EXTRACT_MODES = ("prompt_only", "reasoning_assisted")
REPAIR_MODE = "zero_shot"

# Hugging Face dataset references (gold is loaded at runtime — not shipped in repo)
EXTRACTION_HF = ("interfaze-ai/sob", "default", "test")
REPAIR_HF = ("MoetezF/SOB-with-errors-injection", None, "train")
