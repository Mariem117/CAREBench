"""SOB evaluation framework: metrics, test sets, weighting, CLI."""

from .metrics import METRICS_VERSION, extract_json, score_item, nesting_stats
from .aggregate import (
    aggregate,
    cell_score,
    composite_l1_l2,
    overall_benchmark_score,
    repair_score,
)
from .weighting import WEIGHTING_SCHEMES, overall_sob_score

__all__ = [
    "METRICS_VERSION",
    "extract_json",
    "score_item",
    "nesting_stats",
    "aggregate",
    "composite_l1_l2",
    "cell_score",
    "repair_score",
    "overall_benchmark_score",
    "WEIGHTING_SCHEMES",
    "overall_sob_score",
]
