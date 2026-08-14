"""Unit tests for SOB metrics v2."""

from __future__ import annotations

import unittest

from sob_eval.aggregate import (
    CELL_SCORE_WEIGHTS,
    L1_WEIGHTS,
    L2_EXTRACTION_WEIGHTS,
    L2_REPAIR_WEIGHTS,
    L3_REPAIR_WEIGHTS,
    L3_WEIGHTS,
    OVERALL_WEIGHTS,
    REPAIR_SCORE_WEIGHTS,
    aggregate,
    cell_score,
    composite_l3,
    composite_l3_repair,
    repair_score,
)
from sob_eval.metrics import (
    METRICS_VERSION,
    faithfulness,
    flatten_paths,
    score_item,
    token_f1,
    value_accuracy,
)


class FaithfulnessTests(unittest.TestCase):
    def test_token_f1_partial_credit(self):
        self.assertGreater(token_f1("American", "American country music artist"), 0.0)
        self.assertLess(token_f1("American", "American country music artist"), 1.0)

    def test_sob_appendix_f_worked_example(self):
        gold = {"name": "Hal Ashby", "nationality": "American", "is_alive": False}
        pred = {"name": "Hal Ashby", "nationality": "United States", "is_alive": False}
        g = flatten_paths(gold)
        p = flatten_paths(pred)
        va = value_accuracy(g, p)
        faith = faithfulness(g, p)
        self.assertAlmostEqual(va, 0.667, places=3)
        self.assertAlmostEqual(faith, 0.667, places=3)

    def test_score_item_worked_example(self):
        example = {
            "json_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "nationality": {"type": "string"},
                    "is_alive": {"type": "boolean"},
                },
                "required": ["name", "nationality", "is_alive"],
            },
            "validated_output": {
                "name": "Hal Ashby",
                "nationality": "American",
                "is_alive": False,
            },
        }
        pred = {
            "name": "Hal Ashby",
            "nationality": "United States",
            "is_alive": False,
        }
        scored = score_item(pred, example)
        assert scored is not None
        self.assertEqual(scored["metrics_version"], METRICS_VERSION)
        self.assertAlmostEqual(scored["value_accuracy"], 0.667, places=3)
        self.assertAlmostEqual(scored["faithfulness"], 0.667, places=3)


class AggregationWeightTests(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(L1_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(L2_EXTRACTION_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(L2_REPAIR_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(L3_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(L3_REPAIR_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(CELL_SCORE_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(REPAIR_SCORE_WEIGHTS.values()), 1.0)
        self.assertAlmostEqual(sum(OVERALL_WEIGHTS.values()), 1.0)


class JsonPassRateTests(unittest.TestCase):
    def test_json_pass_rate_is_product_not_l1_mean(self):
        rows = [
            {
                "parse_pass": True,
                "schema_pass": True,
                "json_pass_rate": True,
                "perfect_response": True,
                "prr_pass": True,
                "type_coercion": False,
                "value_accuracy": 1.0,
                "faithfulness": 1.0,
                "path_recall": 1.0,
                "structure_coverage": 1.0,
                "type_safety": 1.0,
                "field_accuracy": 1.0,
                "missing_rate": 0.0,
                "nested_field_accuracy": 1.0,
                "value_accuracy_hardened": 1.0,
                "faithfulness_hardened": 1.0,
                "path_recall_hardened": 1.0,
                "structure_coverage_hardened": 1.0,
                "coverage_gate": 1.0,
                "acc_by_depth": {"1": 1.0},
            },
            {
                "parse_pass": True,
                "schema_pass": False,
                "json_pass_rate": False,
                "perfect_response": False,
                "prr_pass": False,
                "type_coercion": False,
                "value_accuracy": 0.0,
                "faithfulness": 0.0,
                "path_recall": 0.0,
                "structure_coverage": 0.0,
                "type_safety": 0.0,
                "field_accuracy": 0.0,
                "missing_rate": 1.0,
                "nested_field_accuracy": 0.0,
                "value_accuracy_hardened": 0.0,
                "faithfulness_hardened": 0.0,
                "path_recall_hardened": 0.0,
                "structure_coverage_hardened": 0.0,
                "coverage_gate": 0.0,
                "acc_by_depth": {},
            },
        ]
        metrics = aggregate(rows, n_bootstrap=0)
        jpass = metrics["json_pass_rate"]["estimate"]
        parse = metrics["parse_pass"]["estimate"]
        schema = metrics["schema_pass"]["estimate"]
        self.assertAlmostEqual(jpass, parse * schema)
        l1 = cell_score(metrics)["L1_compliance"]
        self.assertNotAlmostEqual(l1, jpass)


class L3GatingTests(unittest.TestCase):
    def test_l3_gated_by_l2(self):
        """Poor accuracy cannot produce high L3 even with flat depth retention."""
        metrics = {
            "nesting_degradation": {"estimate": 0.0083, "retention": 0.8383},
        }
        l2 = 0.0834
        self.assertAlmostEqual(composite_l3(metrics, l2=l2), 0.0699, places=3)

    def test_l3_never_exceeds_l2_when_retention_above_one(self):
        metrics = {
            "nesting_degradation": {"estimate": -0.002, "retention": 1.0025},
        }
        l2 = 0.7762
        self.assertAlmostEqual(composite_l3(metrics, l2=l2), l2, places=4)
        self.assertLessEqual(composite_l3(metrics, l2=l2), l2)


class L3RepairGatingTests(unittest.TestCase):
    def test_repair_l3_penalizes_over_correction(self):
        metrics = {"nesting_degradation": {"estimate": 0.0, "retention": 1.0}}
        l2 = 0.7756
        ocr = 0.374
        self.assertAlmostEqual(
            composite_l3_repair(metrics, l2=l2, over_correction_rate=ocr),
            0.4855,
            places=3,
        )
        self.assertAlmostEqual(
            composite_l3_repair(metrics, l2=l2, over_correction_rate=0.0),
            0.7756,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
