"""Тесты диагностики качества признаков."""

import unittest

import numpy as np
import pandas as pd

from src.utils.data_quality import (
    detect_constant_features,
    detect_duplicate_features,
    detect_high_correlation_pairs,
    estimate_condition_number,
    summarize_feature_quality,
)


class TestDataQuality(unittest.TestCase):
    def setUp(self):
        n = 40
        x = np.linspace(0.0, 1.0, n)
        self.df = pd.DataFrame(
            {
                "a": x,
                "b": x,  # duplicate
                "c": 2.0 * x + 1.0,  # high correlation with a/b
                "d": np.ones(n),  # constant
                "e": np.random.RandomState(42).randn(n),
            }
        )

    def test_detect_constant_features(self):
        constants = detect_constant_features(self.df)
        self.assertIn("d", constants)
        self.assertEqual(len(constants), 1)

    def test_detect_duplicate_features(self):
        duplicates = detect_duplicate_features(self.df)
        self.assertTrue(any(pair[1] == "b" for pair in duplicates))

    def test_detect_high_correlation_pairs(self):
        pairs = detect_high_correlation_pairs(self.df[["a", "c", "e"]], threshold=0.999)
        keys = {(l, r) for l, r, _ in pairs}
        self.assertTrue(("a", "c") in keys or ("c", "a") in keys)

    def test_condition_number(self):
        cond = estimate_condition_number(self.df[["a", "c", "e"]])
        self.assertTrue(np.isfinite(cond))
        self.assertGreater(cond, 1.0)

    def test_summarize_feature_quality(self):
        s = summarize_feature_quality(self.df, corr_threshold=0.999, max_pairs=10)
        self.assertEqual(s["n_features"], 5)
        self.assertIn("d", s["constant_features"])
        self.assertGreaterEqual(len(s["duplicate_pairs"]), 1)
        self.assertGreaterEqual(len(s["high_corr_pairs"]), 1)


if __name__ == "__main__":
    unittest.main()
