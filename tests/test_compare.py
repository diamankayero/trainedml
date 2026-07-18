"""
Tests unitaires de trainedml.compare et du benchmark par validation croisée.
"""
import unittest

import numpy as np
import pandas as pd

from trainedml import compare
from trainedml.benchmark import Benchmark
from trainedml.data.loader import DataLoader
from trainedml.models import CLASSIFIER_MAP, KNNModel, RandomForestModel


class TestCompare(unittest.TestCase):
    def test_compare_classification(self):
        df = compare(dataset="iris", cv=3, show_progress=False)
        self.assertEqual(len(df), len(CLASSIFIER_MAP))
        self.assertIn("accuracy", df.columns)
        self.assertIn("accuracy_std", df.columns)
        self.assertIn("fit_time", df.columns)
        # Trié par accuracy décroissante
        self.assertTrue(df["accuracy"].is_monotonic_decreasing)

    def test_compare_regression_in_memory(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=100)})
        y = pd.Series(2 * X["a"] + rng.normal(0, 0.1, 100))
        df = compare(X=X, y=y, cv=3, show_progress=False)
        self.assertIn("r2", df.columns)
        self.assertNotIn("accuracy", df.columns)
        self.assertTrue(df["r2"].is_monotonic_decreasing)

    def test_compare_custom_models(self):
        df = compare(
            dataset="iris",
            models={"knn": KNNModel(n_neighbors=3), "rf": RandomForestModel(n_estimators=10)},
            cv=3,
            show_progress=False,
        )
        self.assertEqual(set(df.index), {"knn", "rf"})

    def test_compare_requires_data(self):
        with self.assertRaises(ValueError):
            compare(show_progress=False)


class TestBenchmarkCV(unittest.TestCase):
    def setUp(self):
        loader = DataLoader()
        self.X, self.y = loader.load_dataset(name="iris")

    def test_run_cv_and_to_dataframe(self):
        bench = Benchmark({"knn": KNNModel(n_neighbors=3)})
        results = bench.run_cv(self.X, self.y, cv=3, show_progress=False)
        self.assertIn("knn", results)
        self.assertEqual(results["knn"]["cv"], 3)
        self.assertIn("scores_std", results["knn"])

        df = bench.to_dataframe()
        self.assertEqual(list(df.index), ["knn"])
        self.assertIn("accuracy", df.columns)
        self.assertIn("accuracy_std", df.columns)

    def test_to_dataframe_without_run(self):
        bench = Benchmark({"knn": KNNModel()})
        self.assertIsNone(bench.to_dataframe())


if __name__ == "__main__":
    unittest.main()
