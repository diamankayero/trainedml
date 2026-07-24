"""
Tests unitaires de trainedml.tasks : détection de tâche et de déséquilibre
de classes.
"""
import unittest
import warnings

import pandas as pd

from trainedml import Trainer, compare
from trainedml.tasks import check_class_imbalance, detect_task, warn_if_imbalanced


class TestDetectTask(unittest.TestCase):
    def test_text_target_is_classification(self):
        self.assertEqual(detect_task(pd.Series(["a", "b", "a"])), "classification")

    def test_continuous_target_is_regression(self):
        self.assertEqual(detect_task(pd.Series([0.1, 2.7, 3.14, 5.9])), "regression")

    def test_few_unique_integers_is_classification(self):
        self.assertEqual(detect_task(pd.Series([0, 1, 0, 1, 1])), "classification")


class TestCheckClassImbalance(unittest.TestCase):
    def test_balanced_returns_none(self):
        y = pd.Series(["a"] * 55 + ["b"] * 45)
        self.assertIsNone(check_class_imbalance(y))

    def test_imbalanced_returns_details(self):
        y = pd.Series(["a"] * 90 + ["b"] * 10)
        info = check_class_imbalance(y)
        self.assertIsNotNone(info)
        self.assertEqual(info["ratio"], 9.0)
        self.assertEqual(info["majority_class"], "a")
        self.assertEqual(info["majority_count"], 90)
        self.assertEqual(info["minority_class"], "b")
        self.assertEqual(info["minority_count"], 10)

    def test_custom_threshold(self):
        y = pd.Series(["a"] * 60 + ["b"] * 40)  # ratio 1.5
        self.assertIsNone(check_class_imbalance(y, threshold=3.0))
        self.assertIsNotNone(check_class_imbalance(y, threshold=1.4))

    def test_single_class_returns_none(self):
        self.assertIsNone(check_class_imbalance(pd.Series(["a"] * 10)))


class TestWarnIfImbalanced(unittest.TestCase):
    def test_warns_on_imbalance(self):
        y = pd.Series(["a"] * 90 + ["b"] * 10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_imbalanced(y)
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertIn("class_weight", str(caught[0].message))

    def test_no_warning_when_balanced(self):
        y = pd.Series(["a"] * 55 + ["b"] * 45)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_imbalanced(y)
        self.assertEqual(len(caught), 0)


class TestImbalanceIntegration(unittest.TestCase):
    def _imbalanced_xy(self):
        import numpy as np
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
        y = pd.Series(["rare"] * 15 + ["frequent"] * 185)
        return X, y

    def test_trainer_fit_warns_on_imbalanced_data(self):
        X, y = self._imbalanced_xy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Trainer(X=X, y=y, model="random_forest", seed=42).fit()
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))

    def test_trainer_fit_no_warning_on_balanced_data(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Trainer(dataset="wine", model="knn", seed=42).fit()
        self.assertEqual(len(caught), 0)

    def test_compare_warns_on_imbalanced_data(self):
        X, y = self._imbalanced_xy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compare(X=X, y=y, cv=3, seed=42, show_progress=False)
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))


if __name__ == "__main__":
    unittest.main()
