"""
Tests unitaires du Trainer : régression, hyperparamètres, estimateurs sklearn,
prétraitement, persistance et données en mémoire.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from trainedml import Trainer


def _regression_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(0, 10, n)})
    y = pd.Series(3 * X["a"] + 0.5 * X["b"] + rng.normal(0, 0.1, n), name="target")
    return X, y


class TestTrainerClassification(unittest.TestCase):
    def test_fit_evaluate_iris(self):
        trainer = Trainer(dataset="iris", model="knn")
        trainer.fit()
        scores = trainer.evaluate()
        self.assertIn("accuracy", scores)
        self.assertGreater(scores["accuracy"], 0.8)

    def test_model_params(self):
        trainer = Trainer(dataset="iris", model="knn", model_params={"n_neighbors": 1})
        self.assertEqual(trainer.model.model.n_neighbors, 1)

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError):
            Trainer(dataset="iris", model="inexistant")

    def test_predict_before_fit_raises(self):
        trainer = Trainer(dataset="iris", model="knn")
        with self.assertRaises(RuntimeError):
            trainer.predict([[5.1, 3.5, 1.4, 0.2]])

    def test_reseed_invalidates_fit(self):
        trainer = Trainer(dataset="iris", model="knn")
        trainer.fit()
        trainer.load_data(seed=7)
        self.assertFalse(trainer.is_fitted)


class TestTrainerRegression(unittest.TestCase):
    def test_regression_metrics(self):
        """Le Trainer doit retourner des métriques de régression pour un régresseur."""
        X, y = _regression_data()
        trainer = Trainer(X=X, y=y, model="ridge")
        trainer.fit()
        scores = trainer.evaluate()
        self.assertIn("r2", scores)
        self.assertIn("rmse", scores)
        self.assertNotIn("accuracy", scores)
        self.assertGreater(scores["r2"], 0.9)

    def test_task_detection(self):
        X, y = _regression_data()
        trainer = Trainer(X=X, y=y, model="linear")
        trainer.fit()
        self.assertEqual(trainer.task, "regression")


class TestTrainerSklearnEstimator(unittest.TestCase):
    def test_arbitrary_sklearn_classifier(self):
        from sklearn.svm import SVC
        trainer = Trainer(dataset="iris", model=SVC())
        trainer.fit()
        scores = trainer.evaluate()
        self.assertIn("accuracy", scores)
        self.assertGreater(scores["accuracy"], 0.8)

    def test_rejects_non_estimator(self):
        with self.assertRaises(TypeError):
            Trainer(dataset="iris", model=42)

    def test_model_params_rejected_with_instance(self):
        from sklearn.svm import SVC
        with self.assertRaises(ValueError):
            Trainer(dataset="iris", model=SVC(), model_params={"C": 2})


class TestTrainerPreprocessing(unittest.TestCase):
    def test_handles_nan_and_categorical(self):
        """Le prétraitement doit gérer NaN et colonnes catégorielles."""
        rng = np.random.default_rng(1)
        n = 100
        X = pd.DataFrame({
            "num": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
        })
        X.loc[::10, "num"] = np.nan
        y = pd.Series((X["cat"] == "a").astype(int).where(lambda s: s == 1, 0), name="y")
        trainer = Trainer(X=X, y=y, model="logistic")
        trainer.fit()
        scores = trainer.evaluate()
        self.assertIn("accuracy", scores)

    def test_preprocess_disabled(self):
        trainer = Trainer(dataset="iris", model="random_forest", preprocess=False)
        self.assertIsNone(trainer.preprocessor)
        trainer.fit()
        self.assertGreater(trainer.evaluate()["accuracy"], 0.8)


class TestTrainerPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        trainer = Trainer(dataset="iris", model="knn")
        trainer.fit()
        expected = trainer.predict([[5.1, 3.5, 1.4, 0.2]])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.joblib")
            trainer.save(path)
            restored = Trainer.load(path)
            self.assertTrue(restored.is_fitted)
            self.assertEqual(restored.model_name, "knn")
            self.assertEqual(restored.task, "classification")
            np.testing.assert_array_equal(restored.predict([[5.1, 3.5, 1.4, 0.2]]), expected)

    def test_save_before_fit_raises(self):
        trainer = Trainer(dataset="iris", model="knn")
        with self.assertRaises(RuntimeError):
            trainer.save("nope.joblib")


if __name__ == "__main__":
    unittest.main()
