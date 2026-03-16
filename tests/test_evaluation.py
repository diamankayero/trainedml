"""
Tests unitaires pour les métriques de régression et auto_evaluate dans Evaluator.
"""
import unittest
import numpy as np
from trainedml.evaluation import Evaluator


class TestEvaluatorClassification(unittest.TestCase):
    def test_evaluate_all_basic(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        scores = Evaluator.evaluate_all(y_true, y_pred)
        self.assertIn('accuracy', scores)
        self.assertIn('precision', scores)
        self.assertIn('recall', scores)
        self.assertIn('f1', scores)
        self.assertAlmostEqual(scores['accuracy'], 0.8)

    def test_evaluate_all_perfect(self):
        y = [0, 1, 2, 0, 1]
        scores = Evaluator.evaluate_all(y, y)
        self.assertAlmostEqual(scores['accuracy'], 1.0)
        self.assertAlmostEqual(scores['f1'], 1.0)


class TestEvaluatorRegression(unittest.TestCase):
    def test_evaluate_regression_basic(self):
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]
        scores = Evaluator.evaluate_regression(y_true, y_pred)
        self.assertIn('r2', scores)
        self.assertIn('mse', scores)
        self.assertIn('rmse', scores)
        self.assertIn('mae', scores)
        self.assertGreater(scores['r2'], 0.9)
        self.assertAlmostEqual(scores['mae'], 0.5)

    def test_evaluate_regression_perfect(self):
        y = [1.0, 2.0, 3.0, 4.0]
        scores = Evaluator.evaluate_regression(y, y)
        self.assertAlmostEqual(scores['r2'], 1.0)
        self.assertAlmostEqual(scores['mse'], 0.0)
        self.assertAlmostEqual(scores['rmse'], 0.0)
        self.assertAlmostEqual(scores['mae'], 0.0)

    def test_rmse_is_sqrt_mse(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.2, 2.8, 4.1, 5.3]
        scores = Evaluator.evaluate_regression(y_true, y_pred)
        self.assertAlmostEqual(scores['rmse'], np.sqrt(scores['mse']), places=10)


class TestAutoEvaluate(unittest.TestCase):
    def test_auto_classification_int(self):
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 0, 0]
        scores = Evaluator.auto_evaluate(y_true, y_pred)
        self.assertIn('accuracy', scores)

    def test_auto_classification_str(self):
        y_true = ['cat', 'dog', 'cat']
        y_pred = ['cat', 'cat', 'cat']
        scores = Evaluator.auto_evaluate(y_true, y_pred)
        self.assertIn('accuracy', scores)

    def test_auto_regression_float(self):
        y_true = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0,
                  11.1, 12.2, 13.3, 14.4, 15.5, 16.6, 17.7, 18.8, 19.9, 20.0, 21.1]
        y_pred = [1.0, 2.1, 3.4, 4.3, 5.6, 6.5, 7.8, 8.7, 9.8, 10.1,
                  11.0, 12.3, 13.2, 14.5, 15.4, 16.7, 17.6, 18.9, 19.8, 20.1, 21.0]
        scores = Evaluator.auto_evaluate(y_true, y_pred)
        self.assertIn('r2', scores)


if __name__ == '__main__':
    unittest.main()
