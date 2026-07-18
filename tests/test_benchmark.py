"""
Tests unitaires pour le module Benchmark.
"""
import unittest
from sklearn.datasets import make_classification
from trainedml.benchmark import Benchmark
from trainedml.models.knn import KNNModel
from trainedml.models.random_forest import RandomForestModel


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.X_train, self.X_test = X[:70], X[70:]
        self.y_train, self.y_test = y[:70], y[70:]
        self.models = {
            'knn': KNNModel(),
            'rf': RandomForestModel(),
        }

    def test_run_sequential(self):
        bench = Benchmark(self.models)
        results = bench.run(
            self.X_train, self.y_train, self.X_test, self.y_test,
            show_progress=False
        )
        self.assertIsInstance(results, dict)
        self.assertIn('knn', results)
        self.assertIn('rf', results)
        for res in results.values():
            self.assertIn('scores', res)
            self.assertIn('fit_time', res)
            self.assertIn('predict_time', res)
            self.assertIn('accuracy', res['scores'])

    def test_run_parallel(self):
        bench = Benchmark(self.models)
        results = bench.run(
            self.X_train, self.y_train, self.X_test, self.y_test,
            parallel=True, show_progress=False
        )
        self.assertIn('knn', results)
        self.assertIn('rf', results)

    def test_summary_before_run(self):
        bench = Benchmark(self.models)
        self.assertIsNone(bench.summary())

    def test_summary_after_run(self):
        bench = Benchmark(self.models)
        bench.run(
            self.X_train, self.y_train, self.X_test, self.y_test,
            show_progress=False
        )
        summary = bench.summary()
        self.assertIsInstance(summary, str)
        self.assertIn('knn', summary)
        self.assertIn('rf', summary)


if __name__ == '__main__':
    unittest.main()
