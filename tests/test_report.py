"""
Tests unitaires du rapport EDA HTML (trainedml.report).
"""
import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from trainedml.report import generate_report
from trainedml.visualization import Visualizer


class TestReport(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.data = pd.DataFrame({
            "x": rng.normal(size=60),
            "y": rng.uniform(0, 10, 60),
            "cat": rng.choice(["a", "b"], 60),
        })
        self.data.loc[0, "x"] = np.nan

    def test_generate_report_returns_html(self):
        html = generate_report(self.data)
        self.assertIn("<html", html)
        self.assertIn("Descriptive statistics", html)
        self.assertIn("Missing values", html)
        self.assertIn("Correlations", html)
        self.assertIn("data:image/png;base64", html)

    def test_generate_report_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rapport.html")
            generate_report(self.data, path=path, title="Test")
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                self.assertIn("Test", f.read())

    def test_visualizer_report(self):
        html = Visualizer(self.data).report()
        self.assertIn("<html", html)


if __name__ == "__main__":
    unittest.main()
