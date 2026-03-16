"""
Tests unitaires pour le CLI trainedml.
"""
import unittest
import sys
from unittest.mock import patch
from trainedml.cli import main, _is_classification_target
import pandas as pd
import numpy as np


class TestIsClassificationTarget(unittest.TestCase):
    def test_object_dtype(self):
        y = pd.Series(['cat', 'dog', 'bird'])
        self.assertTrue(_is_classification_target(y))

    def test_few_int_values(self):
        y = pd.Series([0, 1, 2, 0, 1])
        self.assertTrue(_is_classification_target(y))

    def test_many_float_values(self):
        y = pd.Series(np.random.randn(100))
        self.assertFalse(_is_classification_target(y))

    def test_categorical_dtype(self):
        y = pd.Series(pd.Categorical(['a', 'b', 'c']))
        self.assertTrue(_is_classification_target(y))


class TestCLIMain(unittest.TestCase):
    def test_default_run(self):
        """Test CLI with default args (iris + random_forest)."""
        test_args = ['trainedml', '--dataset', 'iris', '--model', 'random_forest', '--seed', '42']
        with patch.object(sys, 'argv', test_args):
            # Should not raise
            try:
                main()
            except SystemExit:
                pass  # argparse may call sys.exit

    def test_benchmark_flag(self):
        """Test CLI with benchmark flag."""
        test_args = ['trainedml', '--dataset', 'iris', '--benchmark', '--seed', '42']
        with patch.object(sys, 'argv', test_args):
            try:
                main()
            except SystemExit:
                pass


if __name__ == '__main__':
    unittest.main()
