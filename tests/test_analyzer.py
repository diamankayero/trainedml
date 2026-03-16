"""
Tests unitaires pour le module DataAnalyzer.
"""
import unittest
import numpy as np
import pandas as pd
from trainedml.analyzer import DataAnalyzer


class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100) * 2 + 5,
            'C': np.random.choice(['cat', 'dog', 'bird'], 100),
            'D': np.random.randint(0, 10, 100).astype(float),
        })
        self.df.loc[3, 'A'] = np.nan
        self.df.loc[7, 'B'] = np.nan
        self.analyzer = DataAnalyzer(self.df)

    # --- distribution ---
    def test_distribution_returns_dict(self):
        result = self.analyzer.distribution(columns=['A', 'B'])
        self.assertIsInstance(result, dict)
        self.assertIn('describe', result)
        self.assertIn('figure', result)

    def test_distribution_has_stats(self):
        result = self.analyzer.distribution(columns=['A'])
        desc = result['describe']
        self.assertIn('A', desc.columns)

    # --- correlation ---
    def test_correlation_returns_dataframe(self):
        corr = self.analyzer.correlation()
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertGreater(corr.shape[0], 0)

    def test_correlation_method(self):
        corr = self.analyzer.correlation(method='spearman')
        self.assertIsInstance(corr, pd.DataFrame)

    # --- missing ---
    def test_missing_returns_dataframe(self):
        result = self.analyzer.missing()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('count', result.columns)
        self.assertIn('percent', result.columns)

    def test_missing_detects_nans(self):
        result = self.analyzer.missing()
        self.assertGreater(result.loc['A', 'count'], 0)

    # --- outliers ---
    def test_outliers_iqr(self):
        result = self.analyzer.outliers(method='iqr')
        self.assertIsInstance(result, dict)
        for col_data in result.values():
            self.assertIn('count', col_data)
            self.assertIn('indices', col_data)
            self.assertIn('lower_bound', col_data)
            self.assertIn('upper_bound', col_data)

    def test_outliers_zscore(self):
        result = self.analyzer.outliers(method='zscore', threshold=2.0)
        self.assertIsInstance(result, dict)

    # --- target ---
    def test_target_categorical(self):
        result = self.analyzer.target('C')
        self.assertIsInstance(result, dict)
        self.assertIn('value_counts', result)
        self.assertIn('n_unique', result)
        self.assertEqual(result['n_unique'], 3)

    def test_target_numeric(self):
        result = self.analyzer.target('A')
        self.assertIsInstance(result, dict)

    # --- normality ---
    def test_normality_returns_dict(self):
        result = self.analyzer.normality(columns=['A', 'B'])
        self.assertIsInstance(result, dict)
        self.assertIn('A', result)

    def test_normality_has_tests(self):
        result = self.analyzer.normality(columns=['A'])
        self.assertIn('shapiro', result['A'])
        self.assertIn('skewness', result['A'])
        self.assertIn('kurtosis', result['A'])

    # --- multicollinearity ---
    def test_multicollinearity_returns_dataframe(self):
        result = self.analyzer.multicollinearity()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('feature', result.columns)
        self.assertIn('VIF', result.columns)

    # --- profiling ---
    def test_profiling_returns_dict(self):
        result = self.analyzer.profiling()
        self.assertIsInstance(result, dict)
        self.assertIn('describe', result)
        self.assertIn('missing', result)
        self.assertIn('outliers', result)
        self.assertIn('correlation', result)
        self.assertIn('shape', result)
        self.assertIn('dtypes', result)


if __name__ == '__main__':
    unittest.main()
