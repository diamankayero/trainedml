"""
Test unitaire du module ProfilingViz et de la fonction profiling_report.
"""
import unittest
import pandas as pd
from trainedml.viz.profiling import ProfilingViz, profiling_report

class TestProfilingViz(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, None, 3, None, 5]
        })

    def test_profiling_viz(self):
        viz = ProfilingViz(self.df)
        viz.vizs()
        self.assertIsNotNone(viz.figure)
        self.assertTrue('missing' in viz.figure.columns)

    def test_profiling_report(self):
        report = profiling_report(self.df)
        self.assertIsInstance(report, dict)
        self.assertIn('describe', report)

if __name__ == '__main__':
    unittest.main()
