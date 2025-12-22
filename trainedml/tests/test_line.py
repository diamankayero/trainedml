"""
Test unitaire du line plot avec chargement automatique d'un dataset public (Iris).
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.line import LineViz

class TestLineViz(unittest.TestCase):
    def setUp(self):
        # Chargement du dataset Iris depuis une URL publique
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_line_creation(self):
        """Teste la création d'une courbe entre deux colonnes numériques du dataset Iris."""
        x_column = 'sepal_length'
        y_column = 'sepal_width'
        viz = LineViz(self.data, x_column=x_column, y_column=y_column)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la courbe a échoué : {e}")

if __name__ == '__main__':
    unittest.main()