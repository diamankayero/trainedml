"""
Test unitaire de l'histogramme avec chargement automatique d'un dataset public (Iris).
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.histogram import HistogramViz

class TestHistogramViz(unittest.TestCase):
    def setUp(self):
        # Chargement du dataset Iris depuis une URL publique
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_histogram_creation(self):
        """Teste la création d'un histogramme sur une colonne numérique du dataset Iris."""
        # On choisit une colonne numérique (par exemple 'sepal_length')
        column = ['sepal_length']
        viz = HistogramViz(self.data, columns=column, legend=False, bins=10)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de l'histogramme a échoué : {e}")

if __name__ == '__main__':
    unittest.main()