"""
Test unitaire de la visualisation bivariée avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe BivariateViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.bivariate import BivariateViz

class TestBivariateViz(unittest.TestCase):
    def setUp(self):
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_bivariate_creation(self):
        """Teste la création d'une visualisation bivariée entre deux colonnes numériques."""
        x = 'sepal_length'
        y = 'sepal_width'
        viz = BivariateViz(self.data, x, y)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la visualisation bivariée a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
