"""
Test unitaire de la visualisation de la multicolinéarité avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe MulticollinearityViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.multicollinearity import MulticollinearityViz

class TestMulticollinearityViz(unittest.TestCase):
    def setUp(self):
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_multicollinearity_creation(self):
        """Teste la création d'une visualisation de la multicolinéarité."""
        viz = MulticollinearityViz(self.data)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la visualisation de la multicolinéarité a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
