"""
Test unitaire de la visualisation de distribution avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe DistributionViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.distribution import DistributionViz

class TestDistributionViz(unittest.TestCase):
    def setUp(self):
        # Chargement du dataset Iris depuis une source publique
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_distribution_creation(self):
        """Teste la création d'une distribution sur une colonne numérique du dataset Iris."""
        # On choisit une colonne numérique (par exemple 'sepal_length')
        column = ['sepal_length']
        viz = DistributionViz(self.data, columns=column)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la distribution a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
