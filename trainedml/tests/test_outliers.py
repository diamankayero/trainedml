"""
Test unitaire de la visualisation des outliers avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe OutliersViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.outliers import OutliersViz

class TestOutliersViz(unittest.TestCase):
    def setUp(self):
        # Chargement du dataset Iris depuis une source publique
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_outliers_creation(self):
        """Teste la création d'une visualisation des outliers sur les colonnes numériques du dataset Iris."""
        viz = OutliersViz(self.data)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la visualisation des outliers a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
