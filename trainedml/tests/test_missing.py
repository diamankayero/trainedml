"""
Test unitaire de la visualisation des valeurs manquantes avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe MissingValuesViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.missing import MissingValuesViz

class TestMissingValuesViz(unittest.TestCase):
    def setUp(self):
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_missing_creation(self):
        """Teste la création d'une visualisation des valeurs manquantes."""
        viz = MissingValuesViz(self.data)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la visualisation des valeurs manquantes a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
