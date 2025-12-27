"""
Test unitaire de la visualisation de la cible avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe TargetViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.target import TargetViz

class TestTargetViz(unittest.TestCase):
    def setUp(self):
        X, y = DataLoader().load_dataset(name="iris")
        self.data = X.copy()
        self.data['species'] = y

    def test_target_creation(self):
        """Teste la création d'une visualisation de la cible."""
        viz = TargetViz(self.data, target_column='species')
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la visualisation de la cible a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
