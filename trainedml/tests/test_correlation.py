"""
Test unitaire de la visualisation de corrélation avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe CorrelationViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.correlation import CorrelationViz

class TestCorrelationViz(unittest.TestCase):
    def setUp(self):
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_correlation_creation(self):
        """Teste la création d'une visualisation de corrélation sur les colonnes numériques."""
        features = [col for col in self.data.columns if self.data[col].dtype != 'O']
        viz = CorrelationViz(self.data, features=features)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la visualisation de corrélation a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
