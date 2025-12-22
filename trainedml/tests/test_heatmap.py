"""
Test unitaire de la heatmap avec chargement automatique d'un dataset public (Iris).
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.heatmap import HeatmapViz

class TestHeatmapViz(unittest.TestCase):
    def setUp(self):
        # Chargement du dataset Iris depuis une URL publique
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_heatmap_creation(self):
        """Teste la création d'une heatmap sur le dataset Iris sans erreur."""
        # On ne garde que les colonnes numériques pour la corrélation
        features = [col for col in self.data.columns if self.data[col].dtype != 'O']
        viz = HeatmapViz(self.data, features=features, method='pearson', mask=True)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la heatmap a échoué : {e}")

if __name__ == '__main__':
    unittest.main()