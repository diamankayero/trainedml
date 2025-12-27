"""
Test unitaire du boxplot avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe BoxplotViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.boxplot import BoxplotViz

class TestBoxplotViz(unittest.TestCase):
    def setUp(self):
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_boxplot_creation(self):
        """Teste la création d'un boxplot sur une colonne numérique du dataset Iris."""
        column = ['sepal_length']
        viz = BoxplotViz(self.data, columns=column)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération du boxplot a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
