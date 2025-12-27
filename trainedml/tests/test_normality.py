"""
Test unitaire de la visualisation de la normalité avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe NormalityViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.normality import NormalityViz

class TestNormalityViz(unittest.TestCase):
    def setUp(self):
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_normality_creation(self):
        """Teste la création d'une visualisation de la normalité sur une colonne numérique."""
        column = ['sepal_length']
        viz = NormalityViz(self.data, columns=column)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération de la visualisation de la normalité a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
