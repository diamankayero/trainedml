"""
Test unitaire du profiling automatique avec chargement automatique d'un dataset public (Iris).
Ce test vérifie que la classe ProfilingViz fonctionne sans erreur et génère bien une figure.
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.viz.profiling import ProfilingViz

class TestProfilingViz(unittest.TestCase):
    def setUp(self):
        self.data = DataLoader().load_dataset(name="iris")[0].copy()

    def test_profiling_creation(self):
        """Teste la création d'un rapport de profiling automatique."""
        viz = ProfilingViz(self.data)
        try:
            viz.vizs()
            self.assertIsNotNone(viz.figure)
        except Exception as e:
            self.fail(f"La génération du profiling automatique a échoué : {e}")

if __name__ == '__main__':
    unittest.main()
