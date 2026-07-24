"""
Tests unitaires de DataLoader : datasets intégrés (iris, wine, diabetes).
"""
import unittest

import pandas as pd

from trainedml.data.loader import DataLoader


class TestDataLoaderBuiltins(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()

    def test_iris(self):
        X, y = self.loader.load_dataset(name="iris")
        self.assertEqual(X.shape, (150, 4))
        self.assertEqual(set(y.unique()), {"setosa", "versicolor", "virginica"})

    def test_wine(self):
        X, y = self.loader.load_dataset(name="wine")
        self.assertEqual(X.shape, (178, 13))
        self.assertEqual(set(y.unique()), {0, 1, 2})

    def test_diabetes(self):
        X, y = self.loader.load_dataset(name="diabetes")
        self.assertEqual(X.shape, (442, 10))
        self.assertEqual(y.name, "disease_progression")
        self.assertTrue(pd.api.types.is_float_dtype(y))
        # Cible continue (progression de la maladie), pas une classe :
        # beaucoup plus de valeurs distinctes que d'échantillons partagés.
        self.assertGreater(y.nunique(), 200)

    def test_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            self.loader.load_dataset(name="inexistant")


if __name__ == "__main__":
    unittest.main()
