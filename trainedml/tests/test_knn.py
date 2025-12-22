"""
Test unitaire du modèle KNN avec chargement automatique d'un dataset public (Iris).
"""
import unittest
from trainedml.data.loader import DataLoader
from trainedml.models.knn import KNNModel
from sklearn.model_selection import train_test_split

class TestKNNModel(unittest.TestCase):
    def setUp(self):
        # Chargement du dataset Iris depuis une URL publique
        X, y = DataLoader().load_dataset(name="iris")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def test_fit_predict(self):
        """Teste l'entraînement et la prédiction du modèle KNN sur Iris."""
        model = KNNModel(n_neighbors=3)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.y_test))

    def test_evaluate(self):
        """Teste l'évaluation du modèle KNN sur Iris."""
        model = KNNModel(n_neighbors=3)
        model.fit(self.X_train, self.y_train)
        score = model.evaluate(self.X_test, self.y_test)
        self.assertTrue(0.0 <= score <= 1.0)

if __name__ == '__main__':
    unittest.main()