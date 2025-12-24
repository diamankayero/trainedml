"""
Module de benchmark pour comparer plusieurs modèles sur un même jeu de données.
"""
from .evaluation import Evaluator
import time

class Benchmark:
    """
    Classe pour comparer les performances de plusieurs modèles de classification.
    """
    def __init__(self, models):
        """
        Initialise le benchmark avec un dictionnaire de modèles.
        Args:
            models (dict): dictionnaire {nom: instance_modele}
        """
        self.models = models  # Stocke les modèles à comparer

    def run(self, X_train, y_train, X_test, y_test):
        """
        Entraîne et évalue chaque modèle sur les données fournies.

        Args:
            X_train: Données d'entraînement (features)
            y_train: Labels d'entraînement
            X_test: Données de test (features)
            y_test: Labels de test

        Returns:
            dict: {nom_modele: {scores, fit_time, predict_time}}
        """
        results = {}  # Dictionnaire pour stocker les résultats de chaque modèle
        for name, model in self.models.items():
            # Mesure du temps d'entraînement du modèle
            start_fit = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start_fit

            # Mesure du temps de prédiction du modèle
            start_pred = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - start_pred

            # Évaluation des prédictions avec toutes les métriques disponibles
            scores = Evaluator.evaluate_all(y_test, y_pred)
            results[name] = {
                'scores': scores,           # Dictionnaire des scores de métriques
                'fit_time': fit_time,       # Temps d'entraînement
                'predict_time': predict_time # Temps de prédiction
            }
        return results  # Retourne les résultats pour tous les modèles
