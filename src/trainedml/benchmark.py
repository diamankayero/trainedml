"""
Module de benchmark pour comparer plusieurs modèles sur un même jeu de données.
"""
from .evaluation import Evaluator

class Benchmark:
    """
    Classe pour comparer les performances de plusieurs modèles de classification.
    """
    def __init__(self, models):
        """
        Args:
            models (dict): dictionnaire {nom: instance_modele}
        """
        self.models = models

    def run(self, X_train, y_train, X_test, y_test):
        """
        Entraîne et évalue chaque modèle, retourne les scores et les temps d'exécution.
        Returns:
            dict: {nom_modele: {scores, fit_time, predict_time}}
        """
        import time
        results = {}
        for name, model in self.models.items():
            # Mesure du temps d'entraînement
            start_fit = time.time()
            model.fit(X_train, y_train)
            fit_time = time.time() - start_fit

            # Mesure du temps de prédiction
            start_pred = time.time()
            y_pred = model.predict(X_test)
            predict_time = time.time() - start_pred

            scores = Evaluator.evaluate_all(y_test, y_pred)
            results[name] = {
                'scores': scores,
                'fit_time': fit_time,
                'predict_time': predict_time
            }
        return results
