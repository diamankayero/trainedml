"""
Module principal du package trainedml.
Contient la classe Trainer pour l'entraînement, l'évaluation et la prédiction de modèles ML,
et la fonction main pour l'entrée CLI.
"""

# Ce fichier permet d'importer le package trainedml

# Classe Trainer pour usage API et webapp
from .data.loader import DataLoader
from .models.knn import KNNModel
from .models.logistic import LogisticModel
from .models.random_forest import RandomForestModel
from .evaluation import Evaluator
from sklearn.model_selection import train_test_split

MODEL_MAP = {
	'knn': KNNModel,
	'logistic': LogisticModel,
	'random_forest': RandomForestModel
}

class Trainer:
    """
    Classe haut niveau pour entraîner, évaluer et prédire avec un modèle ML.

    Cette classe centralise le workflow machine learning : chargement des données,
    séparation train/test, entraînement, évaluation et prédiction.
    Elle est conçue pour être utilisée dans une API, une webapp ou en script.
    """
    def __init__(self, dataset=None, model='random_forest', url=None, target=None, test_size=0.2, seed=42):
        """
        Initialise un objet Trainer.

        Args:
            dataset (str, optional): nom du dataset connu ("iris", "wine", etc.)
            model (str): nom du modèle à utiliser ("random_forest", "knn", "logistic")
            url (str, optional): URL d'un CSV distant
            target (str, optional): nom de la colonne cible (si url)
            test_size (float): proportion de test (0-1)
            seed (int): graine aléatoire pour la reproductibilité
        """
        self.dataset = dataset
        self.url = url
        self.target = target
        self.test_size = test_size
        self.seed = seed
        self.model_name = model
        self.model = MODEL_MAP[model]()
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.is_fitted = False

    def load_data(self):
        """
        Charge les données, effectue la séparation train/test et les stocke dans l'objet.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        loader = DataLoader()
        X, y = loader.load_dataset(name=self.dataset, url=self.url, target=self.target)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self):
        """
        Entraîne le modèle sur les données d'entraînement.
        Charge les données si nécessaire.
        """
        if self.X_train is None:
            self.load_data()
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True

    def evaluate(self):
        """
        Évalue le modèle entraîné sur les données de test.

        Returns:
            dict: scores de classification (accuracy, precision, recall, f1)
        Raises:
            RuntimeError: si le modèle n'est pas entraîné
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant l'évaluation.")
        y_pred = self.model.predict(self.X_test)
        return Evaluator.evaluate_all(self.y_test, y_pred)

    def predict(self, X):
        """
        Prédit la cible pour de nouvelles données X.

        Args:
            X (array-like): données d'entrée (mêmes features que l'entraînement)
        Returns:
            array: prédictions du modèle
        Raises:
            RuntimeError: si le modèle n'est pas entraîné
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant la prédiction.")
        import numpy as np
        X = np.array(X)
        return self.model.predict(X)

def main():
    """
    Point d'entrée CLI du package trainedml.
    Lance l'interface en ligne de commande (voir src/trainedml/cli.py).
    """
    from .cli import main as cli_main
    cli_main()
