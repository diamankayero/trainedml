
"""
Module principal du package trainedml.

Ce module expose la classe centrale `Trainer` qui permet de gérer tout le workflow de machine learning :
chargement de données, séparation train/test, entraînement, évaluation et prédiction.
Il sert aussi de point d'entrée pour la CLI (ligne de commande).

Fonctionnalités principales
--------------------------
- API haut niveau pour entraîner, évaluer et prédire avec un modèle ML
- Supporte les datasets publics (Iris, Wine, etc.) ou des CSV distants
- Séparation automatique train/test
- Gestion de plusieurs modèles (KNN, Logistic, Random Forest, etc.)
- Évaluation standard (accuracy, precision, recall, f1)
- Peut être utilisé en script, API, ou webapp

Exemple
-------
>>> from trainedml import Trainer
>>> trainer = Trainer(dataset="iris", model="knn")
>>> trainer.fit()
>>> results = trainer.evaluate()
>>> print(results)
>>> preds = trainer.predict([[5.1, 3.5, 1.4, 0.2]])
>>> print(preds)
"""

# Ce fichier permet d'importer le package trainedml

# Classe Trainer pour usage API et webapp
from .data.loader import DataLoader
from .models import KNNModel, LogisticModel, RandomForestModel, MODEL_MAP, get_model
from .evaluation import Evaluator



class Trainer:
    r"""
    Classe haut niveau pour entraîner, évaluer et prédire avec un modèle de machine learning.

    Cette classe centralise tout le workflow ML : chargement des données, split train/test,
    entraînement, évaluation et prédiction. Elle est conçue pour être utilisée dans une API,
    une webapp ou en script Python.

    Parameters
    ----------
    dataset : str, optional
        Nom du dataset connu ("iris", "wine", etc.).
    model : str
        Nom du modèle à utiliser ("random_forest", "knn", "logistic").
    url : str, optional
        URL d'un CSV distant à charger.
    target : str, optional
        Nom de la colonne cible (si url).
    test_size : float
        Proportion de test (entre 0 et 1).
    seed : int
        Graine aléatoire pour la reproductibilité.

    Attributes
    ----------
    model : BaseModel
        Instance du modèle ML utilisé.
    X_train, X_test, y_train, y_test : array-like
        Données séparées pour l'entraînement et le test.
    is_fitted : bool
        Indique si le modèle a été entraîné.

    Examples
    --------
    >>> trainer = Trainer(dataset="iris", model="knn")
    >>> trainer.fit()
    >>> results = trainer.evaluate()
    >>> print(results)
    >>> preds = trainer.predict([[5.1, 3.5, 1.4, 0.2]])
    >>> print(preds)
    """
    def __init__(self, dataset=None, model='random_forest', url=None, target=None, test_size=0.2, seed=42):
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

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)

        Raises
        ------
        ValueError
            Si le dataset ou la cible n'est pas spécifié correctement.
        """
        loader = DataLoader()
        X, y = loader.load_dataset(name=self.dataset, url=self.url, target=self.target)
        self.X_train, self.X_test, self.y_train, self.y_test = loader.split(
            X, y, test_size=self.test_size, random_state=self.seed)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self):
        """
        Entraîne le modèle sur les données d'entraînement.
        Charge les données si nécessaire.

        Returns
        -------
        self : Trainer
            L'instance courante (pour chaînage).
        """
        if self.X_train is None:
            self.load_data()
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        return self

    def evaluate(self):
        """
        Évalue le modèle entraîné sur les données de test.

        Returns
        -------
        dict
            Dictionnaire des scores de classification (accuracy, precision, recall, f1).

        Raises
        ------
        RuntimeError
            Si le modèle n'est pas entraîné.
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant l'évaluation.")
        y_pred = self.model.predict(self.X_test)
        return Evaluator.evaluate_all(self.y_test, y_pred)

    def predict(self, X):
        """
        Prédit la cible pour de nouvelles données X.

        Parameters
        ----------
        X : array-like
            Données d'entrée (mêmes features que l'entraînement).

        Returns
        -------
        array
            Prédictions du modèle.

        Raises
        ------
        RuntimeError
            Si le modèle n'est pas entraîné.
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
