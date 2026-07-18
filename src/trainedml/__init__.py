"""
Module principal du package trainedml.

Ce module expose la classe centrale `Trainer` qui permet de gérer tout le workflow de machine learning :
chargement de données, séparation train/test, prétraitement, entraînement, évaluation, prédiction
et persistance. Il expose aussi la fonction :func:`compare` pour comparer tous les modèles
d'une tâche en une ligne.

Fonctionnalités principales
--------------------------
- API haut niveau pour entraîner, évaluer et prédire avec un modèle ML
- Supporte les datasets intégrés (Iris, Wine) ou des CSV distants
- Prétraitement automatique (imputation, standardisation, encodage one-hot)
- Séparation automatique train/test, seed variable sans recréer l'objet
- Gestion des modèles trainedml (KNN, Logistic, Random Forest, régresseurs...)
  et de **n'importe quel estimateur scikit-learn** (ou objet fit/predict)
- Évaluation adaptée à la tâche : classification (accuracy, precision, recall, f1)
  ou régression (r2, mse, rmse, mae)
- Sauvegarde et rechargement du modèle entraîné (:meth:`Trainer.save` / :meth:`Trainer.load`)
- Peut être utilisé en script, API, CLI ou webapp

Exemple
-------
>>> from trainedml import Trainer, compare
>>> trainer = Trainer(dataset="iris", model="knn", model_params={"n_neighbors": 5})
>>> trainer.fit()
>>> print(trainer.evaluate())
>>> preds = trainer.predict([[5.1, 3.5, 1.4, 0.2]])
>>> compare(dataset="iris", cv=5)  # comparatif de tous les modèles en DataFrame
"""

from __future__ import annotations

__version__ = "0.2.0"

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd

from .data.loader import DataLoader
from .models import (
    KNNModel, LogisticModel, RandomForestModel,
    MODEL_MAP, CLASSIFIER_MAP, REGRESSOR_MAP, get_model,
)
from .evaluation import Evaluator
from .tasks import detect_task, detect_model_task
from .preprocessing import build_preprocessor, PreprocessedModel
from .benchmark import Benchmark
from .compare import compare
from .visualization import Visualizer

__all__ = [
    "Trainer", "DataLoader", "Evaluator", "Benchmark", "Visualizer",
    "compare", "get_model", "detect_task",
    "MODEL_MAP", "CLASSIFIER_MAP", "REGRESSOR_MAP",
    "KNNModel", "LogisticModel", "RandomForestModel",
    "build_preprocessor", "PreprocessedModel",
]


class Trainer:
    r"""
    Classe haut niveau pour entraîner, évaluer et prédire avec un modèle de machine learning.

    Cette classe centralise tout le workflow ML : chargement des données, split train/test,
    prétraitement, entraînement, évaluation, prédiction et persistance. Elle est conçue pour
    être utilisée dans une API, une webapp ou en script Python.

    Parameters
    ----------
    dataset : str, optional
        Nom du dataset connu ("iris", "wine").
    model : str or object, default='random_forest'
        Nom d'un modèle trainedml ("random_forest", "knn", "logistic", "linear",
        "ridge", "lasso", ...) **ou** n'importe quel estimateur possédant les
        méthodes ``fit`` et ``predict`` (par exemple un estimateur scikit-learn).
    url : str, optional
        URL d'un CSV distant à charger.
    target : str, optional
        Nom de la colonne cible (si url).
    X : pandas.DataFrame or array-like, optional
        Features fournies directement en mémoire (alternative à dataset/url).
    y : pandas.Series or array-like, optional
        Cible correspondante (obligatoire si X est fourni).
    test_size : float, default=0.2
        Proportion de test (entre 0 et 1).
    seed : int, default=42
        Graine aléatoire pour la reproductibilité.
    model_params : dict, optional
        Hyperparamètres passés au constructeur du modèle (uniquement si ``model``
        est un nom ; pour un estimateur déjà instancié, configurez-le directement).
    preprocess : bool, default=True
        Si True, applique le prétraitement standard de trainedml (imputation,
        standardisation des colonnes numériques, encodage one-hot des colonnes
        catégorielles). Le préprocesseur est entraîné sur le train uniquement.

    Attributes
    ----------
    model : object
        Instance du modèle ML utilisé.
    preprocessor : sklearn.compose.ColumnTransformer or None
        Préprocesseur (None si ``preprocess=False``).
    X_train, X_test, y_train, y_test : array-like
        Données séparées pour l'entraînement et le test (non prétraitées).
    task : str
        Type de tâche ('classification' ou 'regression').
    is_fitted : bool
        Indique si le modèle a été entraîné.

    Examples
    --------
    Workflow standard :

    >>> trainer = Trainer(dataset="iris", model="knn")
    >>> trainer.fit()
    >>> print(trainer.evaluate())
    >>> preds = trainer.predict([[5.1, 3.5, 1.4, 0.2]])

    Hyperparamètres et estimateur scikit-learn arbitraire :

    >>> trainer = Trainer(dataset="wine", model="knn", model_params={"n_neighbors": 7})
    >>> from sklearn.svm import SVC
    >>> trainer = Trainer(dataset="iris", model=SVC(kernel="rbf"))

    Varier le seed sans recréer le Trainer :

    >>> for s in range(5):
    ...     print(trainer.fit(seed=s).evaluate())

    Sauvegarde et rechargement :

    >>> trainer.save("model.joblib")
    >>> restored = Trainer.load("model.joblib")
    >>> restored.predict([[5.1, 3.5, 1.4, 0.2]])
    """

    def __init__(self, dataset: Optional[str] = None, model: Union[str, Any] = 'random_forest',
                 url: Optional[str] = None, target: Optional[str] = None,
                 X: Optional["pd.DataFrame"] = None, y: Optional["pd.Series"] = None,
                 test_size: float = 0.2, seed: int = 42,
                 model_params: Optional[Dict[str, Any]] = None, preprocess: bool = True) -> None:
        self.dataset = dataset
        self.url = url
        self.target = target
        self._X, self._y = X, y
        self.test_size = test_size
        self.seed = seed
        self.preprocess = preprocess
        self.model_params = dict(model_params or {})

        if isinstance(model, str):
            if model not in MODEL_MAP:
                raise ValueError(
                    f"Modèle inconnu : {model!r}. Disponibles : {list(MODEL_MAP.keys())} "
                    f"(ou passez directement un estimateur fit/predict)."
                )
            self.model_name = model
            self.model = MODEL_MAP[model](**self.model_params)
        else:
            if not (hasattr(model, "fit") and hasattr(model, "predict")):
                raise TypeError(
                    "`model` doit être un nom de modèle trainedml ou un objet "
                    "avec les méthodes fit et predict (ex. estimateur scikit-learn)."
                )
            if self.model_params:
                raise ValueError(
                    "`model_params` n'est utilisable qu'avec un nom de modèle ; "
                    "configurez directement votre estimateur."
                )
            self.model_name = type(model).__name__
            self.model = model

        self.preprocessor = build_preprocessor() if preprocess else None
        self.feature_names_: Optional[List[str]] = None
        self.X_train: Any = None
        self.X_test: Any = None
        self.y_train: Any = None
        self.y_test: Any = None
        self.is_fitted = False
        self._task: Optional[str] = None

    @property
    def task(self) -> str:
        """Type de tâche du modèle ('classification' ou 'regression')."""
        if self._task is not None:
            return self._task
        return detect_model_task(self.model, self.y_train)

    def load_data(self, test_size: Optional[float] = None,
                  seed: Optional[int] = None) -> Tuple[Any, Any, Any, Any]:
        """
        Charge les données, effectue la séparation train/test et les stocke dans l'objet.

        La séparation utilise l'API du package (:meth:`DataLoader.split`) : il n'est
        jamais nécessaire d'appeler scikit-learn directement. Les ensembles obtenus
        sont accessibles via les attributs ``X_train``, ``X_test``, ``y_train``, ``y_test``.

        Parameters
        ----------
        test_size : float, optional
            Nouvelle proportion de test. Si fournie, remplace celle du constructeur.
        seed : int, optional
            Nouvelle graine aléatoire. Si fournie, remplace celle du constructeur,
            ce qui permet de faire varier le split sans recréer le Trainer.

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)

        Raises
        ------
        ValueError
            Si le dataset ou la cible n'est pas spécifié correctement.

        Examples
        --------
        >>> trainer = Trainer(dataset="iris", model="knn")
        >>> X_train, X_test, y_train, y_test = trainer.load_data(seed=7)
        """
        if test_size is not None:
            self.test_size = test_size
        if seed is not None:
            self.seed = seed
        loader = DataLoader()
        if self._X is not None and self._y is not None:
            # Données fournies directement en mémoire (X=..., y=...)
            X, y = self._X, self._y
        else:
            X, y = loader.load_dataset(name=self.dataset, url=self.url, target=self.target)
        self.X_train, self.X_test, self.y_train, self.y_test = loader.split(
            X, y, test_size=self.test_size, random_state=self.seed)
        # Un nouveau split invalide tout entraînement précédent
        self.is_fitted = False
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _transform(self, X: Any, fit: bool = False) -> Any:
        """Applique le préprocesseur (fit_transform sur le train, transform sinon)."""
        if self.preprocessor is None:
            return X
        return self.preprocessor.fit_transform(X) if fit else self.preprocessor.transform(X)

    def fit(self, test_size: Optional[float] = None, seed: Optional[int] = None) -> "Trainer":
        """
        Entraîne le préprocesseur (si activé) puis le modèle sur les données d'entraînement.
        Charge les données si nécessaire.

        Parameters
        ----------
        test_size : float, optional
            Si fournie, les données sont re-séparées avec cette proportion avant l'entraînement.
        seed : int, optional
            Si fournie, les données sont re-séparées avec cette graine avant l'entraînement.
            Permet d'évaluer la stabilité d'un modèle sur plusieurs splits :
            ``for s in range(5): print(trainer.fit(seed=s).evaluate())``

        Returns
        -------
        self : Trainer
            L'instance courante (pour chaînage).
        """
        if self.X_train is None or test_size is not None or seed is not None:
            self.load_data(test_size=test_size, seed=seed)
        if hasattr(self.X_train, "columns"):
            self.feature_names_ = list(self.X_train.columns)
        X_t = self._transform(self.X_train, fit=True)
        self.model.fit(X_t, self.y_train)
        self._task = detect_model_task(self.model, self.y_train)
        self.is_fitted = True
        return self

    def evaluate(self) -> Dict[str, float]:
        """
        Évalue le modèle entraîné sur les données de test, avec les métriques
        adaptées à la tâche : classification (accuracy, precision, recall, f1)
        ou régression (r2, mse, rmse, mae).

        Returns
        -------
        dict
            Dictionnaire des scores.

        Raises
        ------
        RuntimeError
            Si le modèle n'est pas entraîné.
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant l'évaluation.")
        y_pred = self.model.predict(self._transform(self.X_test))
        return Evaluator.evaluate_for(self.task, self.y_test, y_pred)

    def _as_frame(self, X: Any) -> "pd.DataFrame":
        """Convertit X en DataFrame avec les colonnes de l'entraînement si possible."""
        import numpy as np
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.feature_names_ is not None and X.shape[1] == len(self.feature_names_):
            return pd.DataFrame(X, columns=self.feature_names_)
        return pd.DataFrame(X)

    def predict(self, X: Any) -> "np.ndarray":
        """
        Prédit la cible pour de nouvelles données X.

        Le prétraitement appris à l'entraînement est appliqué automatiquement :
        X doit contenir les mêmes features (mêmes colonnes) que l'entraînement.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
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
        X = self._as_frame(X)
        return self.model.predict(self._transform(X))

    def save(self, path: Union[str, "Path"]) -> None:
        """
        Sauvegarde le modèle entraîné (et son préprocesseur) sur disque.

        Parameters
        ----------
        path : str or pathlib.Path
            Chemin du fichier de sortie (convention : extension ``.joblib``).

        Raises
        ------
        RuntimeError
            Si le modèle n'est pas entraîné.

        Examples
        --------
        >>> trainer.fit().save("model.joblib")
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant la sauvegarde.")
        import joblib
        payload = {
            "trainedml_version": __version__,
            "model": self.model,
            "model_name": self.model_name,
            "model_params": self.model_params,
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names_,
            "task": self.task,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Union[str, "Path"]) -> "Trainer":
        """
        Recharge un Trainer sauvegardé avec :meth:`save`, prêt à prédire.

        Parameters
        ----------
        path : str or pathlib.Path
            Chemin du fichier sauvegardé.

        Returns
        -------
        Trainer
            Instance prête pour :meth:`predict` (les données d'origine ne sont
            pas rechargées ; ``X_train`` etc. valent None).

        Examples
        --------
        >>> restored = Trainer.load("model.joblib")
        >>> restored.predict([[5.1, 3.5, 1.4, 0.2]])
        """
        import joblib
        payload = joblib.load(path)
        trainer = cls.__new__(cls)
        trainer.dataset = None
        trainer.url = None
        trainer.target = None
        # Valeurs par défaut : les données d'origine ne sont pas rechargées
        trainer.test_size = 0.2
        trainer.seed = 42
        trainer.preprocess = payload["preprocessor"] is not None
        trainer.model = payload["model"]
        trainer.model_name = payload["model_name"]
        trainer.model_params = payload.get("model_params", {})
        trainer.preprocessor = payload["preprocessor"]
        trainer.feature_names_ = payload.get("feature_names")
        trainer.X_train = trainer.X_test = trainer.y_train = trainer.y_test = None
        trainer._task = payload.get("task")
        trainer.is_fitted = True
        return trainer


def main() -> None:
    """
    Point d'entrée CLI du package trainedml.
    Lance l'interface en ligne de commande (voir src/trainedml/cli.py).
    """
    from .cli import main as cli_main
    cli_main()
