"""
Prétraitement automatique des données pour trainedml.

Ce module fournit un préprocesseur standard construit avec scikit-learn :

- **colonnes numériques** : imputation par la médiane puis standardisation
  (moyenne 0, écart-type 1) — indispensable pour KNN, la régression logistique,
  Ridge/Lasso... ;
- **colonnes catégorielles** : imputation par le mode puis encodage one-hot
  (les catégories inconnues à la prédiction sont ignorées).

Il est utilisé par défaut par :class:`trainedml.Trainer` (``preprocess=True``)
et par :func:`trainedml.compare`, mais peut aussi s'employer seul.

Exemple
-------
>>> from trainedml.preprocessing import build_preprocessor
>>> pre = build_preprocessor()
>>> X_train_t = pre.fit_transform(X_train)
>>> X_test_t = pre.transform(X_test)
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_onehot() -> OneHotEncoder:
    """Crée un OneHotEncoder dense, compatible avec toutes les versions de scikit-learn."""
    try:
        # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """
    Construit le préprocesseur standard de trainedml.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Transformeur non entraîné : imputation médiane + standardisation pour
        les colonnes numériques, imputation mode + one-hot pour les colonnes
        catégorielles. À entraîner sur les données d'entraînement uniquement
        (``fit_transform``), puis à appliquer aux données de test (``transform``).

    Examples
    --------
    >>> pre = build_preprocessor()
    >>> X_train_t = pre.fit_transform(X_train)
    >>> X_test_t = pre.transform(X_test)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", _make_onehot()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )


class PreprocessedModel:
    """
    Enveloppe un modèle avec le préprocesseur standard de trainedml.

    À chaque ``fit``, le préprocesseur est (ré)entraîné sur les données
    d'entraînement uniquement, puis appliqué aux données de prédiction :
    aucune fuite d'information du test vers l'entraînement, y compris en
    validation croisée.

    Parameters
    ----------
    model : object
        Modèle à envelopper (trainedml ou scikit-learn, tout objet fit/predict).

    Attributes
    ----------
    model : object
        Le modèle enveloppé.
    preprocessor : sklearn.compose.ColumnTransformer
        Le préprocesseur, réentraîné à chaque appel de :meth:`fit`.
    task : str
        Type de tâche du modèle enveloppé (délégué).

    Examples
    --------
    >>> from trainedml.models import KNNModel
    >>> model = PreprocessedModel(KNNModel(n_neighbors=3))
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """

    def __init__(self, model):
        self.model = model
        self.preprocessor = build_preprocessor()

    @property
    def task(self):
        """Type de tâche du modèle enveloppé ('classification' ou 'regression')."""
        return getattr(self.model, "task", None)

    def fit(self, X, y):
        """Entraîne le préprocesseur puis le modèle sur (X, y)."""
        X_t = self.preprocessor.fit_transform(X)
        self.model.fit(X_t, y)
        return self

    def predict(self, X):
        """Prétraite X puis prédit avec le modèle enveloppé."""
        return self.model.predict(self.preprocessor.transform(X))

    def __repr__(self):
        return f"PreprocessedModel({self.model!r})"
