"""
Détection du type de tâche (classification vs régression) pour trainedml.

Ce module centralise l'heuristique utilisée par le Trainer, la CLI, le Benchmark
et :func:`trainedml.compare` pour déterminer si une cible correspond à une tâche
de classification ou de régression.

Heuristique
-----------
- Cible non numérique (texte, catégoriel, ``StringDtype``...) → classification
- Cible entière avec peu de valeurs uniques (≤ 20) → classification
- Sinon → régression

Exemple
-------
>>> from trainedml.tasks import detect_task
>>> detect_task(pd.Series(["setosa", "versicolor"]))
'classification'
>>> detect_task(pd.Series([1.5, 2.3, 4.8]))
'regression'
"""

from __future__ import annotations

import pandas as pd

#: Nombre maximal de valeurs uniques entières pour considérer une cible
#: comme catégorielle (classification).
MAX_UNIQUE_FOR_CLASSIFICATION = 20


def is_classification_target(y) -> bool:
    """
    Détermine si la cible est catégorielle (classification) ou numérique (régression).

    Parameters
    ----------
    y : pandas.Series or array-like
        Colonne cible à analyser.

    Returns
    -------
    bool
        True si classification, False si régression.

    Examples
    --------
    >>> is_classification_target(pd.Series(['cat', 'dog']))
    True
    >>> is_classification_target(pd.Series([0.1, 2.7, 3.14]))
    False
    """
    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    # Si ce n'est pas numérique (texte, catégoriel, StringDtype...), c'est de la classification
    if not pd.api.types.is_numeric_dtype(y):
        return True
    # Si peu de valeurs uniques et entiers, probablement classification
    if y.nunique() <= MAX_UNIQUE_FOR_CLASSIFICATION and pd.api.types.is_integer_dtype(y):
        return True
    return False


def detect_task(y) -> str:
    """
    Retourne le type de tâche associé à une cible.

    Parameters
    ----------
    y : pandas.Series or array-like
        Colonne cible à analyser.

    Returns
    -------
    str
        ``'classification'`` ou ``'regression'``.

    Examples
    --------
    >>> detect_task(pd.Series([0, 1, 2, 0, 1]))
    'classification'
    """
    return "classification" if is_classification_target(y) else "regression"


def detect_model_task(model, y=None) -> str:
    """
    Détermine le type de tâche d'un modèle, quel qu'il soit.

    L'ordre de priorité est :

    1. l'attribut ``task`` du modèle (modèles trainedml, :class:`BaseModel`) ;
    2. les fonctions ``is_classifier`` / ``is_regressor`` de scikit-learn
       (estimateurs sklearn arbitraires) ;
    3. l'heuristique sur la cible ``y`` si elle est fournie.

    Parameters
    ----------
    model : object
        Modèle trainedml, estimateur scikit-learn, ou tout objet fit/predict.
    y : array-like, optional
        Cible, utilisée en dernier recours pour l'heuristique.

    Returns
    -------
    str
        ``'classification'`` ou ``'regression'``.
    """
    task = getattr(model, "task", None)
    if task in ("classification", "regression"):
        return task
    try:
        from sklearn.base import is_classifier, is_regressor
        if is_classifier(model):
            return "classification"
        if is_regressor(model):
            return "regression"
    except Exception:
        pass
    if y is not None:
        return detect_task(y)
    return "classification"
