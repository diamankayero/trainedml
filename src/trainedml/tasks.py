"""
Task type detection (classification vs regression) for trainedml.

This module centralizes the heuristic used by the Trainer, the CLI, the
Benchmark, and :func:`trainedml.compare` to determine whether a target
corresponds to a classification or a regression task.

Heuristic
---------
- Non-numeric target (text, categorical, ``StringDtype``...) -> classification
- Integer target with few unique values (<= 20) -> classification
- Otherwise -> regression

Example
-------
>>> from trainedml.tasks import detect_task
>>> detect_task(pd.Series(["setosa", "versicolor"]))
'classification'
>>> detect_task(pd.Series([1.5, 2.3, 4.8]))
'regression'
"""

from __future__ import annotations

import pandas as pd

#: Maximum number of unique integer values for a target to be considered
#: categorical (classification).
MAX_UNIQUE_FOR_CLASSIFICATION = 20


def is_classification_target(y) -> bool:
    """
    Determine whether the target is categorical (classification) or numeric (regression).

    Parameters
    ----------
    y : pandas.Series or array-like
        Target column to analyze.

    Returns
    -------
    bool
        True if classification, False if regression.

    Examples
    --------
    >>> is_classification_target(pd.Series(['cat', 'dog']))
    True
    >>> is_classification_target(pd.Series([0.1, 2.7, 3.14]))
    False
    """
    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    # If it's not numeric (text, categorical, StringDtype...), it's classification
    if not pd.api.types.is_numeric_dtype(y):
        return True
    # If few unique values and integers, likely classification
    if y.nunique() <= MAX_UNIQUE_FOR_CLASSIFICATION and pd.api.types.is_integer_dtype(y):
        return True
    return False


def detect_task(y) -> str:
    """
    Return the task type associated with a target.

    Parameters
    ----------
    y : pandas.Series or array-like
        Target column to analyze.

    Returns
    -------
    str
        ``'classification'`` or ``'regression'``.

    Examples
    --------
    >>> detect_task(pd.Series([0, 1, 2, 0, 1]))
    'classification'
    """
    return "classification" if is_classification_target(y) else "regression"


def detect_model_task(model, y=None) -> str:
    """
    Determine the task type of a model, whatever it is.

    Priority order:

    1. the model's ``task`` attribute (trainedml models, :class:`BaseModel`);
    2. scikit-learn's ``is_classifier`` / ``is_regressor`` functions
       (arbitrary sklearn estimators);
    3. the heuristic on the ``y`` target, if provided.

    Parameters
    ----------
    model : object
        trainedml model, scikit-learn estimator, or any fit/predict object.
    y : array-like, optional
        Target, used as a last resort for the heuristic.

    Returns
    -------
    str
        ``'classification'`` or ``'regression'``.
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
