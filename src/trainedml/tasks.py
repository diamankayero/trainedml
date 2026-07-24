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

import warnings
from typing import Any, Dict, Optional

import pandas as pd

#: Maximum number of unique integer values for a target to be considered
#: categorical (classification).
MAX_UNIQUE_FOR_CLASSIFICATION = 20

#: Default majority/minority count ratio above which classes are flagged
#: as imbalanced by :func:`check_class_imbalance`.
DEFAULT_IMBALANCE_THRESHOLD = 3.0


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


def check_class_imbalance(y, threshold: float = DEFAULT_IMBALANCE_THRESHOLD) -> Optional[Dict[str, Any]]:
    """
    Check whether classification classes are notably imbalanced.

    Parameters
    ----------
    y : pandas.Series or array-like
        Class labels.
    threshold : float, default=3.0
        Majority/minority count ratio above which classes are considered
        imbalanced.

    Returns
    -------
    dict or None
        ``None`` if the classes are balanced (or if there are fewer than
        two classes). Otherwise, a dict with ``ratio``, ``majority_class``,
        ``majority_count``, ``minority_class``, ``minority_count``.

    Examples
    --------
    >>> check_class_imbalance(pd.Series(["a"] * 90 + ["b"] * 10))
    {'ratio': 9.0, 'majority_class': 'a', 'majority_count': 90, ...}
    >>> check_class_imbalance(pd.Series(["a"] * 55 + ["b"] * 45)) is None
    True
    """
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return None
    ratio = counts.iloc[0] / counts.iloc[-1]
    if ratio < threshold:
        return None
    return {
        "ratio": float(ratio),
        "majority_class": counts.index[0],
        "majority_count": int(counts.iloc[0]),
        "minority_class": counts.index[-1],
        "minority_count": int(counts.iloc[-1]),
    }


def warn_if_imbalanced(y, threshold: float = DEFAULT_IMBALANCE_THRESHOLD) -> None:
    """
    Emit a ``UserWarning`` if the classes in ``y`` are notably imbalanced.

    A no-op if the classes are balanced. Used internally by :class:`Trainer`
    (on fit) and :func:`compare` so imbalance is flagged where it matters,
    without changing any model's behavior.

    Parameters
    ----------
    y : pandas.Series or array-like
        Class labels.
    threshold : float, default=3.0
        Majority/minority count ratio above which classes are considered
        imbalanced.
    """
    info = check_class_imbalance(y, threshold=threshold)
    if info is None:
        return
    warnings.warn(
        f"Imbalanced classes: {info['majority_class']!r} has {info['majority_count']} "
        f"samples versus {info['minority_count']} for {info['minority_class']!r} "
        f"(ratio {info['ratio']:.1f}:1). Accuracy alone can be misleading here: check "
        f"precision/recall per class, and consider model_params={{'class_weight': 'balanced'}} "
        f"for models that support it (random_forest, logistic).",
        UserWarning,
        stacklevel=3,
    )
