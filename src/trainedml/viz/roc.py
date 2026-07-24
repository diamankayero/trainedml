"""
ROC curve visualization for trainedml.

This module provides a single function to plot ROC curves (and their AUC)
for binary or multiclass classification, from true labels and predicted
scores (probabilities or decision function values).

Examples
--------
>>> from trainedml.viz.roc import plot_roc_curve
>>> fig = plot_roc_curve(y_test, y_proba)
>>> fig.show()
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def plot_roc_curve(y_true: Any, y_score: Any, class_names: Optional[List[Any]] = None) -> Figure:
    """
    Plot the ROC curve (Receiver Operating Characteristic) for binary or
    multiclass classification.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_score : array-like
        Predicted scores. For binary classification, a 1-D array of scores
        for the positive class (e.g. ``predict_proba(X)[:, 1]``). For
        multiclass, a 2-D array of shape ``(n_samples, n_classes)`` (e.g.
        the full output of ``predict_proba``): one curve is drawn per class
        using a one-vs-rest strategy.
    class_names : list, optional
        Class names for the legend (multiclass only). Defaults to the
        sorted unique values of ``y_true``.

    Returns
    -------
    matplotlib.figure.Figure
        The generated ROC curve figure.

    Notes
    -----
    ROC AUC quantifies how well the model ranks positive examples above
    negative ones, independently of any decision threshold: 0.5 is
    equivalent to random guessing, 1.0 is a perfect ranking.

    Examples
    --------
    Binary classification:

    >>> proba = trainer.model.model.predict_proba(X_test)[:, 1]
    >>> fig = plot_roc_curve(y_test, proba)

    Multiclass (one-vs-rest):

    >>> proba = trainer.model.model.predict_proba(X_test)
    >>> fig = plot_roc_curve(y_test, proba)
    """
    y_score = np.asarray(y_score)
    fig, ax = plt.subplots(figsize=(6.5, 6))

    if y_score.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    else:
        classes = class_names if class_names is not None else sorted(set(y_true))
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] != y_score.shape[1]:
            raise ValueError(
                f"y_score has {y_score.shape[1]} columns but {y_bin.shape[1]} classes "
                f"were found in y_true; pass class_names to disambiguate the order."
            )
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance level (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
