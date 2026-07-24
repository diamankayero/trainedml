"""
Confusion matrix visualization for trainedml.

This module provides a single function to plot the confusion matrix of a
classifier from true and predicted labels.

Examples
--------
>>> from trainedml.viz.confusion import plot_confusion_matrix
>>> fig = plot_confusion_matrix(y_test, y_pred)
>>> fig.show()
"""

from __future__ import annotations

from typing import Any, List, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_true: Any, y_pred: Any, labels: Optional[List[Any]] = None,
                          normalize: Optional[str] = None, cmap: str = "Blues") -> Figure:
    """
    Plot the confusion matrix for a classification task.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    labels : list, optional
        Class labels to display, in order. Defaults to the sorted labels
        found in ``y_true``/``y_pred``.
    normalize : {'true', 'pred', 'all'}, optional
        Normalize the counts over the true labels (rows), the predicted
        labels (columns), or the whole matrix. ``None`` (default) shows raw
        counts.
    cmap : str, default='Blues'
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.figure.Figure
        The generated confusion matrix figure.

    Examples
    --------
    >>> fig = plot_confusion_matrix(y_test, y_pred, normalize='true')
    >>> fig.show()
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, cmap=cmap, colorbar=False, values_format=".2f" if normalize else "d")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig
