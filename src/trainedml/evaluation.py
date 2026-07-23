r"""
Evaluation utilities for classification and regression models in trainedml.

This module provides standard metrics for evaluating classification models (accuracy,
precision, recall, F1-score) and regression models (R², MSE, RMSE, MAE), using scikit-learn.

Mathematical Formulation
------------------------
**Classification:**

Let $y_i$ be the true label and $\hat{y}_i$ the predicted label for sample $i$.

- **Accuracy**:

  .. math::
      \mathrm{accuracy} = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(y_i = \hat{y}_i)

- **Precision** (for class $k$):

  .. math::
      \mathrm{precision}_k = \frac{TP_k}{TP_k + FP_k}

- **Recall** (for class $k$):

  .. math::
      \mathrm{recall}_k = \frac{TP_k}{TP_k + FN_k}

- **F1-score** (for class $k$):

  .. math::
      \mathrm{F1}_k = 2 \cdot \frac{\mathrm{precision}_k \cdot \mathrm{recall}_k}{\mathrm{precision}_k + \mathrm{recall}_k}

**Regression:**

- **Mean Squared Error (MSE)**:

  .. math::
      \mathrm{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2

- **Root Mean Squared Error (RMSE)**:

  .. math::
      \mathrm{RMSE} = \sqrt{\mathrm{MSE}}

- **Mean Absolute Error (MAE)**:

  .. math::
      \mathrm{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|

- **R² (coefficient of determination)**:

  .. math::
      R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}

Examples
--------
>>> from trainedml.evaluation import Evaluator
>>> y_true = [0, 1, 1, 0]
>>> y_pred = [0, 1, 0, 0]
>>> scores = Evaluator.evaluate_all(y_true, y_pred)
>>> print(scores)

>>> y_true_reg = [3.0, -0.5, 2.0, 7.0]
>>> y_pred_reg = [2.5, 0.0, 2.0, 8.0]
>>> scores_reg = Evaluator.evaluate_regression(y_true_reg, y_pred_reg)
>>> print(scores_reg)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


class Evaluator:
    r"""
    Utility class for evaluating classification and regression models.

    Provides static methods to compute standard classification and regression metrics.
    """

    @staticmethod
    def evaluate_all(y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Compute accuracy, precision, recall, and F1-score for classification.

        Parameters
        ----------
        y_true : array-like
            True class labels.
        y_pred : array-like
            Predicted class labels.

        Returns
        -------
        dict
            Dictionary with keys 'accuracy', 'precision', 'recall', 'f1'.

        Examples
        --------
        >>> Evaluator.evaluate_all([0, 1, 1], [0, 1, 0])
        {'accuracy': 0.666..., 'precision': 0.666..., 'recall': 0.666..., 'f1': 0.666...}
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    @staticmethod
    def evaluate_regression(y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Compute R², MSE, RMSE, and MAE for regression.

        Parameters
        ----------
        y_true : array-like
            True continuous values.
        y_pred : array-like
            Predicted continuous values.

        Returns
        -------
        dict
            Dictionary with keys 'r2', 'mse', 'rmse', 'mae'.

        Examples
        --------
        >>> Evaluator.evaluate_regression([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0])
        {'r2': 0.948..., 'mse': 0.375, 'rmse': 0.612..., 'mae': 0.5}
        """
        mse = mean_squared_error(y_true, y_pred)
        return {
            'r2': r2_score(y_true, y_pred),
            'mse': mse,
            'rmse': float(np.sqrt(mse)),
            'mae': mean_absolute_error(y_true, y_pred),
        }

    @staticmethod
    def evaluate_for(task: Optional[str], y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Compute the metrics appropriate for a given task type.

        Parameters
        ----------
        task : str or None
            ``'classification'``, ``'regression'``, or None (auto-detect from y_true).
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.

        Returns
        -------
        dict
            Classification metrics (accuracy, precision, recall, f1) or
            regression metrics (r2, mse, rmse, mae).

        Examples
        --------
        >>> Evaluator.evaluate_for('regression', [3.0, 2.5], [3.1, 2.4])
        {'r2': ..., 'mse': ..., 'rmse': ..., 'mae': ...}
        """
        if task == 'regression':
            return Evaluator.evaluate_regression(y_true, y_pred)
        if task == 'classification':
            return Evaluator.evaluate_all(y_true, y_pred)
        return Evaluator.auto_evaluate(y_true, y_pred)

    @staticmethod
    def auto_evaluate(y_true: Any, y_pred: Any) -> Dict[str, float]:
        """
        Automatically detect classification vs regression and compute the appropriate metrics.

        Parameters
        ----------
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.

        Returns
        -------
        dict
            Classification or regression metrics (auto-detected).

        Examples
        --------
        >>> Evaluator.auto_evaluate([0, 1, 1], [0, 1, 0])  # classification
        {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
        >>> Evaluator.auto_evaluate([3.0, 2.5, 7.1], [3.1, 2.4, 7.0])  # regression
        {'r2': ..., 'mse': ..., 'rmse': ..., 'mae': ...}
        """
        y_true = np.asarray(y_true)
        # Heuristic: if dtype is object/str or few unique values -> classification
        if y_true.dtype.kind in ('U', 'S', 'O'):
            return Evaluator.evaluate_all(y_true, y_pred)
        if len(np.unique(y_true)) <= 20 and np.issubdtype(y_true.dtype, np.integer):
            return Evaluator.evaluate_all(y_true, y_pred)
        return Evaluator.evaluate_regression(y_true, y_pred)
