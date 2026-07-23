r"""
Logistic Regression classifier for trainedml.

This module implements a supervised classification model using logistic regression (scikit-learn).
It exposes a consistent API (fit, predict, evaluate) as required by the trainedml framework.

Mathematical Formulation
------------------------
The logistic regression model estimates the probability :math:`P(y=1|\mathbf{x})` as:

.. math::
    P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x} - b}}

where :math:`\sigma` is the sigmoid function, :math:`\mathbf{w}` the weights, :math:`b` the bias.

Examples
--------
>>> model = LogisticModel(max_iter=200)
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
>>> acc = model.evaluate(X_test, y_test)
"""

from .base import BaseModel
from sklearn.linear_model import LogisticRegression


class LogisticModel(BaseModel):
    r"""
    Logistic Regression classification model.

    This class wraps scikit-learn's LogisticRegression and exposes a unified API.

    Parameters
    ----------
    max_iter : int, default=200
        Maximum number of iterations for the solver.
    **kwargs :
        Additional keyword arguments passed to LogisticRegression.

    Attributes
    ----------
    model : LogisticRegression
        The underlying scikit-learn estimator.

    Examples
    --------
    >>> model = LogisticModel(max_iter=200)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> acc = model.evaluate(X, y)
    """
    def __init__(self, max_iter=200, **kwargs):
        super().__init__()
        self.model = LogisticRegression(max_iter=max_iter, **kwargs)

    def fit(self, X, y):
        """
        Fit the logistic regression model on training data.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the class for new data.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        array-like
            Predicted class labels.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Return the accuracy of the model on test data.

        Parameters
        ----------
        X : array-like
            Test data.
        y : array-like
            True labels.

        Returns
        -------
        float
            Accuracy score.
        """
        return self.model.score(X, y)
