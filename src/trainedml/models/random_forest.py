"""
Random Forest classifier for trainedml.

This module implements a supervised classification model using the Random Forest algorithm (scikit-learn).
It exposes a consistent API (fit, predict, evaluate) as required by the trainedml framework.

Mathematical Formulation
------------------------
A random forest is an ensemble of :math:`M` decision trees :math:`\{T_1, ..., T_M\}`, each trained on a bootstrap sample of the data.
The prediction for a new sample :math:`\mathbf{x}` is given by majority vote:

.. math::
    \hat{y} = \mathrm{mode}\left( T_1(\mathbf{x}), ..., T_M(\mathbf{x}) \right)

Examples
--------
>>> model = RandomForestModel(n_estimators=100)
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
>>> acc = model.evaluate(X_test, y_test)
"""

from .base import BaseModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(BaseModel):
    r"""
    Random Forest classification model.

    This class wraps scikit-learn's RandomForestClassifier and exposes a unified API.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    **kwargs :
        Additional keyword arguments passed to RandomForestClassifier.

    Attributes
    ----------
    model : RandomForestClassifier
        The underlying scikit-learn estimator.

    Examples
    --------
    >>> model = RandomForestModel(n_estimators=100)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> acc = model.evaluate(X, y)
    """
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)

    def fit(self, X, y):
        """
        Fit the random forest model on training data.

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
