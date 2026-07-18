
"""
Base classes for all machine learning models in trainedml.

This module defines the abstract interfaces for supervised models (classification and regression)
to ensure a consistent API across all estimators in the package.

Main features
-------------
- Abstract base class for all ML models (fit, predict, evaluate)
- Distinction between classification and regression via the `task` attribute
- Designed for extension by all concrete model classes (KNN, Logistic, RandomForest, etc.)

Examples
--------
>>> class MyModel(BaseModel):
...     def fit(self, X, y): ...
...     def predict(self, X): ...
...     def evaluate(self, X, y): ...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union



class BaseModel(ABC):
    r"""
    Abstract base class for all supervised machine learning models in trainedml.

    This class defines the standard interface for all models (fit, predict, evaluate)
    and ensures a consistent API for all estimators in the package.

    Features
    --------
    - Abstract interface for supervised models (classification and regression)
    - Enforces implementation of fit, predict, evaluate
    - Stores the underlying estimator (e.g., scikit-learn model) in self.model
    - The 'task' attribute distinguishes between classification and regression

    Attributes
    ----------
    model : object
        The underlying estimator (e.g., scikit-learn model).
    task : str
        Type of task ('classification' or 'regression').

    Methods
    -------
    fit(X, y)
        Train the model on data X and target y.
    predict(X)
        Predict the target for new data X.
    evaluate(X, y)
        Evaluate the model on test data and return a performance metric.

    Examples
    --------
    # Minimal custom model
    >>> class DummyModel(BaseModel):
    ...     def fit(self, X, y): pass
    ...     def predict(self, X): return [0]*len(X)
    ...     def evaluate(self, X, y): return 0.0

    # Using a scikit-learn estimator
    >>> from sklearn.linear_model import LogisticRegression
    >>> class MyLogistic(BaseModel):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.model = LogisticRegression()
    ...     def fit(self, X, y): self.model.fit(X, y)
    ...     def predict(self, X): return self.model.predict(X)
    ...     def evaluate(self, X, y):
    ...         from sklearn.metrics import accuracy_score
    ...         return accuracy_score(y, self.model.predict(X))

    # Usage
    >>> model = MyLogistic()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> acc = model.evaluate(X_test, y_test)
    >>> print(acc)

    Notes
    -----
    - All models in trainedml must inherit from BaseModel or BaseRegressor.
    - The 'task' attribute is used for automatic model selection and benchmarking.
    - The evaluate method should return a scalar (score) or a dict of metrics.
    """
    task: str = 'classification'  # Default: classification

    def __init__(self) -> None:
        self.model: Any = None  # Underlying estimator (e.g., scikit-learn)

    @abstractmethod
    def fit(self, X: Any, y: Any) -> Any:
        """
        Train the model on data X and target y.

        This method must be implemented by all subclasses. It should fit the underlying
        estimator (e.g., scikit-learn model) to the provided data.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Training data (features).
        y : array-like or pandas.Series
            Target values.

        Examples
        --------
        >>> model.fit(X_train, y_train)
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Predict the target for new data X.

        This method must be implemented by all subclasses. It should use the trained
        estimator to predict the target for the provided data.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Input data (features).

        Returns
        -------
        array-like
            Predicted values.

        Examples
        --------
        >>> y_pred = model.predict(X_test)
        >>> print(y_pred)
        """
        pass

    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> Union[float, Dict[str, float]]:
        """
        Evaluate the model on test data and return a performance metric.

        This method must be implemented by all subclasses. It should compute a score
        (e.g., accuracy, R^2, F1, etc.) or a dictionary of metrics on the provided data.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Test data (features).
        y : array-like or pandas.Series
            True target values.

        Returns
        -------
        float or dict
            Performance metric(s) (e.g., accuracy, R^2, etc.).

        Examples
        --------
        >>> score = model.evaluate(X_test, y_test)
        >>> print(score)
        """
        pass



class BaseRegressor(BaseModel):
    r"""
    Abstract base class for regression models in trainedml.

    Inherits from BaseModel and sets task='regression'.
    All regression models should inherit from this class.

    Examples
    --------
    >>> class DummyRegressor(BaseRegressor):
    ...     def fit(self, X, y): pass
    ...     def predict(self, X): return [y.mean()] * len(X)
    ...     def evaluate(self, X, y): return 0.0
    """
    task: str = 'regression'
