"""
Automatic data preprocessing for trainedml.

This module provides a standard preprocessor built with scikit-learn:

- **numeric columns**: median imputation then standardization
  (mean 0, std 1) - essential for KNN, logistic regression,
  Ridge/Lasso...;
- **categorical columns**: mode imputation then one-hot encoding
  (categories unseen at prediction time are ignored).

It is used by default by :class:`trainedml.Trainer` (``preprocess=True``)
and by :func:`trainedml.compare`, but can also be used on its own.

Example
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
    """Create a dense OneHotEncoder, compatible with every scikit-learn version."""
    try:
        # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """
    Build trainedml's standard preprocessor.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Untrained transformer: median imputation + standardization for
        numeric columns, mode imputation + one-hot for categorical
        columns. To be fit on the training data only (``fit_transform``),
        then applied to the test data (``transform``).

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
    Wraps a model with trainedml's standard preprocessor.

    On every ``fit``, the preprocessor is (re)trained on the training data
    only, then applied to the prediction data: no information leaks from
    test to train, including during cross-validation.

    Parameters
    ----------
    model : object
        Model to wrap (trainedml or scikit-learn, any fit/predict object).

    Attributes
    ----------
    model : object
        The wrapped model.
    preprocessor : sklearn.compose.ColumnTransformer
        The preprocessor, retrained on every call to :meth:`fit`.

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
        """Task type of the wrapped model ('classification' or 'regression')."""
        return getattr(self.model, "task", None)

    def fit(self, X, y):
        """Fit the preprocessor then the model on (X, y)."""
        X_t = self.preprocessor.fit_transform(X)
        self.model.fit(X_t, y)
        return self

    def predict(self, X):
        """Preprocess X then predict with the wrapped model."""
        return self.model.predict(self.preprocessor.transform(X))

    def __repr__(self):
        return f"PreprocessedModel({self.model!r})"
