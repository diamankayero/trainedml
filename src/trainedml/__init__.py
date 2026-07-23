"""
Main module of the trainedml package.

This module exposes the central `Trainer` class, which handles the whole
machine learning workflow: data loading, train/test splitting,
preprocessing, training, evaluation, prediction, and persistence. It also
exposes the :func:`compare` function to compare every model for a task in
one line.

Main features
-------------
- High-level API to train, evaluate, and predict with an ML model
- Supports built-in datasets (Iris, Wine) or remote CSV files
- Automatic preprocessing (imputation, standardization, one-hot encoding)
- Automatic train/test split, seed can vary without recreating the object
- Handles trainedml models (KNN, Logistic, Random Forest, regressors...)
  and **any scikit-learn estimator** (or any fit/predict object)
- Task-aware evaluation: classification (accuracy, precision, recall, f1)
  or regression (r2, mse, rmse, mae)
- Save and reload a trained model (:meth:`Trainer.save` / :meth:`Trainer.load`)
- Usable as a script, an API, a CLI, or a webapp backend

Example
-------
>>> from trainedml import Trainer, compare
>>> trainer = Trainer(dataset="iris", model="knn", model_params={"n_neighbors": 5})
>>> trainer.fit()
>>> print(trainer.evaluate())
>>> preds = trainer.predict([[5.1, 3.5, 1.4, 0.2]])
>>> compare(dataset="iris", cv=5)  # comparison of every model as a DataFrame
"""

from __future__ import annotations

__version__ = "0.2.0"

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd

from .data.loader import DataLoader
from .models import (
    KNNModel, LogisticModel, RandomForestModel,
    MODEL_MAP, CLASSIFIER_MAP, REGRESSOR_MAP, get_model,
)
from .evaluation import Evaluator
from .tasks import detect_task, detect_model_task
from .preprocessing import build_preprocessor, PreprocessedModel
from .benchmark import Benchmark
from .compare import compare
from .visualization import Visualizer

__all__ = [
    "Trainer", "DataLoader", "Evaluator", "Benchmark", "Visualizer",
    "compare", "get_model", "detect_task",
    "MODEL_MAP", "CLASSIFIER_MAP", "REGRESSOR_MAP",
    "KNNModel", "LogisticModel", "RandomForestModel",
    "build_preprocessor", "PreprocessedModel",
]


class Trainer:
    r"""
    High-level class to train, evaluate, and predict with a machine
    learning model.

    This class centralizes the whole ML workflow: data loading, train/test
    split, preprocessing, training, evaluation, prediction, and
    persistence. It is designed to be used in an API, a webapp, or a plain
    Python script.

    Parameters
    ----------
    dataset : str, optional
        Name of a built-in dataset ("iris", "wine").
    model : str or object, default='random_forest'
        Name of a trainedml model ("random_forest", "knn", "logistic",
        "linear", "ridge", "lasso", ...) **or** any estimator that has
        ``fit`` and ``predict`` methods (for example a scikit-learn
        estimator).
    url : str, optional
        URL of a remote CSV to load.
    target : str, optional
        Name of the target column (if url).
    X : pandas.DataFrame or array-like, optional
        Features provided directly in memory (alternative to dataset/url).
    y : pandas.Series or array-like, optional
        Matching target (required if X is provided).
    test_size : float, default=0.2
        Test proportion (between 0 and 1).
    seed : int, default=42
        Random seed for reproducibility.
    model_params : dict, optional
        Hyperparameters passed to the model's constructor (only usable
        with a model name; for an already-instantiated estimator, configure
        it directly).
    preprocess : bool, default=True
        If True, applies trainedml's standard preprocessing (imputation,
        standardization of numeric columns, one-hot encoding of categorical
        columns). The preprocessor is fit on the training set only.

    Attributes
    ----------
    model : object
        The underlying ML model instance.
    preprocessor : sklearn.compose.ColumnTransformer or None
        The preprocessor (None if ``preprocess=False``).
    X_train, X_test, y_train, y_test : array-like
        Train/test splits (not preprocessed).
    is_fitted : bool
        Whether the model has been trained.

    Examples
    --------
    Standard workflow:

    >>> trainer = Trainer(dataset="iris", model="knn")
    >>> trainer.fit()
    >>> print(trainer.evaluate())
    >>> preds = trainer.predict([[5.1, 3.5, 1.4, 0.2]])

    Hyperparameters and an arbitrary scikit-learn estimator:

    >>> trainer = Trainer(dataset="wine", model="knn", model_params={"n_neighbors": 7})
    >>> from sklearn.svm import SVC
    >>> trainer = Trainer(dataset="iris", model=SVC(kernel="rbf"))

    Varying the seed without recreating the Trainer:

    >>> for s in range(5):
    ...     print(trainer.fit(seed=s).evaluate())

    Save and reload:

    >>> trainer.save("model.joblib")
    >>> restored = Trainer.load("model.joblib")
    >>> restored.predict([[5.1, 3.5, 1.4, 0.2]])
    """

    def __init__(self, dataset: Optional[str] = None, model: Union[str, Any] = 'random_forest',
                 url: Optional[str] = None, target: Optional[str] = None,
                 X: Optional["pd.DataFrame"] = None, y: Optional["pd.Series"] = None,
                 test_size: float = 0.2, seed: int = 42,
                 model_params: Optional[Dict[str, Any]] = None, preprocess: bool = True) -> None:
        self.dataset = dataset
        self.url = url
        self.target = target
        self._X, self._y = X, y
        self.test_size = test_size
        self.seed = seed
        self.preprocess = preprocess
        self.model_params = dict(model_params or {})

        if isinstance(model, str):
            if model not in MODEL_MAP:
                raise ValueError(
                    f"Unknown model: {model!r}. Available: {list(MODEL_MAP.keys())} "
                    f"(or pass a fit/predict estimator directly)."
                )
            self.model_name = model
            self.model = MODEL_MAP[model](**self.model_params)
        else:
            if not (hasattr(model, "fit") and hasattr(model, "predict")):
                raise TypeError(
                    "`model` must be a trainedml model name or an object "
                    "with fit and predict methods (e.g. a scikit-learn estimator)."
                )
            if self.model_params:
                raise ValueError(
                    "`model_params` can only be used with a model name; "
                    "configure your estimator directly instead."
                )
            self.model_name = type(model).__name__
            self.model = model

        self.preprocessor = build_preprocessor() if preprocess else None
        self.feature_names_: Optional[List[str]] = None
        self.X_train: Any = None
        self.X_test: Any = None
        self.y_train: Any = None
        self.y_test: Any = None
        self.is_fitted = False
        self._task: Optional[str] = None

    @property
    def task(self) -> str:
        """Task type of the model ('classification' or 'regression')."""
        if self._task is not None:
            return self._task
        return detect_model_task(self.model, self.y_train)

    def load_data(self, test_size: Optional[float] = None,
                  seed: Optional[int] = None) -> Tuple[Any, Any, Any, Any]:
        """
        Load the data, perform the train/test split, and store it on the object.

        The split uses the package's own API (:meth:`DataLoader.split`):
        there is never a need to call scikit-learn directly. The resulting
        sets are available via the ``X_train``, ``X_test``, ``y_train``,
        ``y_test`` attributes.

        Parameters
        ----------
        test_size : float, optional
            New test proportion. If provided, overrides the constructor's value.
        seed : int, optional
            New random seed. If provided, overrides the constructor's value,
            which lets you vary the split without recreating the Trainer.

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)

        Raises
        ------
        ValueError
            If the dataset or target is not specified correctly.

        Examples
        --------
        >>> trainer = Trainer(dataset="iris", model="knn")
        >>> X_train, X_test, y_train, y_test = trainer.load_data(seed=7)
        """
        if test_size is not None:
            self.test_size = test_size
        if seed is not None:
            self.seed = seed
        loader = DataLoader()
        if self._X is not None and self._y is not None:
            # Data provided directly in memory (X=..., y=...)
            X, y = self._X, self._y
        else:
            X, y = loader.load_dataset(name=self.dataset, url=self.url, target=self.target)
        self.X_train, self.X_test, self.y_train, self.y_test = loader.split(
            X, y, test_size=self.test_size, random_state=self.seed)
        # A new split invalidates any previous training
        self.is_fitted = False
        return self.X_train, self.X_test, self.y_train, self.y_test

    def _transform(self, X: Any, fit: bool = False) -> Any:
        """Apply the preprocessor (fit_transform on train, transform otherwise)."""
        if self.preprocessor is None:
            return X
        return self.preprocessor.fit_transform(X) if fit else self.preprocessor.transform(X)

    def fit(self, test_size: Optional[float] = None, seed: Optional[int] = None) -> "Trainer":
        """
        Fit the preprocessor (if enabled) then the model on the training data.
        Loads the data first if needed.

        Parameters
        ----------
        test_size : float, optional
            If provided, the data is re-split with this proportion before training.
        seed : int, optional
            If provided, the data is re-split with this seed before training.
            Useful to check a model's stability across splits:
            ``for s in range(5): print(trainer.fit(seed=s).evaluate())``

        Returns
        -------
        self : Trainer
            The current instance (for chaining).
        """
        if self.X_train is None or test_size is not None or seed is not None:
            self.load_data(test_size=test_size, seed=seed)
        if hasattr(self.X_train, "columns"):
            self.feature_names_ = list(self.X_train.columns)
        X_t = self._transform(self.X_train, fit=True)
        self.model.fit(X_t, self.y_train)
        self._task = detect_model_task(self.model, self.y_train)
        self.is_fitted = True
        return self

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the trained model on the test data, with metrics adapted
        to the task: classification (accuracy, precision, recall, f1) or
        regression (r2, mse, rmse, mae).

        Returns
        -------
        dict
            Dictionary of scores.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be trained before evaluation.")
        y_pred = self.model.predict(self._transform(self.X_test))
        return Evaluator.evaluate_for(self.task, self.y_test, y_pred)

    def _as_frame(self, X: Any) -> "pd.DataFrame":
        """Convert X to a DataFrame using the training columns when possible."""
        import numpy as np
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.feature_names_ is not None and X.shape[1] == len(self.feature_names_):
            return pd.DataFrame(X, columns=self.feature_names_)
        return pd.DataFrame(X)

    def predict(self, X: Any) -> "np.ndarray":
        """
        Predict the target for new data X.

        The preprocessing learned during training is applied automatically:
        X must contain the same features (same columns) as the training data.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Input data (same features as training).

        Returns
        -------
        array
            Model predictions.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be trained before prediction.")
        X = self._as_frame(X)
        return self.model.predict(self._transform(X))

    def save(self, path: Union[str, "Path"]) -> None:
        """
        Save the trained model (and its preprocessor) to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path (convention: ``.joblib`` extension).

        Raises
        ------
        RuntimeError
            If the model has not been trained.

        Examples
        --------
        >>> trainer.fit().save("model.joblib")
        """
        if not self.is_fitted:
            raise RuntimeError("The model must be trained before saving.")
        import joblib
        payload = {
            "trainedml_version": __version__,
            "model": self.model,
            "model_name": self.model_name,
            "model_params": self.model_params,
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names_,
            "task": self.task,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Union[str, "Path"]) -> "Trainer":
        """
        Reload a Trainer saved with :meth:`save`, ready to predict.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the saved file.

        Returns
        -------
        Trainer
            Instance ready for :meth:`predict` (the original data is not
            reloaded; ``X_train`` etc. are None).

        Examples
        --------
        >>> restored = Trainer.load("model.joblib")
        >>> restored.predict([[5.1, 3.5, 1.4, 0.2]])
        """
        import joblib
        payload = joblib.load(path)
        trainer = cls.__new__(cls)
        trainer.dataset = None
        trainer.url = None
        trainer.target = None
        # Default values: the original data is not reloaded
        trainer.test_size = 0.2
        trainer.seed = 42
        trainer.preprocess = payload["preprocessor"] is not None
        trainer.model = payload["model"]
        trainer.model_name = payload["model_name"]
        trainer.model_params = payload.get("model_params", {})
        trainer.preprocessor = payload["preprocessor"]
        trainer.feature_names_ = payload.get("feature_names")
        trainer.X_train = trainer.X_test = trainer.y_train = trainer.y_test = None
        trainer._task = payload.get("task")
        trainer.is_fitted = True
        return trainer


def main() -> None:
    """
    CLI entry point of the trainedml package.
    Launches the command-line interface (see src/trainedml/cli.py).
    """
    from .cli import main as cli_main
    cli_main()
