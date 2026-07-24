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
- Supports built-in datasets (Iris, Wine, Diabetes) or remote CSV files
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

__version__ = "0.3.0"

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure

from .data.loader import DataLoader
from .models import (
    KNNModel, LogisticModel, RandomForestModel,
    MODEL_MAP, CLASSIFIER_MAP, REGRESSOR_MAP, get_model,
)
from .evaluation import Evaluator
from .tasks import detect_task, detect_model_task, check_class_imbalance, warn_if_imbalanced
from .preprocessing import build_preprocessor, PreprocessedModel
from .benchmark import Benchmark
from .compare import compare
from .visualization import Visualizer

__all__ = [
    "Trainer", "DataLoader", "Evaluator", "Benchmark", "Visualizer",
    "compare", "get_model", "detect_task", "check_class_imbalance",
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
        Name of a built-in dataset ("iris", "wine", "diabetes").
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
    best_params_ : dict or None
        Best hyperparameters found by :meth:`grid_search` or
        :meth:`random_search` (``None`` until one of them is called).
    best_score_ : float or None
        Cross-validated score of ``best_params_``.

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
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None

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
        if self._task == "classification":
            warn_if_imbalanced(self.y_train)
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

    @property
    def _estimator(self) -> Any:
        """
        The raw underlying estimator, whichever way ``model`` was built:
        a trainedml wrapper (e.g. ``RandomForestModel``) exposes it via its
        own ``.model`` attribute, while an external estimator passed
        directly (``Trainer(model=SVC())``) already *is* the raw object.
        """
        return getattr(self.model, "model", self.model)

    def _preprocessed_feature_names(self, n_features: int) -> List[str]:
        """Best-effort feature names matching a fitted estimator's input width."""
        if self.preprocessor is not None:
            try:
                names = [name.split("__", 1)[-1] for name in self.preprocessor.get_feature_names_out()]
                if len(names) == n_features:
                    return names
            except Exception:
                pass
        if self.feature_names_ is not None and len(self.feature_names_) == n_features:
            return list(self.feature_names_)
        return [f"feature_{i}" for i in range(n_features)]

    def _permutation_importances(self) -> Tuple["np.ndarray", List[str]]:
        """Model-agnostic fallback for :meth:`feature_importances`, used when
        the estimator has neither ``feature_importances_`` nor ``coef_``."""
        from sklearn.inspection import permutation_importance

        trainer = self

        class _ScoringAdapter:
            """Minimal fit/predict/score duck-type so permutation_importance
            can drive Trainer's own preprocessing + model, on the original
            (un-preprocessed) feature columns."""

            def fit(self, X: Any, y: Any) -> "_ScoringAdapter":
                return self

            def predict(self, X: Any) -> Any:
                return trainer.model.predict(trainer._transform(X))

            def score(self, X: Any, y: Any) -> float:
                from sklearn.metrics import accuracy_score, r2_score
                y_pred = self.predict(X)
                if trainer.task == "classification":
                    return accuracy_score(y, y_pred)
                return r2_score(y, y_pred)

        result = permutation_importance(
            _ScoringAdapter(), self.X_test, self.y_test, n_repeats=10, random_state=self.seed,
        )
        names = self._preprocessed_feature_names(len(result.importances_mean))
        return result.importances_mean, names

    def feature_importances(self) -> "pd.Series":
        """
        Feature importance, sorted from most to least important.

        Uses the estimator's native attribute when available
        (``feature_importances_`` for tree-based models, the mean absolute
        ``coef_`` for linear models), and falls back to permutation
        importance otherwise (works for any model, including KNN and SVM,
        but is slower since it re-scores the model many times on the test
        set).

        Returns
        -------
        pandas.Series
            Importance per feature, indexed by feature name and sorted
            descending.

        Raises
        ------
        RuntimeError
            If the model has not been trained.

        Examples
        --------
        >>> trainer = Trainer(dataset="wine", model="random_forest").fit()
        >>> print(trainer.feature_importances().head())
        """
        import numpy as np
        import pandas as pd

        if not self.is_fitted:
            raise RuntimeError("The model must be trained before computing feature importances.")
        estimator = self._estimator

        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_)
            names = self._preprocessed_feature_names(len(values))
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_)
            values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            names = self._preprocessed_feature_names(len(values))
        else:
            values, names = self._permutation_importances()

        return pd.Series(values, index=names, name="importance").sort_values(ascending=False)

    def plot_feature_importances(self, top_n: Optional[int] = None) -> "Figure":
        """
        Bar chart of :meth:`feature_importances`.

        Parameters
        ----------
        top_n : int, optional
            If given, only the ``top_n`` most important features are shown.

        Returns
        -------
        matplotlib.figure.Figure
            The generated bar chart.

        Examples
        --------
        >>> trainer.plot_feature_importances(top_n=10)
        """
        import matplotlib.pyplot as plt

        importances = self.feature_importances()
        if top_n is not None:
            importances = importances.head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(importances))))
        importances.sort_values().plot(kind="barh", ax=ax, color="#578F00")
        ax.set_xlabel("Importance")
        ax.set_title("Feature importances")
        fig.tight_layout()
        return fig

    def confusion_matrix(self, normalize: Optional[str] = None) -> "Figure":
        """
        Plot the confusion matrix of the trained model on the test set.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, optional
            Normalize counts over true labels (rows), predicted labels
            (columns), or the whole matrix. ``None`` (default) shows raw
            counts.

        Returns
        -------
        matplotlib.figure.Figure
            The generated confusion matrix figure.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        ValueError
            If the task is not classification.

        Examples
        --------
        >>> trainer.confusion_matrix(normalize="true")
        """
        from .viz.confusion import plot_confusion_matrix

        if not self.is_fitted:
            raise RuntimeError("The model must be trained before plotting a confusion matrix.")
        if self.task != "classification":
            raise ValueError("confusion_matrix() is only meaningful for a classification task.")
        y_pred = self.model.predict(self._transform(self.X_test))
        return plot_confusion_matrix(self.y_test, y_pred, normalize=normalize)

    def roc_curve(self) -> "Figure":
        """
        Plot the ROC curve (and AUC) of the trained model on the test set.

        Binary classification draws a single curve; multiclass draws one
        curve per class using a one-vs-rest strategy.

        Returns
        -------
        matplotlib.figure.Figure
            The generated ROC curve figure.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        ValueError
            If the task is not classification.
        AttributeError
            If the estimator exposes neither ``predict_proba`` nor
            ``decision_function`` (needed to rank predictions).

        Examples
        --------
        >>> trainer.roc_curve()
        """
        from .viz.roc import plot_roc_curve

        if not self.is_fitted:
            raise RuntimeError("The model must be trained before plotting a ROC curve.")
        if self.task != "classification":
            raise ValueError("roc_curve() is only meaningful for a classification task.")
        estimator = self._estimator
        X_t = self._transform(self.X_test)
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_t)
            scores = proba[:, 1] if proba.shape[1] == 2 else proba
        elif hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X_t)
        else:
            raise AttributeError(
                f"{type(estimator).__name__} exposes neither predict_proba nor "
                f"decision_function: a ROC curve needs one of them to rank predictions."
            )
        return plot_roc_curve(self.y_test, scores)

    def _run_search(self, search_cls: Any, param_space: Dict[str, List[Any]],
                    cv: int, scoring: Optional[str], **search_kwargs: Any) -> "pd.DataFrame":
        """Shared implementation for :meth:`grid_search` and :meth:`random_search`."""
        import pandas as pd
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline

        if self.X_train is None:
            self.load_data()

        steps = []
        if self.preprocessor is not None:
            steps.append(("preprocessor", clone(self.preprocessor)))
        steps.append(("model", clone(self._estimator)))
        pipeline = Pipeline(steps)

        prefixed_space = {f"model__{key}": value for key, value in param_space.items()}
        search = search_cls(pipeline, prefixed_space, cv=cv, scoring=scoring, **search_kwargs)
        search.fit(self.X_train, self.y_train)

        # Adopt the best pipeline found: re-wrap into a trainedml model if
        # the original `model` was a trainedml wrapper, otherwise assign
        # the raw estimator directly, mirroring `_estimator`'s duality.
        best_estimator = search.best_estimator_.named_steps["model"]
        if hasattr(self.model, "model"):
            self.model.model = best_estimator
        else:
            self.model = best_estimator
        if self.preprocessor is not None:
            self.preprocessor = search.best_estimator_.named_steps["preprocessor"]
        self.model_params = {**self.model_params, **{
            key.split("model__", 1)[-1]: value for key, value in search.best_params_.items()
        }}
        self._task = detect_model_task(self.model, self.y_train)
        if self._task == "classification":
            warn_if_imbalanced(self.y_train)
        self.is_fitted = True

        self.best_params_ = {key.split("model__", 1)[-1]: value for key, value in search.best_params_.items()}
        self.best_score_ = search.best_score_

        results = pd.DataFrame(search.cv_results_)
        results.columns = [
            col.replace("param_model__", "param_") for col in results.columns
        ]
        keep = [c for c in results.columns if c.startswith("param_")] + [
            "mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time",
        ]
        return results[keep].sort_values("rank_test_score").reset_index(drop=True)

    def grid_search(self, param_grid: Dict[str, List[Any]], cv: int = 5,
                    scoring: Optional[str] = None) -> "pd.DataFrame":
        """
        Exhaustively search hyperparameters by cross-validated grid search.

        Every combination of ``param_grid`` is tried; the model is
        re-trained on the full training set with the best combination found
        (and ``self.model_params``, ``self.best_params_``, ``self.best_score_``
        are updated), so the ``Trainer`` is immediately ready for
        :meth:`evaluate` / :meth:`predict` afterward.

        Parameters
        ----------
        param_grid : dict
            ``{hyperparameter_name: [values to try]}``, using the same bare
            parameter names as ``model_params`` (no ``model__`` prefix
            needed).
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, optional
            A scikit-learn scoring name (e.g. ``"f1_macro"``, ``"r2"``).
            Defaults to the estimator's own scorer (accuracy for
            classification, R² for regression).

        Returns
        -------
        pandas.DataFrame
            One row per combination tried, sorted by rank (best first):
            the parameters, mean/std test score, and mean fit time.

        Examples
        --------
        >>> trainer = Trainer(dataset="wine", model="knn")
        >>> results = trainer.grid_search({"n_neighbors": [3, 5, 7, 9]})
        >>> print(trainer.best_params_, trainer.best_score_)
        >>> print(trainer.evaluate())
        """
        from sklearn.model_selection import GridSearchCV
        return self._run_search(GridSearchCV, param_grid, cv, scoring)

    def random_search(self, param_distributions: Dict[str, List[Any]], n_iter: int = 20,
                      cv: int = 5, scoring: Optional[str] = None,
                      seed: Optional[int] = None) -> "pd.DataFrame":
        """
        Search hyperparameters by cross-validated random sampling.

        Draws ``n_iter`` random combinations from ``param_distributions``
        instead of trying every combination: a practical middle ground when
        :meth:`grid_search`'s exhaustive search would be too slow (many
        hyperparameters, or wide ranges). The model is re-trained on the
        full training set with the best combination found, exactly like
        :meth:`grid_search`.

        Parameters
        ----------
        param_distributions : dict
            ``{hyperparameter_name: values or a scipy.stats distribution}``,
            same bare parameter names as ``model_params``.
        n_iter : int, default=20
            Number of random combinations to try.
        cv : int, default=5
            Number of cross-validation folds.
        scoring : str, optional
            A scikit-learn scoring name. Defaults to the estimator's own
            scorer.
        seed : int, optional
            Random seed for sampling combinations. Defaults to the
            ``Trainer``'s own ``seed``.

        Returns
        -------
        pandas.DataFrame
            One row per combination tried, sorted by rank (best first).

        Examples
        --------
        >>> trainer = Trainer(dataset="wine", model="random_forest")
        >>> results = trainer.random_search(
        ...     {"n_estimators": range(50, 500, 25), "max_depth": [None, 3, 5, 10, 20]},
        ...     n_iter=15,
        ... )
        >>> print(trainer.best_params_)
        """
        from sklearn.model_selection import RandomizedSearchCV
        return self._run_search(
            RandomizedSearchCV, param_distributions, cv, scoring,
            n_iter=n_iter, random_state=seed if seed is not None else self.seed,
        )

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
        trainer.best_params_ = None
        trainer.best_score_ = None
        return trainer


def main() -> None:
    """
    CLI entry point of the trainedml package.
    Launches the command-line interface (see src/trainedml/cli.py).
    """
    from .cli import main as cli_main
    cli_main()
