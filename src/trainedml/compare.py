"""
Compare every model in one line for trainedml.

This module exposes :func:`compare`, which chains the whole pipeline:
data loading, task type detection, preprocessing, cross-validation of
every suitable model, and returns a pandas DataFrame sorted from best to
worst model.

Example
-------
>>> from trainedml import compare
>>> df = compare(dataset="wine", cv=5)
>>> print(df)
                     accuracy  accuracy_std  ...  fit_time  predict_time
model                                        ...
random_forest          0.98          0.02    ...    0.081        0.004
...
"""

from __future__ import annotations

from typing import Optional

from .benchmark import Benchmark
from .data.loader import DataLoader
from .models import CLASSIFIER_MAP, REGRESSOR_MAP
from .preprocessing import PreprocessedModel
from .tasks import detect_task, warn_if_imbalanced


def compare(
    dataset: Optional[str] = None,
    url: Optional[str] = None,
    target: Optional[str] = None,
    X=None,
    y=None,
    models: Optional[dict] = None,
    cv: int = 5,
    preprocess: bool = True,
    seed: int = 42,
    show_progress: bool = True,
    sort: bool = True,
):
    """
    Compare every model suited to a dataset and return a sorted DataFrame.

    The task type (classification or regression) is automatically detected
    from the target, and only suitable models are compared. Each model is
    evaluated by cross-validation; preprocessing (imputation,
    standardization, one-hot) is refit on every fold to avoid any
    information leakage.

    Parameters
    ----------
    dataset : str, optional
        Name of a built-in dataset ("iris", "wine", "diabetes").
    url : str, optional
        URL of a remote CSV (requires ``target``).
    target : str, optional
        Name of the target column (if ``url``).
    X : pandas.DataFrame or array-like, optional
        Features provided directly in memory (alternative to dataset/url).
    y : pandas.Series or array-like, optional
        Matching target (required if X is provided).
    models : dict, optional
        Dictionary {name: instance} to compare custom models
        (trainedml or scikit-learn). Defaults to every trainedml model
        suited to the detected task.
    cv : int, default=5
        Number of cross-validation folds.
    preprocess : bool, default=True
        Applies trainedml's standard preprocessing on every fold.
    seed : int, default=42
        Random seed for fold shuffling.
    show_progress : bool, default=True
        Show a progress bar.
    sort : bool, default=True
        Sort the table by the primary metric (accuracy or r2), descending.

    Returns
    -------
    pandas.DataFrame
        One row per model: average metrics, standard deviations
        (``*_std``), average training and prediction times.

    Raises
    ------
    ValueError
        If neither ``dataset`` nor ``url``+``target`` is provided.

    Examples
    --------
    Compare every classifier on Wine:

    >>> from trainedml import compare
    >>> print(compare(dataset="wine"))

    Compare custom models (including scikit-learn):

    >>> from sklearn.svm import SVC
    >>> from trainedml.models import KNNModel
    >>> print(compare(dataset="iris", models={"svc": SVC(), "knn": KNNModel()}))

    On a remote CSV (regression):

    >>> print(compare(url="https://.../winequality-red.csv", target="quality"))
    """
    if X is None or y is None:
        loader = DataLoader()
        X, y = loader.load_dataset(name=dataset, url=url, target=target)

    task = detect_task(y)
    if task == "classification":
        warn_if_imbalanced(y)

    if models is None:
        model_map = CLASSIFIER_MAP if task == "classification" else REGRESSOR_MAP
        models = {name: cls() for name, cls in model_map.items()}

    if preprocess:
        models = {name: PreprocessedModel(model) for name, model in models.items()}

    bench = Benchmark(models)
    bench.run_cv(X, y, cv=cv, show_progress=show_progress, random_state=seed)
    return bench.to_dataframe(sort=sort)
