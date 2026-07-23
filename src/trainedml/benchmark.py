"""
Benchmark utilities for comparing multiple models in trainedml.

This module provides the Benchmark class to compare the performance (accuracy, speed, etc.)
of several models on the same dataset, with optional parallelization and progress bar.

Mathematical Formulation
------------------------
Let $\mathcal{M} = \{M_1, ..., M_K\}$ be a set of models. For each model $M_k$:
- Fit time: $T_{fit}^{(k)}$
- Predict time: $T_{pred}^{(k)}$
- Score: $S^{(k)}$ (e.g., accuracy)

The benchmark returns a dictionary:

.. code-block:: python

    {
        'model_name': {
            'scores': {...},
            'fit_time': ...,
            'predict_time': ...
        },
        ...
    }

Examples
--------
>>> from trainedml.benchmark import Benchmark
>>> models = {'knn': KNNModel(), 'rf': RandomForestModel()}
>>> bench = Benchmark(models)
>>> results = bench.run(X_train, y_train, X_test, y_test)
>>> print(results)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from .evaluation import Evaluator
from .tasks import detect_model_task, detect_task


def _train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    """
    Helper function to train and evaluate a single model (for parallelization).

    Parameters
    ----------
    name : str
        Model name.
    model : object
        Model instance (must implement fit, predict).
    X_train, y_train, X_test, y_test : array-like
        Data splits.

    Returns
    -------
    tuple
        (model name, results dict)
    """
    # Measure training time
    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Measure prediction time
    start_pred = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_pred

    # Metrics adapted to the model's task type (classification or regression)
    task = detect_model_task(model, y_test)
    scores = Evaluator.evaluate_for(task, y_test, y_pred)
    return name, {
        'scores': scores,
        'fit_time': fit_time,
        'predict_time': predict_time
    }


class Benchmark:
    r"""
    Class for comparing the performance of multiple classification/regression models.

    Supports sequential or parallel execution, progress bar, and timing.

    Parameters
    ----------
    models : dict
        Dictionary {name: model_instance}.

    Attributes
    ----------
    models : dict
        Models to benchmark.
    results : dict or None
        Results after running the benchmark.

    Methods
    -------
    run(X_train, y_train, X_test, y_test, parallel=False, n_jobs=-1, show_progress=True)
        Run the benchmark and return results.
    summary()
        Return a formatted summary of the results.
    print_summary()
        Print the summary to stdout.

    Examples
    --------
    >>> bench = Benchmark({'knn': KNNModel(), 'rf': RandomForestModel()})
    >>> results = bench.run(X_train, y_train, X_test, y_test)
    >>> bench.print_summary()
    """
    def __init__(self, models: Dict[str, Any]) -> None:
        """
        Args:
            models (dict): dictionary {name: model_instance}
        """
        self.models = models
        self.results: Optional[Dict[str, Dict[str, Any]]] = None

    def run(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        parallel: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Train and evaluate each model, returning scores and timing.

        Parameters
        ----------
        X_train, y_train, X_test, y_test : array-like
            Data splits.
        parallel : bool, default=False
            If True, run models in parallel.
        n_jobs : int, default=-1
            Number of jobs for parallelization.
        show_progress : bool, default=True
            Show a progress bar.

        Returns
        -------
        dict
            {model_name: {scores, fit_time, predict_time}}
        """
        results = {}

        if parallel:
            # Parallel execution with joblib
            model_items = list(self.models.items())

            if show_progress:
                print(f"Running {len(model_items)} models in parallel...")

            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(_train_and_evaluate)(
                    name, model, X_train, y_train, X_test, y_test
                )
                for name, model in tqdm(
                    model_items,
                    desc="Training",
                    disable=not show_progress
                )
            )

            for name, res in parallel_results:
                results[name] = res
        else:
            # Sequential execution with progress bar
            iterator: Any = self.models.items()
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(self.models),
                    desc="Benchmark",
                    unit="model"
                )

            for name, model in iterator:
                if show_progress:
                    iterator.set_postfix({"model": name})

                # Measure training time
                start_fit = time.time()
                model.fit(X_train, y_train)
                fit_time = time.time() - start_fit

                # Measure prediction time
                start_pred = time.time()
                y_pred = model.predict(X_test)
                predict_time = time.time() - start_pred

                scores = Evaluator.evaluate_for(detect_model_task(model, y_test), y_test, y_pred)
                results[name] = {
                    'scores': scores,
                    'fit_time': fit_time,
                    'predict_time': predict_time
                }

        self.results = results
        return results

    def run_cv(
        self,
        X,
        y,
        cv: int = 5,
        show_progress: bool = True,
        random_state: int = 42,
    ) -> Dict[str, Dict]:
        r"""
        Compare models by cross-validation (K-fold).

        Each model is trained and evaluated on ``cv`` folds; the returned
        metrics are the averages across folds, with the standard deviation
        in ``scores_std``. For classification, folds are stratified.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Features (full dataset, not split).
        y : pandas.Series or array-like
            Target.
        cv : int, default=5
            Number of folds.
        show_progress : bool, default=True
            Show a progress bar.
        random_state : int, default=42
            Seed for fold shuffling.

        Returns
        -------
        dict
            {model_name: {'scores': averages, 'scores_std': standard deviations,
            'fit_time': average time, 'predict_time': average time, 'cv': cv}}

        Examples
        --------
        >>> bench = Benchmark({'knn': KNNModel(), 'rf': RandomForestModel()})
        >>> results = bench.run_cv(X, y, cv=5)
        >>> print(bench.to_dataframe())
        """
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import KFold, StratifiedKFold

        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = y if isinstance(y, pd.Series) else pd.Series(y)

        task = detect_task(y)
        splitter_cls = StratifiedKFold if task == "classification" else KFold
        splitter = splitter_cls(n_splits=cv, shuffle=True, random_state=random_state)

        results = {}
        iterator: Any = self.models.items()
        if show_progress:
            iterator = tqdm(iterator, total=len(self.models), desc=f"CV {cv}-fold", unit="model")

        for name, model in iterator:
            fold_scores, fit_times, predict_times = [], [], []
            model_task = detect_model_task(model, y)
            for train_idx, test_idx in splitter.split(X, y):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                start = time.time()
                model.fit(X_tr, y_tr)
                fit_times.append(time.time() - start)

                start = time.time()
                y_pred = model.predict(X_te)
                predict_times.append(time.time() - start)

                fold_scores.append(Evaluator.evaluate_for(model_task, y_te, y_pred))

            metrics = fold_scores[0].keys()
            results[name] = {
                'scores': {m: float(np.mean([s[m] for s in fold_scores])) for m in metrics},
                'scores_std': {m: float(np.std([s[m] for s in fold_scores])) for m in metrics},
                'fit_time': float(np.mean(fit_times)),
                'predict_time': float(np.mean(predict_times)),
                'cv': cv,
            }

        self.results = results
        return results

    def to_dataframe(self, sort: bool = True) -> Optional["pd.DataFrame"]:
        """
        Convert the benchmark results to a pandas DataFrame.

        Columns are the metrics (plus ``fit_time`` and ``predict_time``),
        rows are the models. If the results come from :meth:`run_cv`,
        ``<metric>_std`` columns are added. The table is sorted by the
        primary metric (``accuracy`` or ``r2``), from best to worst.

        Parameters
        ----------
        sort : bool, default=True
            Sort by the primary metric, descending.

        Returns
        -------
        pandas.DataFrame or None
            Comparison table, or None if run()/run_cv() has not been called.

        Examples
        --------
        >>> bench.run(X_train, y_train, X_test, y_test)
        >>> df = bench.to_dataframe()
        >>> print(df)
        """
        if self.results is None:
            return None
        import pandas as pd

        rows = {}
        for name, res in self.results.items():
            row = dict(res['scores'])
            for m, v in res.get('scores_std', {}).items():
                row[f"{m}_std"] = v
            row['fit_time'] = res['fit_time']
            row['predict_time'] = res['predict_time']
            rows[name] = row
        df = pd.DataFrame.from_dict(rows, orient='index')
        df.index.name = 'model'

        if sort:
            for primary in ('accuracy', 'r2'):
                if primary in df.columns:
                    df = df.sort_values(primary, ascending=False)
                    break
        return df

    def summary(self) -> Optional[str]:
        """
        Return a formatted summary of the benchmark results.

        Returns
        -------
        str or None
            Text summary, or None if no results.
        """
        if self.results is None:
            return None

        lines = ["=" * 60, "BENCHMARK SUMMARY", "=" * 60]

        # Find the best model by accuracy
        best_model = None
        best_accuracy = -1

        for name, res in self.results.items():
            lines.append(f"\n{name}")
            lines.append("-" * 40)
            for metric, value in res['scores'].items():
                lines.append(f"  {metric}: {value:.4f}")
            lines.append(f"  fit_time: {res['fit_time']:.4f}s")
            lines.append(f"  predict_time: {res['predict_time']:.4f}s")

            if res['scores'].get('accuracy', 0) > best_accuracy:
                best_accuracy = res['scores'].get('accuracy', 0)
                best_model = name

        if best_model:
            lines.append("\n" + "=" * 60)
            lines.append(f"BEST MODEL: {best_model} (accuracy: {best_accuracy:.4f})")
            lines.append("=" * 60)

        return "\n".join(lines)

    def print_summary(self) -> None:
        """
        Print the summary of the benchmark results.
        """
        summary = self.summary()
        if summary:
            print(summary)
        else:
            print("No results. Run run() first.")
