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
    # Mesure du temps d'entraînement
    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Mesure du temps de prédiction
    start_pred = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_pred

    # Métriques adaptées au type de tâche du modèle (classification ou régression)
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
            models (dict): dictionnaire {nom: instance_modele}
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
            # Exécution parallèle avec joblib
            model_items = list(self.models.items())
            
            if show_progress:
                print(f"🚀 Benchmark parallèle de {len(model_items)} modèles...")
            
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(_train_and_evaluate)(
                    name, model, X_train, y_train, X_test, y_test
                )
                for name, model in tqdm(
                    model_items,
                    desc="Entraînement",
                    disable=not show_progress
                )
            )
            
            for name, res in parallel_results:
                results[name] = res
        else:
            # Exécution séquentielle avec barre de progression
            iterator: Any = self.models.items()
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(self.models),
                    desc="Benchmark",
                    unit="modèle"
                )
            
            for name, model in iterator:
                if show_progress:
                    iterator.set_postfix({"modèle": name})
                
                # Mesure du temps d'entraînement
                start_fit = time.time()
                model.fit(X_train, y_train)
                fit_time = time.time() - start_fit

                # Mesure du temps de prédiction
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
        Compare les modèles par validation croisée (K-fold).

        Chaque modèle est entraîné et évalué sur ``cv`` plis ; les métriques
        retournées sont les moyennes sur les plis, avec l'écart-type dans
        ``scores_std``. Pour la classification, les plis sont stratifiés.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Features (jeu complet, non splitté).
        y : pandas.Series or array-like
            Cible.
        cv : int, default=5
            Nombre de plis.
        show_progress : bool, default=True
            Affiche une barre de progression.
        random_state : int, default=42
            Graine pour le mélange des plis.

        Returns
        -------
        dict
            {model_name: {'scores': moyennes, 'scores_std': écarts-types,
            'fit_time': temps moyen, 'predict_time': temps moyen, 'cv': cv}}

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
            iterator = tqdm(iterator, total=len(self.models), desc=f"CV {cv}-fold", unit="modèle")

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
        Convertit les résultats du benchmark en DataFrame pandas.

        Les colonnes sont les métriques (plus ``fit_time`` et ``predict_time``),
        les lignes les modèles. Si les résultats proviennent de :meth:`run_cv`,
        des colonnes ``<metric>_std`` sont ajoutées. Le tableau est trié par la
        métrique principale (``accuracy`` ou ``r2``), du meilleur au moins bon.

        Parameters
        ----------
        sort : bool, default=True
            Trier par la métrique principale, décroissante.

        Returns
        -------
        pandas.DataFrame or None
            Tableau comparatif, ou None si run()/run_cv() n'a pas été exécuté.

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
        
        lines = ["=" * 60, "📊 RÉSUMÉ DU BENCHMARK", "=" * 60]
        
        # Trouver le meilleur modèle par accuracy
        best_model = None
        best_accuracy = -1
        
        for name, res in self.results.items():
            lines.append(f"\n🔹 {name}")
            lines.append("-" * 40)
            for metric, value in res['scores'].items():
                lines.append(f"  {metric}: {value:.4f}")
            lines.append(f"  ⏱️ fit_time: {res['fit_time']:.4f}s")
            lines.append(f"  ⏱️ predict_time: {res['predict_time']:.4f}s")
            
            if res['scores'].get('accuracy', 0) > best_accuracy:
                best_accuracy = res['scores'].get('accuracy', 0)
                best_model = name
        
        if best_model:
            lines.append("\n" + "=" * 60)
            lines.append(f"🏆 MEILLEUR MODÈLE: {best_model} (accuracy: {best_accuracy:.4f})")
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
            print("⚠️ Aucun résultat. Exécutez d'abord run().")
