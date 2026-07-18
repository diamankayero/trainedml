"""
Comparaison de tous les modèles en une ligne pour trainedml.

Ce module expose :func:`compare`, qui enchaîne tout le pipeline :
chargement des données, détection du type de tâche, prétraitement,
validation croisée de tous les modèles adaptés, et retourne un
DataFrame pandas trié du meilleur au moins bon modèle.

Exemple
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
from .tasks import detect_task


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
    Compare tous les modèles adaptés à un dataset et retourne un DataFrame trié.

    Le type de tâche (classification ou régression) est détecté automatiquement
    à partir de la cible, et seuls les modèles adaptés sont comparés. Chaque
    modèle est évalué par validation croisée ; le prétraitement (imputation,
    standardisation, one-hot) est réentraîné à chaque pli pour éviter toute
    fuite d'information.

    Parameters
    ----------
    dataset : str, optional
        Nom d'un dataset intégré ("iris", "wine").
    url : str, optional
        URL d'un CSV distant (nécessite ``target``).
    target : str, optional
        Nom de la colonne cible (si ``url``).
    X : pandas.DataFrame or array-like, optional
        Features fournies directement en mémoire (alternative à dataset/url).
    y : pandas.Series or array-like, optional
        Cible correspondante (obligatoire si X est fourni).
    models : dict, optional
        Dictionnaire {nom: instance} pour comparer des modèles personnalisés
        (trainedml ou scikit-learn). Par défaut : tous les modèles trainedml
        adaptés à la tâche détectée.
    cv : int, default=5
        Nombre de plis de validation croisée.
    preprocess : bool, default=True
        Applique le prétraitement standard de trainedml à chaque pli.
    seed : int, default=42
        Graine aléatoire pour le mélange des plis.
    show_progress : bool, default=True
        Affiche une barre de progression.
    sort : bool, default=True
        Trie le tableau par la métrique principale (accuracy ou r2), décroissante.

    Returns
    -------
    pandas.DataFrame
        Une ligne par modèle : métriques moyennes, écarts-types (``*_std``),
        temps moyens d'entraînement et de prédiction.

    Raises
    ------
    ValueError
        Si ni ``dataset`` ni ``url``+``target`` n'est fourni.

    Examples
    --------
    Comparer tous les classificateurs sur Wine :

    >>> from trainedml import compare
    >>> print(compare(dataset="wine"))

    Comparer des modèles personnalisés (y compris scikit-learn) :

    >>> from sklearn.svm import SVC
    >>> from trainedml.models import KNNModel
    >>> print(compare(dataset="iris", models={"svc": SVC(), "knn": KNNModel()}))

    Sur un CSV distant (régression) :

    >>> print(compare(url="https://.../winequality-red.csv", target="quality"))
    """
    if X is None or y is None:
        loader = DataLoader()
        X, y = loader.load_dataset(name=dataset, url=url, target=target)

    if models is None:
        task = detect_task(y)
        model_map = CLASSIFIER_MAP if task == "classification" else REGRESSOR_MAP
        models = {name: cls() for name, cls in model_map.items()}

    if preprocess:
        models = {name: PreprocessedModel(model) for name, model in models.items()}

    bench = Benchmark(models)
    bench.run_cv(X, y, cv=cv, show_progress=show_progress, random_state=seed)
    return bench.to_dataframe(sort=sort)
