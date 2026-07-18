
"""
Module de chargement de données publiques pour trainedml.

Ce module fournit la classe `DataLoader` qui permet de charger facilement des jeux de données open data
ou des fichiers CSV distants, avec gestion du cache local et adaptation automatique du format.

Fonctionnalités principales
--------------------------
- Datasets intégrés chargés localement via scikit-learn (Iris, Wine) — aucun réseau requis
- Chargement de CSV depuis une URL avec cache local pooch (séparateur et hash gérés)
- Retourne X (features) et y (cible) prêts à l'emploi pour le ML
- Séparation train/test via :meth:`DataLoader.split`
- Peut être étendu pour supporter d'autres sources (INSEE, data.gouv.fr, etc.)

Exemple
-------
>>> loader = DataLoader()
>>> X, y = loader.load_dataset(name="iris")
>>> print(X.shape, y.shape)
"""


from __future__ import annotations

from typing import Any, Optional, Tuple

import pandas as pd
import pooch
from sklearn.model_selection import train_test_split as _sklearn_split


class DataLoader:
    r"""
    Classe responsable du chargement et de l'abstraction des jeux de données publics.

    Cette classe isole la logique d'accès aux données : les autres modules n'ont pas à connaître
    la provenance (URL, open data, local, etc.).

    Fonctionnalités principales
    --------------------------
    - Téléchargement et cache automatique de jeux de données publics (Iris, Wine, etc.)
    - Chargement de CSV depuis une URL (avec gestion du séparateur et du hash)
    - Retourne X (features) et y (cible) prêts à l'emploi pour le ML
    - Peut être étendu pour supporter d'autres sources (INSEE, data.gouv.fr, etc.)

    Exemples détaillés
    -----------------
    Chargement du dataset Iris (public) :
    >>> loader = DataLoader()
    >>> X, y = loader.load_dataset(name="iris")
    >>> print(X.shape, y.unique())

    Chargement du dataset Wine (public) :
    >>> X, y = loader.load_dataset(name="wine")
    >>> print(X.columns)

    Chargement d'un CSV distant avec colonne cible :
    >>> url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    >>> X, y = loader.load_dataset(url=url, target="quality")
    >>> print(X.head())

    Chargement d'un CSV custom (séparateur automatique) :
    >>> X, y = loader.load_dataset(url="https://.../data.csv", target="classe")
    >>> print(X.info())

    Notes
    -----
    - Pour ajouter un nouveau dataset, il suffit d'ajouter un bloc dans load_dataset.
    - Le cache local évite de re-télécharger les fichiers à chaque appel.
    """
    def __init__(self) -> None:
        """
        Initialise un DataLoader.

        Prévu pour extension future : configuration, gestion avancée du cache, etc.

        Examples
        --------
        >>> loader = DataLoader()
        """
        pass


    def load_csv_from_url(self, url: str, known_hash: Optional[str] = None,
                          sep: str = ",") -> pd.DataFrame:
        """
        Télécharge un fichier CSV depuis une URL (avec cache local) et le charge dans un DataFrame pandas.

        Parameters
        ----------
        url : str
            Lien direct vers le fichier CSV.
        known_hash : str, optional
            Hash du fichier pour vérification d'intégrité (voir doc pooch).
        sep : str, default=','
            Séparateur du CSV (',' ou ';', etc.).

        Returns
        -------
        pd.DataFrame
            Données chargées dans un DataFrame pandas.

        Raises
        ------
        RuntimeError
            Si le téléchargement ou la lecture échoue.

        Examples
        --------
        Chargement d'un CSV public :
        >>> loader = DataLoader()
        >>> df = loader.load_csv_from_url("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
        >>> print(df.head())

        Chargement d'un CSV avec séparateur point-virgule :
        >>> df = loader.load_csv_from_url("https://.../winequality-red.csv", sep=';')
        >>> print(df.columns)
        """
        try:
            fname = pooch.retrieve(
                url=url,
                known_hash=known_hash or None,
                progressbar=True
            )
            return pd.read_csv(fname, sep=sep)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement des données depuis {url} : {e}")



    def load_dataset(self, name: Optional[str] = None, url: Optional[str] = None,
                     target: Optional[str] = None,
                     sep: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge un dataset par nom connu ou URL, et retourne X, y séparés.

        Cette méthode gère automatiquement le téléchargement, le parsing, et la séparation
        features/cible pour les datasets connus ou les CSV distants.

        Parameters
        ----------
        name : str, optional
            Nom du dataset connu ("iris", "wine", etc.).
        url : str, optional
            URL d'un CSV distant à charger.
        target : str, optional
            Nom de la colonne cible (obligatoire si url).
        sep : str, optional
            Séparateur du CSV (détecté automatiquement pour certains jeux).

        Returns
        -------
        X : pd.DataFrame
            Features (variables explicatives).
        y : pd.Series
            Cible (variable à prédire).

        Raises
        ------
        ValueError
            Si aucun dataset connu ou url+target n'est spécifié.

        Examples
        --------
        Chargement du dataset Iris :
        >>> loader = DataLoader()
        >>> X, y = loader.load_dataset(name="iris")
        >>> print(X.shape, y.unique())

        Chargement du dataset Wine :
        >>> X, y = loader.load_dataset(name="wine")
        >>> print(X.columns)

        Chargement d'un CSV distant :
        >>> url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        >>> X, y = loader.load_dataset(url=url, target="quality")
        >>> print(X.head())

        Chargement d'un CSV custom (séparateur automatique) :
        >>> X, y = loader.load_dataset(url="https://.../data.csv", target="classe")
        >>> print(X.info())
        """
        if name == "iris":
            # Jeu de données Iris, chargé localement via scikit-learn (aucun réseau requis).
            # Colonnes renommées au format seaborn pour rester compatible avec les
            # anciennes versions qui téléchargeaient le CSV seaborn.
            from sklearn.datasets import load_iris
            bunch = load_iris(as_frame=True)
            X = bunch.data.rename(columns={
                "sepal length (cm)": "sepal_length",
                "sepal width (cm)": "sepal_width",
                "petal length (cm)": "petal_length",
                "petal width (cm)": "petal_width",
            })
            y = pd.Series(
                pd.Categorical.from_codes(bunch.target, categories=list(bunch.target_names)),
                name="species",
            ).astype(str)
            return X, y
        elif name == "wine":
            # Jeu de données Wine, chargé localement via scikit-learn (aucun réseau requis).
            from sklearn.datasets import load_wine
            bunch = load_wine(as_frame=True)
            X = bunch.data
            y = bunch.target.rename("class")
            return X, y
        elif url is not None and target is not None:
            # Chargement générique d'un CSV distant
            # Si le CSV est winequality, utiliser sep=';'
            sep_to_use = sep
            if sep_to_use is None:
                if "winequality" in url:
                    sep_to_use = ";"
                else:
                    sep_to_use = ","
            df = self.load_csv_from_url(url, sep=sep_to_use)
            X = df.drop(columns=[target])
            y = df[target]
            return X, y
        else:
            raise ValueError("Spécifiez un nom de dataset connu ou une url+target.")

    def split(self, X: Any, y: Any, test_size: float = 0.2,
              random_state: int = 42) -> Tuple[Any, Any, Any, Any]:
        """
        Sépare les données en ensembles d'entraînement et de test.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Cible.
        test_size : float, default=0.2
            Proportion de l'ensemble de test.
        random_state : int, default=42
            Graine aléatoire pour la reproductibilité.

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)

        Examples
        --------
        >>> loader = DataLoader()
        >>> X, y = loader.load_dataset(name="iris")
        >>> X_train, X_test, y_train, y_test = loader.split(X, y, test_size=0.2)
        """
        return _sklearn_split(X, y, test_size=test_size, random_state=random_state)

    # TODO: Ajouter ici d'autres méthodes pour charger d'autres datasets publics (INSEE, data.gouv.fr, etc.)
