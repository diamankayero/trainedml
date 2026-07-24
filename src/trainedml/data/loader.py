
"""
Public data loading module for trainedml.

This module provides the `DataLoader` class, which makes it easy to load
open datasets or remote CSV files, with local caching and automatic format
handling.

Main features
-------------
- Built-in datasets loaded locally via scikit-learn (Iris, Wine, Diabetes) - no network required
- CSV loading from a URL with local pooch cache (separator and hash handled)
- Returns X (features) and y (target) ready to use for ML
- Train/test split via :meth:`DataLoader.split`
- Can be extended to support other sources (INSEE, data.gouv.fr, etc.)

Example
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
    Class responsible for loading and abstracting away public datasets.

    This class isolates data-access logic: other modules never need to
    know where the data comes from (URL, open data, local, etc.).

    Features
    --------
    - Automatic download and caching of public datasets (Iris, Wine, Diabetes, etc.)
    - CSV loading from a URL (with separator and hash handling)
    - Returns X (features) and y (target) ready to use for ML
    - Can be extended to support other sources (INSEE, data.gouv.fr, etc.)

    Detailed examples
    -----------------
    Loading the Iris dataset (public):

    >>> loader = DataLoader()
    >>> X, y = loader.load_dataset(name="iris")
    >>> print(X.shape, y.unique())

    Loading the Wine dataset (public):

    >>> X, y = loader.load_dataset(name="wine")
    >>> print(X.columns)

    Loading a remote CSV with a target column:

    >>> url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    >>> X, y = loader.load_dataset(url=url, target="quality")
    >>> print(X.head())

    Loading a custom CSV (automatic separator):

    >>> X, y = loader.load_dataset(url="https://.../data.csv", target="class")
    >>> print(X.info())

    Notes
    -----
    - To add a new dataset, just add a branch in load_dataset.
    - The local cache avoids re-downloading files on every call.
    """
    def __init__(self) -> None:
        """
        Initialize a DataLoader.

        Reserved for future extension: configuration, advanced cache
        management, etc.

        Examples
        --------
        >>> loader = DataLoader()
        """
        pass


    def load_csv_from_url(self, url: str, known_hash: Optional[str] = None,
                          sep: str = ",") -> pd.DataFrame:
        """
        Download a CSV file from a URL (with local caching) and load it into a pandas DataFrame.

        Parameters
        ----------
        url : str
            Direct link to the CSV file.
        known_hash : str, optional
            File hash for integrity verification (see pooch docs).
        sep : str, default=','
            CSV separator (',' or ';', etc.).

        Returns
        -------
        pd.DataFrame
            Data loaded into a pandas DataFrame.

        Raises
        ------
        RuntimeError
            If the download or read fails.

        Examples
        --------
        Loading a public CSV:

        >>> loader = DataLoader()
        >>> df = loader.load_csv_from_url("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
        >>> print(df.head())

        Loading a CSV with a semicolon separator:

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
            raise RuntimeError(f"Error loading data from {url}: {e}")



    def load_dataset(self, name: Optional[str] = None, url: Optional[str] = None,
                     target: Optional[str] = None,
                     sep: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a dataset by known name or URL, and return X, y separately.

        This method automatically handles downloading, parsing, and
        splitting features/target for known datasets or remote CSV files.

        Parameters
        ----------
        name : str, optional
            Name of a known dataset ("iris", "wine", "diabetes", etc.).
        url : str, optional
            URL of a remote CSV to load.
        target : str, optional
            Name of the target column (required if url).
        sep : str, optional
            CSV separator (auto-detected for some datasets).

        Returns
        -------
        X : pd.DataFrame
            Features (explanatory variables).
        y : pd.Series
            Target (variable to predict).

        Raises
        ------
        ValueError
            If neither a known dataset nor url+target is specified.

        Examples
        --------
        Loading the Iris dataset:

        >>> loader = DataLoader()
        >>> X, y = loader.load_dataset(name="iris")
        >>> print(X.shape, y.unique())

        Loading the Wine dataset:

        >>> X, y = loader.load_dataset(name="wine")
        >>> print(X.columns)

        Loading a remote CSV:

        >>> url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        >>> X, y = loader.load_dataset(url=url, target="quality")
        >>> print(X.head())

        Loading a custom CSV (automatic separator):

        >>> X, y = loader.load_dataset(url="https://.../data.csv", target="class")
        >>> print(X.info())
        """
        if name == "iris":
            # Iris dataset, loaded locally via scikit-learn (no network required).
            # Columns renamed to the seaborn format to stay compatible with
            # older versions that downloaded the seaborn CSV.
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
            # Wine dataset, loaded locally via scikit-learn (no network required).
            from sklearn.datasets import load_wine
            bunch = load_wine(as_frame=True)
            X = bunch.data
            y = bunch.target.rename("class")
            return X, y
        elif name == "diabetes":
            # Diabetes dataset (regression), loaded locally via scikit-learn
            # (no network required): the built-in counterpart to iris/wine
            # for regression tasks.
            from sklearn.datasets import load_diabetes
            bunch = load_diabetes(as_frame=True)
            X = bunch.data
            y = bunch.target.rename("disease_progression")
            return X, y
        elif url is not None and target is not None:
            # Generic loading of a remote CSV
            # If the CSV is winequality, use sep=';'
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
            raise ValueError("Specify a known dataset name or a url+target.")

    def split(self, X: Any, y: Any, test_size: float = 0.2,
              random_state: int = 42) -> Tuple[Any, Any, Any, Any]:
        """
        Split the data into training and test sets.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target.
        test_size : float, default=0.2
            Proportion of the test set.
        random_state : int, default=42
            Random seed for reproducibility.

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

    # TODO: Add other methods here to load other public datasets (INSEE, data.gouv.fr, etc.)
