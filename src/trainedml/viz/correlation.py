from __future__ import annotations

"""
Correlation analysis utilities for trainedml.

This module provides functions and classes for computing and visualizing correlation matrices
between variables, supporting different correlation methods and visual outputs.

Examples
--------
>>> from trainedml.viz.correlation import correlation_matrix
>>> corr = correlation_matrix(df)
>>> print(corr)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .vizs import Vizs

def correlation_matrix(data, features='all', method='pearson'):
    """
    Calculate the correlation matrix for selected variables.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    features : 'all' or list, default='all'
        Variables to include.
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', 'kendall').

    Returns
    -------
    pandas.DataFrame
        Correlation matrix.

    Examples
    --------
    >>> correlation_matrix(data=df, features=['col1', 'col2'], method='spearman')
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data doit être un DataFrame pandas")
    if features == 'all':
        cols = data.select_dtypes(include='number').columns.tolist()
    elif isinstance(features, list):
        for col in features:
            if col not in data.columns:
                raise ValueError(f"Colonne inconnue : {col}")
        cols = features
    else:
        raise ValueError("features doit être 'all' ou une liste de colonnes")
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("method doit être 'pearson', 'spearman' ou 'kendall'")
    return data[cols].corr(method=method)

class CorrelationViz(Vizs):
    """
    Visualization class for correlation matrices.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to visualize.

    Examples
    --------
    >>> viz = CorrelationViz(data)
    >>> viz.plot()
    """
    def __init__(self, data: pd.DataFrame, features: 'list[str]' | str = 'all', method: str = 'pearson', mask: bool = True):
        super().__init__(data)
        self._features = features
        self._method = method
        self._mask = mask

    def vizs(self) -> None:
        if not isinstance(self._data, pd.DataFrame):
            raise TypeError("data doit être un DataFrame pandas")
        if self._features == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        elif isinstance(self._features, list):
            for col in self._features:
                if col not in self._data.columns:
                    raise ValueError(f"Colonne inconnue : {col}")
            cols = self._features
        else:
            raise ValueError("features doit être 'all' ou une liste de colonnes")
        if self._method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("method doit être 'pearson', 'spearman' ou 'kendall'")
        corr = self._data[cols].corr(method=self._method)
        mask = None
        if self._mask:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Matrice de corrélation')
        self._figure = fig
