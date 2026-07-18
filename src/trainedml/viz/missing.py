"""
Missing value analysis utilities for trainedml.

This module provides functions for analyzing missing values in a pandas DataFrame,
including counts and visualizations.

Examples
--------
>>> from trainedml.viz.missing import missing_summary
>>> summary = missing_summary(df)
>>> print(summary)
"""

import matplotlib.pyplot as plt
from .vizs import Vizs

def missing_summary(data):
    """
    Compute the count of missing values per column.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.

    Returns
    -------
    pandas.Series
        Count of missing values per column.

    Examples
    --------
    >>> summary = missing_summary(df)
    >>> print(summary)
    """
    return data.isnull().sum()

class MissingValuesViz(Vizs):
    """
    Classe pour visualiser les valeurs manquantes.
    """
    def __init__(self, data):
        super().__init__(data)

    def vizs(self):
        missing = self._data.isnull().mean() * 100
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=(8, 4))
        if not missing.empty:
            missing.sort_values().plot(kind='barh', ax=ax, color='orange')
            ax.set_xlabel('% de valeurs manquantes')
            ax.set_title('Valeurs manquantes par colonne')
        else:
            ax.text(0.5, 0.5, 'Aucune valeur manquante', ha='center', va='center', fontsize=12)
            ax.set_axis_off()
        self._figure = fig
