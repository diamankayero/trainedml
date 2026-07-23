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

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from .vizs import Vizs

if TYPE_CHECKING:
    import pandas as pd


def missing_summary(data: "pd.DataFrame") -> "pd.Series":
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
    Class to visualize missing values.
    """
    def __init__(self, data: "pd.DataFrame") -> None:
        super().__init__(data)

    def vizs(self) -> None:
        missing = self._data.isnull().mean() * 100
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=(8, 4))
        if not missing.empty:
            missing.sort_values().plot(kind='barh', ax=ax, color='orange')
            ax.set_xlabel('% missing values')
            ax.set_title('Missing values per column')
        else:
            ax.text(0.5, 0.5, 'No missing values', ha='center', va='center', fontsize=12)
            ax.set_axis_off()
        self._figure = fig
