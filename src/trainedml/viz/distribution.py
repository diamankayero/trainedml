"""
Distribution analysis utilities for trainedml.

This module provides functions and classes for analyzing and visualizing the distribution
of variables, including histograms and summary statistics.

Examples
--------
>>> from trainedml.viz.distribution import distribution_summary
>>> summary = distribution_summary(df)
>>> print(summary)
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd
import matplotlib.pyplot as plt
from .vizs import Vizs

def distribution_summary(data: pd.DataFrame, columns: Union[str, List[str]] = 'all') -> pd.DataFrame:
    """
    Compute summary statistics for selected columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    columns : 'all' or list, default='all'
        Columns to summarize.

    Returns
    -------
    pandas.DataFrame
        Summary statistics (mean, std, min, max, etc.).

    Examples
    --------
    >>> summary = distribution_summary(df, columns=['A', 'B'])
    >>> print(summary)
    """
    cols = data.columns.tolist() if columns == 'all' else columns
    return data[cols].describe()

class DistributionViz(Vizs):
    """
    Class to generate distribution histograms for each variable.
    """
    def __init__(self, data: pd.DataFrame, columns: Union[str, List[str]] = 'all', bins: int = 10) -> None:
        super().__init__(data)
        self._columns = columns
        self._bins = bins

    def vizs(self) -> None:
        if self._columns == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        else:
            cols = self._columns
        fig, axes = plt.subplots(len(cols), 1, figsize=(8, 4*len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.hist(self._data[col].dropna(), bins=self._bins, color='skyblue', edgecolor='black')
            ax.set_title(f"Distribution of {col}")
        plt.tight_layout()
        self._figure = fig
