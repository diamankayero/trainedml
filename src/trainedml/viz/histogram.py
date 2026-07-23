"""
Histogram visualization for trainedml.

This module provides the HistogramViz class, which generates histograms for one or more columns
using matplotlib, supporting custom binning and legend options.

Examples
--------
>>> from trainedml.viz.histogram import HistogramViz
>>> viz = HistogramViz(df, columns=['A', 'B'])
>>> viz.vizs()
>>> viz.figure.show()
"""

from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .vizs import Vizs


class HistogramViz(Vizs):
    r"""
    Histogram visualization for one or more columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    columns : 'all' or list, default='all'
        Columns to plot.
    legend : bool, default=False
        Show legend if multiple columns.
    bins : int, default=10
        Number of bins.

    Attributes
    ----------
    data : pandas.DataFrame
        The data.
    columns : list
        Columns used.
    legend : bool
        Legend option.
    bins : int
        Number of bins.
    figure : matplotlib.figure.Figure
        The generated figure (after calling vizs).

    Examples
    --------
    >>> viz = HistogramViz(df, columns=['A', 'B'], bins=20)
    >>> viz.vizs()
    >>> viz.figure.show()
    """
    def __init__(self, data: pd.DataFrame, columns: Union[str, List[str]] = 'all', legend: bool = False,
                bins: int = 10, save_path: Optional[str] = None) -> None:
        super().__init__(data, save_path=save_path)
        # Argument validation
        if not isinstance(columns, str) and not isinstance(columns, list):
            raise ValueError('columns must be a string or a list')
        if isinstance(columns, str) and columns != 'all':
            raise ValueError('columns must be "all" or a list of column names')
        if isinstance(columns, list):
            for col in columns:
                if col not in self._data.columns.tolist():
                    raise ValueError(f'Unknown column: {col}')
        if not isinstance(legend, bool):
            raise ValueError('legend must be a boolean')
        if not isinstance(bins, int) or bins < 1:
            raise ValueError('bins must be a positive integer')
        self._columns = columns
        self._legend = legend
        self._bins = bins

    def vizs(self) -> Figure:
        """
        Generate and display the histogram.

        Returns
        -------
        matplotlib.figure.Figure
            The generated histogram figure.
        """
        if self._columns == 'all':
            cols = self._data.columns.tolist()
        else:
            cols = self._columns
        fig, ax = plt.subplots(figsize=(8, 6))
        for col in cols:
            ax.hist(self._data[col].dropna(), bins=self._bins, alpha=0.7, label=col, edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram')
        if self._legend and (len(cols) > 1):
            ax.legend()
        plt.tight_layout()
        self._figure = fig
        self._auto_save()
        return self._figure
