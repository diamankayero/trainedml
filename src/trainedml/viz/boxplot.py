"""
Boxplot visualization for trainedml.

This module provides the BoxplotViz class, which generates boxplots for one or more columns
using matplotlib, supporting grouping by another variable.

Examples
--------
>>> from trainedml.viz.boxplot import BoxplotViz
>>> viz = BoxplotViz(df, columns=['A', 'B'])
>>> viz.vizs()
>>> viz.figure.show()
"""

from __future__ import annotations

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from .vizs import Vizs

class BoxplotViz(Vizs):
    r"""
    Boxplot visualization for one or more columns.

    Boxplot parameters
    -------------------
    data : pandas.DataFrame
        The dataset.
    columns : 'all' or list, default='all'
        Columns to plot.
    by : str or None, default=None
        Grouping variable.
    """
    def __init__(self, data: pd.DataFrame, columns: Union[List[str], str] = 'all',
                by: Optional[str] = None) -> None:
        super().__init__(data)
        self._columns = columns
        self._by = by

    def vizs(self) -> Figure:
        """
        Generate the boxplot figure.

        Returns
        -------
        matplotlib.figure.Figure
            The generated boxplot figure.
        """
        if self._columns == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        else:
            cols = self._columns
        fig, axes = plt.subplots(len(cols), 1, figsize=(8, 4*len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            if self._by:
                self._data.boxplot(column=col, by=self._by, ax=ax)
                ax.set_title(f"Boxplot of {col} by {self._by}")
            else:
                ax.boxplot(self._data[col].dropna(), vert=False)
                ax.set_title(f"Boxplot of {col}")
        plt.tight_layout()
        self._figure = fig
        return fig
