"""
Line plot visualization for trainedml.

This module provides the LineViz class, which generates line plots between two columns
using matplotlib, supporting custom styling and axis labeling.

Examples
--------
>>> from trainedml.viz.line import LineViz
>>> viz = LineViz(df, x_column='A', y_column='B')
>>> viz.vizs()
>>> viz.figure.show()
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from .vizs import Vizs


class LineViz(Vizs):
    r"""
    Line plot visualization between two columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    x_column : str
        Column for the x-axis.
    y_column : str
        Column for the y-axis.
    save_path : str or None
        Optional save path.
    """
    def __init__(self, data: pd.DataFrame, x_column: str, y_column: str,
                save_path: Optional[str] = None) -> None:
        super().__init__(data, save_path)
        self._x_column = x_column
        self._y_column = y_column

    def vizs(self) -> None:
        """
        Generate the line plot figure.

        The plotted lines (matplotlib.lines.Line2D) are stored on
        ``self._figure`` for consistency with the other Vizs subclasses,
        even though they are not a matplotlib Figure.
        """
        plt.figure(figsize=(8, 6))
        self._figure = plt.plot(self._data[self._x_column], self._data[self._y_column], marker='o')
        plt.title(f"{self._y_column} vs {self._x_column}")
        plt.xlabel(self._x_column)
        plt.ylabel(self._y_column)
        plt.tight_layout()
        self._auto_save()
