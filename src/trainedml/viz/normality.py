"""
Normality analysis utilities for trainedml.

This module provides functions for testing the normality of variables in a pandas DataFrame,
using Shapiro-Wilk, D'Agostino, and Anderson-Darling tests, and the NormalityViz class,
which shows a QQ-plot for each numeric variable.

Examples
--------
>>> from trainedml.viz.normality import normality_tests
>>> results = normality_tests(df)
>>> print(results)
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from .vizs import Vizs


def normality_tests(data: pd.DataFrame, columns: Union[str, List[str]] = 'all') -> Dict[str, Dict[str, Any]]:
    """
    Perform normality tests on selected columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    columns : 'all' or list, default='all'
        Columns to test.

    Returns
    -------
    dict
        Dictionary of test results per column.

    Examples
    --------
    >>> results = normality_tests(df, columns=['A', 'B'])
    >>> print(results)
    """
    cols = data.columns.tolist() if columns == 'all' else columns
    results = {}
    for col in cols:
        x = data[col].dropna()
        result: Dict[str, Any] = {'shapiro': stats.shapiro(x)}
        # D'Agostino requires n >= 8
        if len(x) >= 8:
            result['dagostino'] = stats.normaltest(x)
        else:
            result['dagostino'] = None
        try:
            result['anderson'] = stats.anderson(x, dist='norm', method='interpolate')
        except TypeError:
            # old scipy: no method parameter (nor an interpolated p-value)
            result['anderson'] = stats.anderson(x, dist='norm')
        results[col] = result
    return results


class NormalityViz(Vizs):
    """
    Class to generate QQ-plots for normality testing.
    """
    def __init__(self, data: pd.DataFrame, columns: Union[str, List[str]] = 'all') -> None:
        super().__init__(data)
        self._columns = columns

    def vizs(self) -> None:
        if self._columns == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        else:
            cols = self._columns
        fig, axes = plt.subplots(len(cols), 1, figsize=(6, 4*len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            stats.probplot(self._data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f"QQ-plot of {col}")
        plt.tight_layout()
        self._figure = fig
