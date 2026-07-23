"""
Profiling utilities for trainedml.

This module provides functions for generating a global profiling report of a pandas DataFrame,
including summary statistics, missing values, outliers, and correlation.

Examples
--------
>>> from trainedml.viz.profiling import profiling_report
>>> report = profiling_report(df)
>>> print(report)
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import numpy as np
from .vizs import Vizs
from .outliers import outlier_summary

class ProfilingViz(Vizs):
    """
    Class to generate an automatic profiling report.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def vizs(self) -> None:
        # Generates a DataFrame of descriptive statistics and missing values
        desc = self._data.describe(include='all').T
        missing = self._data.isnull().sum()
        desc['missing'] = missing
        self._figure = desc  # Here we return a DataFrame, not a matplotlib figure

def profiling_report(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a profiling report for the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.

    Returns
    -------
    dict
        Profiling report (summary statistics, missing, outliers, correlation).

    Examples
    --------
    >>> report = profiling_report(df)
    >>> print(report)
    """
    numeric_data = data.select_dtypes(include=[np.number])
    summary = {
        'describe': data.describe(),
        'missing': data.isnull().sum(),
        'outliers': outlier_summary(data) if not numeric_data.empty else {},
        'correlation': numeric_data.corr() if not numeric_data.empty else pd.DataFrame(),
    }
    return summary
