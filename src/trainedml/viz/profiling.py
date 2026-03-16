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

import pandas as pd
import numpy as np
from .vizs import Vizs
from .outliers import outlier_summary

class ProfilingViz(Vizs):
    """
    Classe pour générer un rapport de profiling automatique.
    """
    def __init__(self, data):
        super().__init__(data)

    def vizs(self):
        # Génère un DataFrame de statistiques descriptives et de valeurs manquantes
        desc = self._data.describe(include='all').T
        missing = self._data.isnull().sum()
        desc['missing'] = missing
        self._figure = desc  # Ici, on retourne un DataFrame, pas une figure matplotlib

def profiling_report(data):
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
