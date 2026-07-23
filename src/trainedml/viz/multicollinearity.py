r"""
Multicollinearity analysis utilities for trainedml.

This module provides functions for computing the Variance Inflation Factor (VIF)
to assess multicollinearity among features in a pandas DataFrame.

Mathematical context
--------------------
- VIF: $VIF_j = \frac{1}{1 - R_j^2}$

Examples
--------
>>> from trainedml.viz.multicollinearity import vif_summary
>>> vif = vif_summary(df)
>>> print(vif)
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .vizs import Vizs


def vif_summary(data: pd.DataFrame) -> pd.Series:
    """
    Compute the Variance Inflation Factor (VIF) for each feature.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset (numeric features only).

    Returns
    -------
    pandas.Series
        VIF per feature.

    Examples
    --------
    >>> vif = vif_summary(data=df)
    >>> print(vif)
    """
    X = data.select_dtypes(include=[float, int])
    vif = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns
    )
    return vif

class MulticollinearityViz(Vizs):
    """
    Class to compute and visualize the VIF of each variable.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def vizs(self) -> None:
        X = self._data.select_dtypes(include='number').dropna()
        vif_data = pd.DataFrame()
        vif_data['variable'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        fig, ax = plt.subplots(figsize=(8, 4))
        vif_data.set_index('variable')['VIF'].plot(kind='bar', ax=ax, color='red')
        ax.set_ylabel('VIF')
        ax.set_title('Variance Inflation Factor (VIF)')
        self._figure = fig
