r"""
Outlier analysis for trainedml.
Shows boxplots to detect outliers for each numeric variable.

Outlier detection using the IQR and Z-score methods.

Mathematical background
------------------------
- IQR: :math:`IQR = Q_3 - Q_1`
- Z-score: :math:`z = \frac{x - \mu}{\sigma}`

Examples
--------
>>> from trainedml.viz.outliers import outlier_summary
>>> summary = outlier_summary(df)
>>> print(summary)
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
from .vizs import Vizs
import numpy as np

class OutliersViz(Vizs):
    """
    Class to visualize outliers via boxplots.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data)

    def vizs(self) -> None:
        cols = self._data.select_dtypes(include='number').columns.tolist()
        fig, axes = plt.subplots(len(cols), 1, figsize=(8, 4*len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            values = self._data[col].dropna()
            try:
                ax.boxplot(values, orientation='horizontal')
            except TypeError:
                # matplotlib < 3.10: the parameter is called vert
                ax.boxplot(values, vert=False)
            ax.set_title(f"Boxplot of {col}")
        plt.tight_layout()
        self._figure = fig

def outlier_summary(data: pd.DataFrame, method: str = 'iqr',
                    threshold: float = 1.5) -> Dict[str, pd.Series]:
    r"""
    Detect outliers in the dataset using IQR or Z-score.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    method : str, default='iqr'
        Outlier detection method ('iqr', 'zscore').
    threshold : float, default=1.5
        Threshold for outlier detection.

    Returns
    -------
    dict
        Outlier summary per column.

    Notes
    -----
    IQR method:
    :math:`Q_1 = 25\%` percentile, :math:`Q_3 = 75\%` percentile
    :math:`IQR = Q_3 - Q_1`
    Outlier if :math:`x < Q_1 - k \cdot IQR` or :math:`x > Q_3 + k \cdot IQR`

    Z-score method:
    :math:`z = \frac{x - \mu}{\sigma}`
    Outlier if :math:`|z| >` threshold

    Examples
    --------
    >>> summary = outlier_summary(df, method='zscore', threshold=3)
    >>> print(summary)
    """
    summary = {}
    for col in data.select_dtypes(include=[float, int]).columns:
        x = data[col].dropna()
        if method == 'iqr':
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outliers = x[(x < lower) | (x > upper)]
        elif method == 'zscore':
            z = (x - x.mean()) / x.std()
            outliers = x[np.abs(z) > threshold]
        else:
            raise ValueError('Unknown method')
        summary[col] = outliers
    return summary
