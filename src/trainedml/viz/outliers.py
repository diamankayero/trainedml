r"""
Analyse des outliers (valeurs aberrantes) pour trainedml.
Affiche les boxplots pour détecter les outliers par variable numérique.

Détection d'outliers par les méthodes IQR et Z-score.

Contexte mathématique
--------------------
- IQR: $IQR = Q_3 - Q_1$
- Z-score: $z = \frac{x - \mu}{\sigma}$

Exemples
--------
>>> from trainedml.viz.outliers import outlier_summary
>>> summary = outlier_summary(df)
>>> print(summary)
"""

import matplotlib.pyplot as plt
from .vizs import Vizs
import pandas as pd
import numpy as np

class OutliersViz(Vizs):
    """
    Classe pour visualiser les outliers via boxplots.
    """
    def __init__(self, data):
        super().__init__(data)

    def vizs(self):
        cols = self._data.select_dtypes(include='number').columns.tolist()
        fig, axes = plt.subplots(len(cols), 1, figsize=(8, 4*len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.boxplot(self._data[col].dropna(), orientation='horizontal')
            ax.set_title(f"Boxplot de {col}")
        plt.tight_layout()
        self._figure = fig

def outlier_summary(data, method='iqr', threshold=1.5):
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
    $Q_1 = 25\%$ percentile, $Q_3 = 75\%$ percentile
    $IQR = Q_3 - Q_1$
    Outlier if $x < Q_1 - k \cdot IQR$ or $x > Q_3 + k \cdot IQR$

    Z-score method:
    $z = \frac{x - \mu}{\sigma}$
    Outlier if $|z| >$ threshold

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
