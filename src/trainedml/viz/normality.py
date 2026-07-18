"""
Normality test utilities for trainedml.

This module provides functions for testing the normality of variables in a pandas DataFrame,
using Shapiro-Wilk, D'Agostino, and Anderson-Darling tests.

Examples
--------
>>> from trainedml.viz.normality import normality_tests
>>> results = normality_tests(df)
>>> print(results)
"""

from scipy import stats

def normality_tests(data, columns='all'):
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
        result = {'shapiro': stats.shapiro(x)}
        # D'Agostino requires n >= 8
        if len(x) >= 8:
            result['dagostino'] = stats.normaltest(x)
        else:
            result['dagostino'] = None
        try:
            result['anderson'] = stats.anderson(x, dist='norm', method='interpolate')
        except TypeError:
            # scipy ancien : pas de paramètre method (ni de p-value interpolée)
            result['anderson'] = stats.anderson(x, dist='norm')
        results[col] = result
    return results

"""
Analyse de la normalité pour trainedml.
Affiche un QQ-plot pour chaque variable numérique.
"""

import matplotlib.pyplot as plt
from .vizs import Vizs

class NormalityViz(Vizs):
    """
    Classe pour générer des QQ-plots pour tester la normalité.
    """
    def __init__(self, data, columns='all'):
        super().__init__(data)
        self._columns = columns

    def vizs(self):
        if self._columns == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        else:
            cols = self._columns
        fig, axes = plt.subplots(len(cols), 1, figsize=(6, 4*len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            stats.probplot(self._data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f"QQ-plot de {col}")
        plt.tight_layout()
        self._figure = fig
