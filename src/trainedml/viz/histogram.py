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

import matplotlib.pyplot as plt
from typing import Optional
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
    def __init__(self, data, columns='all', legend=False, bins=10, save_path: Optional[str] = None):
        super().__init__(data, save_path=save_path)
        # Vérification des arguments
        if not isinstance(columns, str) and not isinstance(columns, list):
            raise ValueError('columns doit être une chaîne ou une liste')
        if isinstance(columns, str) and columns != 'all':
            raise ValueError('columns doit être "all" ou une liste de noms de colonnes')
        if isinstance(columns, list):
            for col in columns:
                if col not in self._data.columns.tolist():
                    raise ValueError(f'Colonne inconnue : {col}')
        if not isinstance(legend, bool):
            raise ValueError('legend doit être un booléen')
        if not isinstance(bins, int) or bins < 1:
            raise ValueError('bins doit être un entier positif')
        self._columns = columns
        self._legend = legend
        self._bins = bins

    def vizs(self):
        """
        Génère et affiche l'histogramme.

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
        ax.set_xlabel('Valeur')
        ax.set_ylabel('Fréquence')
        ax.set_title('Histogramme')
        if self._legend and (len(cols) > 1):
            ax.legend()
        plt.tight_layout()
        self._figure = fig
        self._auto_save()
        return self._figure
