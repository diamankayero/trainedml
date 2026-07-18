"""
Heatmap visualization for correlation matrices in trainedml.

This module provides the HeatmapViz class, which generates correlation heatmaps
using matplotlib and seaborn, supporting various correlation methods and masking options.

Mathematical context
--------------------
- Pearson, Spearman, Kendall correlation
- Masking upper triangle for symmetric matrices

Examples
--------
>>> from trainedml.viz.heatmap import HeatmapViz
>>> viz = HeatmapViz(df)
>>> viz.vizs()
>>> viz.figure.show()
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from .vizs import Vizs


class HeatmapViz(Vizs):
    r"""
    Correlation heatmap visualization.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset.
    features : 'all' or list, default='all'
        Features to include.
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', 'kendall').
    mask : bool, default=True
        Whether to mask the upper triangle.

    Attributes
    ----------
    data : pandas.DataFrame
        The data.
    features : list
        Features used.
    method : str
        Correlation method.
    mask : bool
        Masking option.
    figure : matplotlib.figure.Figure
        The generated figure (after calling vizs).

    Examples
    --------
    >>> viz = HeatmapViz(df, features=['A', 'B'])
    >>> viz.vizs()
    >>> viz.figure.show()
    """
    def __init__(self, data, features='all', method='pearson', mask=True, save_path: Optional[str] = None):
        super().__init__(data, save_path=save_path)
        # Vérification des arguments
        if not isinstance(features, str) and not isinstance(features, list):
            raise ValueError('features doit être une chaîne ou une liste')
        if isinstance(features, str) and features != 'all':
            raise ValueError('features doit être "all" ou une liste de colonnes')
        if isinstance(features, list):
            for e in features:
                if e not in self._data.columns.tolist():
                    raise ValueError(f'Colonne inconnue : {e}')
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError('Méthode de corrélation inconnue')
        if not isinstance(mask, bool):
            raise ValueError('mask doit être un booléen')
        self._features = features
        self._method = method
        self._mask = mask

    def vizs(self):
        """
        Calcule la matrice de corrélation et affiche la heatmap.
        """
        # Sélection des colonnes/features à corréler
        # (avec 'all', seules les colonnes numériques sont retenues : la
        # corrélation n'est pas définie pour les colonnes texte/catégorielles)
        if self._features == 'all':
            cols = self._data.select_dtypes(include=np.number).columns.tolist()
        else:
            cols = self._features
        df = self._data[cols]
        # Calcul de la matrice de corrélation
        corr = df.corr(method=self._method)
        # Création du masque si demandé
        mask = None
        if self._mask:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(10, 8))
        self._figure = sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', square=True)
        plt.title(f"Matrice de corrélation ({self._method})")
        plt.tight_layout()
        self._auto_save()
        return self._figure