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

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    def __init__(self, data: pd.DataFrame, features: Union[str, List[str]] = 'all', method: str = 'pearson',
                mask: bool = True, save_path: Optional[str] = None) -> None:
        super().__init__(data, save_path=save_path)
        # Argument validation
        if not isinstance(features, str) and not isinstance(features, list):
            raise ValueError('features must be a string or a list')
        if isinstance(features, str) and features != 'all':
            raise ValueError('features must be "all" or a list of columns')
        if isinstance(features, list):
            for e in features:
                if e not in self._data.columns.tolist():
                    raise ValueError(f'Unknown column: {e}')
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError('Unknown correlation method')
        if not isinstance(mask, bool):
            raise ValueError('mask must be a boolean')
        self._features = features
        self._method = method
        self._mask = mask

    def vizs(self) -> Any:
        """
        Compute the correlation matrix and display the heatmap.

        Returns
        -------
        matplotlib.axes.Axes
            The heatmap axes (as returned by seaborn), also stored on ``self._figure``.
        """
        # Select the columns/features to correlate
        # (with 'all', only numeric columns are kept: correlation is not
        # defined for text/categorical columns)
        if self._features == 'all':
            cols = self._data.select_dtypes(include=np.number).columns.tolist()
        else:
            cols = self._features
        df = self._data[cols]
        # Compute the correlation matrix
        corr = df.corr(method=self._method)
        # Build the mask if requested
        mask = None
        if self._mask:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(10, 8))
        self._figure = sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', square=True)
        plt.title(f"Correlation matrix ({self._method})")
        plt.tight_layout()
        self._auto_save()
        return self._figure