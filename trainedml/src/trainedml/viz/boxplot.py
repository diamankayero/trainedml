"""
Boxplots par variable pour trainedml.
Affiche les boxplots pour chaque variable numérique, éventuellement groupés par une variable catégorielle.
"""

import matplotlib.pyplot as plt
from .vizs import Vizs

class BoxplotViz(Vizs):
    """
    Classe pour générer des boxplots par variable.
    """
    def __init__(self, data, columns='all', by=None):
        super().__init__(data)
        self._columns = columns
        self._by = by

    def vizs(self):
        if self._columns == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        else:
            cols = self._columns
        fig, axes = plt.subplots(len(cols), 1, figsize=(8, 4*len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            if self._by:
                self._data.boxplot(column=col, by=self._by, ax=ax)
                ax.set_title(f"Boxplot de {col} par {self._by}")
            else:
                ax.boxplot(self._data[col].dropna(), vert=False)
                ax.set_title(f"Boxplot de {col}")
        plt.tight_layout()
        self._figure = fig
