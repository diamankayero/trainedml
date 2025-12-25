"""
Distribution des variables (histogrammes) pour trainedml.
Permet de visualiser la distribution de chaque variable numérique.
"""

import matplotlib.pyplot as plt
from .vizs import Vizs

class DistributionViz(Vizs):
    """
    Classe pour générer des histogrammes de distribution pour chaque variable.
    """
    def __init__(self, data, columns='all', bins=10):
        super().__init__(data)
        self._columns = columns
        self._bins = bins

    def vizs(self):
        if self._columns == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        else:
            cols = self._columns
        fig, axes = plt.subplots(len(cols), 1, figsize=(8, 4*len(cols)))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.hist(self._data[col].dropna(), bins=self._bins, color='skyblue', edgecolor='black')
            ax.set_title(f"Distribution de {col}")
        plt.tight_layout()
        self._figure = fig
