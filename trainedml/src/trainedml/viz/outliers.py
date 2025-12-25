"""
Analyse des outliers (valeurs aberrantes) pour trainedml.
Affiche les boxplots pour détecter les outliers par variable numérique.
"""

import matplotlib.pyplot as plt
from .vizs import Vizs

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
            ax.boxplot(self._data[col].dropna(), vert=False)
            ax.set_title(f"Boxplot de {col}")
        plt.tight_layout()
        self._figure = fig
