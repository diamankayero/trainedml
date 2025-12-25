"""
Analyse bivariée pour trainedml.
Affiche un nuage de points (scatter plot) entre deux variables.
"""

import matplotlib.pyplot as plt
from .vizs import Vizs

class BivariateViz(Vizs):
    """
    Classe pour générer une analyse bivariée (scatter plot).
    """
    def __init__(self, data, x, y):
        super().__init__(data)
        self._x = x
        self._y = y

    def vizs(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(self._data[self._x], self._data[self._y], alpha=0.7)
        ax.set_xlabel(self._x)
        ax.set_ylabel(self._y)
        ax.set_title(f"Scatter plot : {self._x} vs {self._y}")
        self._figure = fig
