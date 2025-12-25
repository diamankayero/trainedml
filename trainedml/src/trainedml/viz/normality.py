"""
Analyse de la normalité pour trainedml.
Affiche un QQ-plot pour chaque variable numérique.
"""

import scipy.stats as stats
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
