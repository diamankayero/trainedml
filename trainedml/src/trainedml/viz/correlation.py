"""
Matrice de corrélation (heatmap) pour trainedml.
Permet de visualiser les corrélations entre variables numériques.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .vizs import Vizs

class CorrelationViz(Vizs):
    """
    Classe pour générer une heatmap de corrélation.
    """
    def __init__(self, data, features='all', method='pearson', mask=True):
        super().__init__(data)
        self._features = features
        self._method = method
        self._mask = mask

    def vizs(self):
        if self._features == 'all':
            cols = self._data.select_dtypes(include='number').columns.tolist()
        else:
            cols = self._features
        corr = self._data[cols].corr(method=self._method)
        mask = None
        if self._mask:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Matrice de corrélation')
        self._figure = fig
