"""
Analyse de la variable cible pour trainedml.
Affiche la distribution de la cible (classification ou régression).
"""

import matplotlib.pyplot as plt
from .vizs import Vizs

class TargetViz(Vizs):
    """
    Classe pour visualiser la distribution de la variable cible.
    """
    def __init__(self, data, target_column):
        super().__init__(data)
        self._target_column = target_column

    def vizs(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        if self._data[self._target_column].dtype == 'object':
            self._data[self._target_column].value_counts().plot(kind='bar', ax=ax, color='purple')
            ax.set_ylabel('Nombre d\'occurrences')
        else:
            ax.hist(self._data[self._target_column].dropna(), bins=20, color='purple', edgecolor='black')
            ax.set_ylabel('Effectif')
        ax.set_title(f"Distribution de la cible : {self._target_column}")
        self._figure = fig
