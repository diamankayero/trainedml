"""
Analyse des valeurs manquantes pour trainedml.
Affiche la proportion de valeurs manquantes par colonne.
"""

import matplotlib.pyplot as plt
from .vizs import Vizs

class MissingValuesViz(Vizs):
    """
    Classe pour visualiser les valeurs manquantes.
    """
    def __init__(self, data):
        super().__init__(data)

    def vizs(self):
        missing = self._data.isnull().mean() * 100
        missing = missing[missing > 0]
        fig, ax = plt.subplots(figsize=(8, 4))
        if not missing.empty:
            missing.sort_values().plot(kind='barh', ax=ax, color='orange')
            ax.set_xlabel('% de valeurs manquantes')
            ax.set_title('Valeurs manquantes par colonne')
        else:
            ax.text(0.5, 0.5, 'Aucune valeur manquante', ha='center', va='center', fontsize=12)
            ax.set_axis_off()
        self._figure = fig
