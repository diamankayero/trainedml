"""
Classe de base pour toutes les visualisations du projet trainedml.
Fournit une interface commune et vérifie le type des données.
"""

import pandas as pd

class Vizs(object):
    """
    Classe de base pour toutes les visualisations.
    Toutes les visualisations doivent hériter de cette classe et surcharger la méthode vizs().
    """
    def __init__(self, data):
        # Vérifie que les données sont bien un DataFrame pandas
        if not isinstance(data, pd.DataFrame):
            raise ValueError('data doit être un DataFrame pandas')
        self._data = data
        self._figure = None  # Stocke la figure générée (matplotlib, seaborn, etc.)

    def vizs(self):
        # Méthode à surcharger dans les classes filles pour générer la visualisation
        raise NotImplementedError('Les sous-classes doivent implémenter cette méthode')

    @property
    def figure(self):
        # Retourne la figure générée
        return self._figure