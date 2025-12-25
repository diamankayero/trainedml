"""
Profiling automatique pour trainedml.
Génère un rapport global de profiling (statistiques descriptives, valeurs manquantes, etc.).
"""

import pandas as pd
from .vizs import Vizs

class ProfilingViz(Vizs):
    """
    Classe pour générer un rapport de profiling automatique.
    """
    def __init__(self, data):
        super().__init__(data)

    def vizs(self):
        # Génère un DataFrame de statistiques descriptives et de valeurs manquantes
        desc = self._data.describe(include='all').T
        missing = self._data.isnull().sum()
        desc['missing'] = missing
        self._figure = desc  # Ici, on retourne un DataFrame, pas une figure matplotlib
