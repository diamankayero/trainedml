"""
Visualisation des corrélations entre variables sous forme de carte de chaleur (heatmap).
Permet de choisir les colonnes/features à inclure, la méthode de corrélation et l'affichage du masque.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .vizs import Vizs

class HeatmapViz(Vizs):
    """
    Classe pour générer une heatmap de corrélation entre variables (colonnes).
    """
    def __init__(self, data, features='all', method='pearson', mask=True):
        super().__init__(data)
        # Vérification des arguments
        if not isinstance(features, str) and not isinstance(features, list):
            raise ValueError('features doit être une chaîne ou une liste')
        if isinstance(features, str) and features != 'all':
            raise ValueError('features doit être "all" ou une liste de colonnes')
        if isinstance(features, list):
            for e in features:
                if e not in self._data.columns.tolist():
                    raise ValueError(f'Colonne inconnue : {e}')
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError('Méthode de corrélation inconnue')
        if not isinstance(mask, bool):
            raise ValueError('mask doit être un booléen')
        self._features = features
        self._method = method
        self._mask = mask

    def vizs(self):
        """
        Calcule la matrice de corrélation et affiche la heatmap.
        """
        # Sélection des colonnes/features à corréler
        if self._features == 'all':
            cols = self._data.columns.tolist()
        else:
            cols = self._features
        df = self._data[cols]
        # Calcul de la matrice de corrélation
        corr = df.corr(method=self._method)
        # Création du masque si demandé
        mask = None
        if self._mask:
            mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(10, 8))
        self._figure = sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', square=True)
        plt.title(f"Matrice de corrélation ({self._method})")
        plt.tight_layout()
        return self._figure