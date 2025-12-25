"""
Analyse de la multicolinéarité pour trainedml.
Calcule le VIF (Variance Inflation Factor) pour chaque variable numérique.
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .vizs import Vizs

class MulticollinearityViz(Vizs):
    """
    Classe pour calculer et visualiser le VIF de chaque variable.
    """
    def __init__(self, data):
        super().__init__(data)

    def vizs(self):
        X = self._data.select_dtypes(include='number').dropna()
        vif_data = pd.DataFrame()
        vif_data['variable'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        fig, ax = plt.subplots(figsize=(8, 4))
        vif_data.set_index('variable')['VIF'].plot(kind='bar', ax=ax, color='red')
        ax.set_ylabel('VIF')
        ax.set_title('Variance Inflation Factor (VIF)')
        self._figure = fig
