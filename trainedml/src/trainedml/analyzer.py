"""
Module d'analyse exploratoire des données pour trainedml.
Fournit une interface centralisée pour les analyses classiques :
- Distribution des variables
- Matrice de corrélation
- Analyse des valeurs manquantes
- Analyse des outliers
- Analyse de la cible
- Boxplots par variable
- Analyse bivariée
- Analyse de la normalité
- Analyse de la multicolinéarité
- Profiling automatique

Chaque analyse est implémentée dans une classe dédiée dans le dossier viz/ et accessible via DataAnalyzer.
"""

import pandas as pd
from trainedml.viz.distribution import DistributionViz
from trainedml.viz.correlation import CorrelationViz
from trainedml.viz.missing import MissingValuesViz
from trainedml.viz.outliers import OutliersViz
from trainedml.viz.target import TargetViz
from trainedml.viz.boxplot import BoxplotViz
from trainedml.viz.bivariate import BivariateViz
from trainedml.viz.normality import NormalityViz
from trainedml.viz.multicollinearity import MulticollinearityViz
from trainedml.viz.profiling import ProfilingViz

class DataAnalyzer:
    """
    Classe centrale pour l'analyse exploratoire des données.
    Permet d'accéder à toutes les analyses via une interface unique.
    """
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError('data doit être un DataFrame pandas')
        self.data = data

    def distribution(self, columns='all', **kwargs):
        """Distribution des variables (histogrammes)."""
        viz = DistributionViz(self.data, columns=columns, **kwargs)
        viz.vizs()
        return viz.figure

    def correlation(self, features='all', method='pearson', mask=True, **kwargs):
        """Matrice de corrélation (heatmap)."""
        viz = CorrelationViz(self.data, features=features, method=method, mask=mask, **kwargs)
        viz.vizs()
        return viz.figure

    def missing(self, **kwargs):
        """Analyse des valeurs manquantes."""
        viz = MissingValuesViz(self.data, **kwargs)
        viz.vizs()
        return viz.figure

    def outliers(self, **kwargs):
        """Analyse des outliers (valeurs aberrantes)."""
        viz = OutliersViz(self.data, **kwargs)
        viz.vizs()
        return viz.figure

    def target(self, target_column, **kwargs):
        """Analyse de la variable cible."""
        viz = TargetViz(self.data, target_column=target_column, **kwargs)
        viz.vizs()
        return viz.figure

    def boxplot(self, columns='all', by=None, **kwargs):
        """Boxplots par variable."""
        viz = BoxplotViz(self.data, columns=columns, by=by, **kwargs)
        viz.vizs()
        return viz.figure

    def bivariate(self, x, y, **kwargs):
        """Analyse bivariée (scatter, etc.)."""
        viz = BivariateViz(self.data, x=x, y=y, **kwargs)
        viz.vizs()
        return viz.figure

    def normality(self, columns='all', **kwargs):
        """Analyse de la normalité (tests, QQ-plots, etc.)."""
        viz = NormalityViz(self.data, columns=columns, **kwargs)
        viz.vizs()
        return viz.figure

    def multicollinearity(self, **kwargs):
        """Analyse de la multicolinéarité (VIF, etc.)."""
        viz = MulticollinearityViz(self.data, **kwargs)
        viz.vizs()
        return viz.figure

    def profiling(self, **kwargs):
        """Profiling automatique (rapport global)."""
        viz = ProfilingViz(self.data, **kwargs)
        viz.vizs()
        return viz.figure
