
"""
Module central de visualisation et d'analyse exploratoire pour trainedml.
Permet de générer différents types de graphiques et d'analyses à partir d'un DataFrame pandas :
- Visualisations (heatmap, histogramme, courbe)
- Analyses exploratoires (distribution, corrélation, valeurs manquantes, outliers, cible, boxplots, bivariée, normalité, multicolinéarité, profiling)
"""


from trainedml.viz.heatmap import HeatmapViz
from trainedml.viz.histogram import HistogramViz
from trainedml.viz.line import LineViz
from trainedml.analyzer import DataAnalyzer


class Visualizer:
    """
    Classe centrale pour la visualisation et l'analyse exploratoire.
    Permet d'accéder à toutes les visualisations et analyses via une interface unique.
    """
    def __init__(self, data):
        self.data = data
        self.analyzer = DataAnalyzer(data)

    # Visualisations classiques
    def heatmap(self, features='all', method='pearson', mask=True, **kwargs):
        """Génère une heatmap de corrélation entre variables."""
        viz = HeatmapViz(self.data, features=features, method=method, mask=mask)
        viz.vizs()
        return viz.figure

    def histogram(self, columns='all', legend=False, bins=10, **kwargs):
        """Génère un ou plusieurs histogrammes pour les colonnes sélectionnées."""
        viz = HistogramViz(self.data, columns=columns, legend=legend, bins=bins)
        viz.vizs()
        return viz.figure

    def line(self, x_column, y_column, **kwargs):
        """Génère une courbe (line plot) entre deux colonnes."""
        viz = LineViz(self.data, x_column=x_column, y_column=y_column)
        viz.vizs()
        return viz.figure

    def get_features(self):
        """Retourne la liste des colonnes/features du DataFrame."""
        return self.data.columns.tolist()

    # Analyses exploratoires (via DataAnalyzer)
    def distribution(self, columns='all', **kwargs):
        """Distribution des variables (histogrammes)."""
        return self.analyzer.distribution(columns=columns, **kwargs)

    def correlation(self, features='all', method='pearson', mask=True, **kwargs):
        """Matrice de corrélation (heatmap)."""
        return self.analyzer.correlation(features=features, method=method, mask=mask, **kwargs)

    def missing(self, **kwargs):
        """Analyse des valeurs manquantes."""
        return self.analyzer.missing(**kwargs)

    def outliers(self, **kwargs):
        """Analyse des outliers (valeurs aberrantes)."""
        return self.analyzer.outliers(**kwargs)

    def target(self, target_column, **kwargs):
        """Analyse de la variable cible."""
        return self.analyzer.target(target_column=target_column, **kwargs)

    def boxplot(self, columns='all', by=None, **kwargs):
        """Boxplots par variable."""
        return self.analyzer.boxplot(columns=columns, by=by, **kwargs)

    def bivariate(self, x, y, **kwargs):
        """Analyse bivariée (scatter, etc.)."""
        return self.analyzer.bivariate(x=x, y=y, **kwargs)

    def normality(self, columns='all', **kwargs):
        """Analyse de la normalité (tests, QQ-plots, etc.)."""
        return self.analyzer.normality(columns=columns, **kwargs)

    def multicollinearity(self, **kwargs):
        """Analyse de la multicolinéarité (VIF, etc.)."""
        return self.analyzer.multicollinearity(**kwargs)

    def profiling(self, **kwargs):
        """Profiling automatique (rapport global)."""
        return self.analyzer.profiling(**kwargs)