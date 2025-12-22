"""
Module central de visualisation pour trainedml.
Permet de générer différents types de graphiques à partir d'un DataFrame pandas.
"""

from trainedml.viz.heatmap import HeatmapViz
from trainedml.viz.histogram import HistogramViz
from trainedml.viz.line import LineViz

class Visualizer:
    def __init__(self, data):
        """
        Initialise le visualiseur avec un DataFrame pandas.
        """
        self.data = data

    def heatmap(self, features='all', method='pearson', mask=True, **kwargs):
        """
        Génère une heatmap de corrélation entre variables.
        """
        viz = HeatmapViz(self.data, features=features, method=method, mask=mask)
        viz.vizs()
        return viz.figure

    def histogram(self, columns='all', legend=False, bins=10, **kwargs):
        """
        Génère un ou plusieurs histogrammes pour les colonnes sélectionnées.
        """
        viz = HistogramViz(self.data, columns=columns, legend=legend, bins=bins)
        viz.vizs()
        return viz.figure

    def line(self, x_column, y_column, **kwargs):
        """
        Génère une courbe (line plot) entre deux colonnes.
        """
        viz = LineViz(self.data, x_column=x_column, y_column=y_column)
        viz.vizs()
        return viz.figure

    def get_features(self):
        """
        Retourne la liste des colonnes/features du DataFrame.
        """
        return self.data.columns.tolist()