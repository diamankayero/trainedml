"""
Visualisation d'une courbe (line plot) pour trainedml.
"""
import matplotlib.pyplot as plt
from .vizs import Vizs

class LineViz(Vizs):
    """
    Classe pour générer une courbe à partir de deux colonnes de données.
    """
    def __init__(self, data, x_column, y_column):
        super().__init__(data)
        self.x_column = x_column
        self.y_column = y_column

    def vizs(self):
        """
        Affiche une courbe (line plot) entre deux colonnes.
        """
        plt.figure(figsize=(8, 6))
        self._figure = plt.plot(self._data[self.x_column], self._data[self.y_column], marker='o')
        plt.title(f"Courbe {self.y_column} en fonction de {self.x_column}")
        plt.xlabel(self.x_column)
        plt.ylabel(self.y_column)
        plt.tight_layout()
