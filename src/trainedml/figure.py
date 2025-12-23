"""
Classe Figure adaptée au projet trainedml :
Encapsule une figure matplotlib, plotly, etc. Permet l'affichage, la sauvegarde et l'annotation multi-backend.
"""

import matplotlib.pyplot as plt
try:
    import plotly.graph_objects as go
    _plotly_available = True
except ImportError:
    _plotly_available = False

class Figure:
    def __init__(self, figure=None, backend='matplotlib'):
        """
        Args:
            figure: objet figure (matplotlib.figure.Figure, go.Figure, ...)
            backend: 'matplotlib' (défaut) ou 'plotly'
        """
        self.figure = figure
        self.backend = backend
        self._title = None
        self._xlabel = None
        self._ylabel = None

    def show(self):
        """
        Affiche la figure selon le backend.
        """
        if self.figure is None:
            return
        if self.backend == 'matplotlib':
            plt.figure(self.figure.number)
            plt.show()
        elif self.backend == 'plotly' and _plotly_available:
            self.figure.show()
        else:
            raise NotImplementedError(f"Backend non supporté : {self.backend}")

    def save(self, output_path):
        """
        Sauvegarde la figure selon le backend.
        """
        if self.figure is None:
            return
        if self.backend == 'matplotlib':
            self.figure.tight_layout()
            self.figure.savefig(output_path)
        elif self.backend == 'plotly' and _plotly_available:
            self.figure.write_image(output_path)
        else:
            raise NotImplementedError(f"Backend non supporté : {self.backend}")

    def annotate(self, title='', xlabel='', ylabel=''):
        """
        Ajoute un titre et des labels à la figure (pour un seul axe principal).
        """
        for arg in [title, xlabel, ylabel]:
            if not isinstance(arg, str):
                raise ValueError(f'invalid argument {arg}')
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        if self.figure is None:
            return
        if self.backend == 'matplotlib':
            ax = self.figure.gca()
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
        elif self.backend == 'plotly' and _plotly_available:
            updates = {}
            if title:
                updates['title'] = title
            if xlabel:
                updates['xaxis_title'] = xlabel
            if ylabel:
                updates['yaxis_title'] = ylabel
            self.figure.update_layout(**updates)
        else:
            raise NotImplementedError(f"Backend non supporté : {self.backend}")
