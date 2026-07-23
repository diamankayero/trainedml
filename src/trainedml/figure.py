"""
Figure class adapted for the trainedml project:
Wraps a matplotlib, plotly, etc. figure. Provides multi-backend display, saving, and annotation.

Figure utilities for trainedml visualizations.

This module provides helper functions and classes for managing matplotlib figures
and axes, ensuring consistent style and easy integration with the trainedml visualization API.

Examples
--------
>>> from trainedml.figure import get_figure
>>> fig, ax = get_figure()
>>> ax.plot([1, 2, 3], [4, 5, 6])
>>> fig.show()
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
try:
    import plotly.graph_objects as go  # noqa: F401 -- plotly availability probe
    _plotly_available = True
except ImportError:
    _plotly_available = False

def get_figure(figsize: Tuple[float, float] = (8, 6), dpi: int = 100, nrows: int = 1,
                ncols: int = 1, **kwargs: Any) -> Tuple[MplFigure, Any]:
    """
    Create a matplotlib Figure and Axes with standard style.

    Parameters
    ----------
    figsize : tuple, default=(8, 6)
        Figure size in inches.
    dpi : int, default=100
        Dots per inch.
    nrows : int, default=1
        Number of subplot rows.
    ncols : int, default=1
        Number of subplot columns.
    **kwargs :
        Additional arguments for plt.subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes or array of Axes
        The created axes object(s).

    Examples
    --------
    >>> fig, ax = get_figure(figsize=(10, 4), nrows=2)
    >>> ax[0].hist([1, 2, 2, 3])
    >>> fig.tight_layout()
    >>> fig.show()
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, **kwargs)
    return fig, ax

class Figure:
    """
    Universal figure wrapper for trainedml visualizations (matplotlib, plotly, etc).

    This class encapsulates a figure object and provides unified methods for display,
    saving, and annotation, regardless of the backend (matplotlib or plotly).

    Features
    --------
    - Unified interface for figure display and export
    - Supports matplotlib and plotly backends (auto-detects availability)
    - Methods for show, save, annotate (title/xlabel/ylabel)
    - Stores annotation state for later updates

    Parameters
    ----------
    figure : object, optional
        The figure object (matplotlib.figure.Figure, plotly.graph_objects.Figure, ...).
    backend : {'matplotlib', 'plotly'}, default='matplotlib'
        The backend to use for display and export.

    Examples
    --------
    Basic usage with matplotlib:
    >>> import matplotlib.pyplot as plt
    >>> from trainedml.figure import Figure
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> f = Figure(fig, backend='matplotlib')
    >>> f.annotate(title='Plot', xlabel='X', ylabel='Y')
    >>> f.show()
    >>> f.save('plot.png')

    Basic usage with plotly:
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> fig.add_scatter(x=[1,2,3], y=[4,5,6])
    >>> f = Figure(fig, backend='plotly')
    >>> f.annotate(title='Plot', xlabel='X', ylabel='Y')
    >>> f.show()
    >>> f.save('plot_plotly.png')

    Notes
    -----
    - The backend must match the type of the figure object.
    - For matplotlib, always call `tight_layout()` before saving for best results.
    - For plotly, the `kaleido` package is required for image export.
    """
    def __init__(self, figure: Optional[Any] = None, backend: str = 'matplotlib') -> None:
        """
        Args:
            figure: figure object (matplotlib.figure.Figure, go.Figure, ...)
            backend: 'matplotlib' (default) or 'plotly'
        """
        self.figure = figure
        self.backend = backend
        self._title: Optional[str] = None
        self._xlabel: Optional[str] = None
        self._ylabel: Optional[str] = None

    def show(self) -> None:
        """
        Display the figure using the selected backend.

        Examples
        --------
        >>> fig, ax = get_figure()
        >>> f = Figure(fig)
        >>> f.show()
        """
        if self.figure is None:
            return
        if self.backend == 'matplotlib':
            plt.figure(self.figure.number)
            plt.show()
        elif self.backend == 'plotly' and _plotly_available:
            self.figure.show()
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")

    def save(self, output_path: str) -> None:
        """
        Save the figure to a file using the selected backend.

        Parameters
        ----------
        output_path : str
            Path to save the figure (e.g., 'figure.png').

        Examples
        --------
        >>> fig, ax = get_figure()
        >>> f = Figure(fig)
        >>> f.save('figure.png')
        """
        if self.figure is None:
            return
        if self.backend == 'matplotlib':
            self.figure.tight_layout()
            self.figure.savefig(output_path)
        elif self.backend == 'plotly' and _plotly_available:
            self.figure.write_image(output_path)
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")

    def annotate(self, title: str = '', xlabel: str = '', ylabel: str = '') -> None:
        """
        Add a title and axis labels to the figure (main axis only).

        Parameters
        ----------
        title : str, optional
            Title for the plot.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.

        Examples
        --------
        >>> fig, ax = get_figure()
        >>> f = Figure(fig)
        >>> f.annotate(title='Title', xlabel='X', ylabel='Y')
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
            updates: Dict[str, str] = {}
            if title:
                updates['title'] = title
            if xlabel:
                updates['xaxis_title'] = xlabel
            if ylabel:
                updates['yaxis_title'] = ylabel
            self.figure.update_layout(**updates)
        else:
            raise NotImplementedError(f"Unsupported backend: {self.backend}")
