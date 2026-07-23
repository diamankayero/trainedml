"""
Base class for trainedml visualizations.

This module provides the Vizs class, which serves as a base for all visualization classes
in trainedml. It defines the interface and common attributes for visualizations.

Examples
--------
>>> from trainedml.viz.vizs import Vizs
>>> class MyViz(Vizs):
...     def vizs(self):
...         # custom plotting code
...         pass
"""

from __future__ import annotations

import os
from typing import Any, Optional
import pandas as pd
import matplotlib.pyplot as plt


class Vizs(object):
    """
    Base class for every visualization.
    All visualizations must inherit from this class and override the vizs() method.

    Attributes:
        _data: pandas DataFrame containing the data
        _figure: generated matplotlib figure
        _save_path: optional path to automatically save the figure
    """
    def __init__(self, data: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Initialize the visualization.

        Args:
            data: pandas DataFrame containing the data
            save_path (str, optional): path to automatically save the figure.
                                       Supported formats: png, pdf, svg, jpg, etc.
        """
        # Check that data is indeed a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError('data must be a pandas DataFrame')
        self._data = data
        # Stores the generated output: usually a matplotlib Figure, but some
        # subclasses (e.g. ProfilingViz) return a pandas DataFrame instead.
        self._figure: Any = None
        self._save_path = save_path

    def vizs(self) -> Any:
        """
        Method to override in subclasses to generate the visualization.
        Automatically calls save() if a save_path was set.

        Most subclasses return None and store their output on
        ``self._figure`` instead; a few (e.g. HistogramViz, BoxplotViz)
        also return it directly, hence the ``Any`` return type here.
        """
        raise NotImplementedError('Subclasses must implement this method')

    def save(self, path: Optional[str] = None, dpi: int = 150, **kwargs: Any) -> Optional[str]:
        """
        Save the figure to a file.

        Args:
            path (str, optional): file path. If None, uses self._save_path
            dpi (int): image resolution (default: 150)
            **kwargs: extra arguments passed to plt.savefig()

        Returns:
            str: path to the saved file, or None on failure

        Raises:
            ValueError: if no path is specified and _save_path is None
        """
        save_path = path or self._save_path

        if save_path is None:
            raise ValueError("No path specified. Pass 'path' or set save_path at init time.")

        # Create the parent directory if needed
        parent_dir = os.path.dirname(save_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        try:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', **kwargs)
            print(f"Figure saved: {save_path}")
            return save_path
        except Exception as e:
            print(f"Error while saving: {e}")
            return None

    def _auto_save(self) -> None:
        """
        Automatically save if a save_path was set.
        Should be called at the end of vizs() in subclasses.
        """
        if self._save_path:
            self.save()

    @property
    def figure(self) -> Any:
        """Return the generated figure."""
        return self._figure

    @property
    def save_path(self) -> Optional[str]:
        """Return the configured save path."""
        return self._save_path

    @save_path.setter
    def save_path(self, value: Optional[str]) -> None:
        """Set the save path."""
        self._save_path = value
