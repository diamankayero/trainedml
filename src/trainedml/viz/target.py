"""
Target variable analysis for trainedml.
Shows the target's distribution (classification or regression).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from .vizs import Vizs

if TYPE_CHECKING:
    import pandas as pd


class TargetViz(Vizs):
    """
    Class to visualize the target variable's distribution.
    """
    def __init__(self, data: "pd.DataFrame", target_column: str) -> None:
        super().__init__(data)
        self._target_column = target_column

    def vizs(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        if self._data[self._target_column].dtype == 'object':
            self._data[self._target_column].value_counts().plot(kind='bar', ax=ax, color='purple')
            ax.set_ylabel('Count')
        else:
            ax.hist(self._data[self._target_column].dropna(), bins=20, color='purple', edgecolor='black')
            ax.set_ylabel('Count')
        ax.set_title(f"Target distribution: {self._target_column}")
        self._figure = fig
