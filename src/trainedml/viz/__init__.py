# This file makes it possible to import the viz sub-package
"""
This module makes it possible to import every visualization and
exploratory analysis in trainedml.viz for centralized access in the package.
"""

from .vizs import Vizs
from .heatmap import HeatmapViz
from .histogram import HistogramViz
from .line import LineViz
from .missing import MissingValuesViz
from .outliers import OutliersViz
from .target import TargetViz
from .normality import NormalityViz
from .multicollinearity import MulticollinearityViz
from .confusion import plot_confusion_matrix
from .roc import plot_roc_curve

__all__ = [
    "Vizs", "HeatmapViz", "HistogramViz", "LineViz",
    "MissingValuesViz", "OutliersViz", "TargetViz",
    "NormalityViz", "MulticollinearityViz",
    "plot_confusion_matrix", "plot_roc_curve",
]
