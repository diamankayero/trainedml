"""
Central visualization and exploratory analysis module for trainedml.

This module provides the Visualizer class, which offers a unified interface for generating
various types of plots and exploratory data analyses from a pandas DataFrame.

Visualization Features
-----------------------
- Correlation heatmaps
- Histograms
- Line plots
- Exploratory analyses (distribution, correlation, missing values, outliers, target, boxplots, bivariate, normality, multicollinearity, profiling)
- Companion plots for outliers, normality, multicollinearity, and the
  target's distribution (``plot_outliers``, ``plot_normality``,
  ``plot_multicollinearity``, ``plot_target``)

Examples
--------
>>> from trainedml.visualization import Visualizer
>>> viz = Visualizer(df)
>>> fig = viz.heatmap()
>>> fig.show()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from matplotlib.figure import Figure

from trainedml.viz.heatmap import HeatmapViz
from trainedml.viz.histogram import HistogramViz
from trainedml.viz.line import LineViz
from trainedml.viz.outliers import OutliersViz
from trainedml.viz.normality import NormalityViz
from trainedml.viz.multicollinearity import MulticollinearityViz
from trainedml.viz.target import TargetViz
from trainedml.analyzer import DataAnalyzer


class Visualizer:
    """
    Central class for visualization and exploratory data analysis.

    This class provides a unified, high-level interface to all visualizations and analyses
    available in trainedml. It is designed to make exploratory data analysis (EDA) and
    scientific visualization as simple and reproducible as possible, with a focus on
    clarity, flexibility, and extensibility.

    Features
    --------
    - Correlation heatmaps (Pearson, Spearman, Kendall)
    - Histograms (single or multiple columns)
    - Line plots (any two columns)
    - Boxplots, bivariate plots, target analysis
    - Full exploratory analysis: distribution, missing values, outliers, normality, VIF, profiling
    - Companion plots for outliers, normality (QQ-plots), multicollinearity
      (VIF bar chart), and the target's distribution
    - All methods return matplotlib Figure or pandas objects for further customization

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to visualize/analyze. Must be a pandas DataFrame with columns as features.

    Attributes
    ----------
    data : pandas.DataFrame
        The underlying data.
    analyzer : DataAnalyzer
        Helper for advanced analyses (distribution, correlation, missing, etc.).

    Examples
    --------
    >>> from trainedml.visualization import Visualizer
    >>> viz = Visualizer(df)
    >>> fig = viz.heatmap()
    >>> fig.show()

    >>> fig1 = viz.histogram(columns=['A', 'B'], bins=20)
    >>> fig2 = viz.line(x_column='A', y_column='B')
    >>> fig1.show(); fig2.show()

    >>> corr = viz.correlation()
    >>> print(corr)
    >>> missing = viz.missing()
    >>> print(missing)

    >>> fig = viz.boxplot(columns=['A', 'B'])
    >>> fig.show()
    >>> fig = viz.bivariate(x='A', y='B')
    >>> fig.show()

    >>> report = viz.profiling()
    >>> print(report['describe'])

    >>> fig = viz.heatmap(features=['A', 'B', 'C'], method='spearman', mask=False)
    >>> fig.show()
    >>> fig = viz.histogram(columns=['A'], bins=50, legend=True)
    >>> fig.show()

    >>> viz = Visualizer(df)
    >>> print(viz.get_features())
    >>> print(viz.missing())
    >>> print(viz.outliers())
    >>> print(viz.normality())
    >>> print(viz.multicollinearity())
    >>> print(viz.profiling())

    Notes
    -----
    - All plotting methods return matplotlib Figure objects (can be saved, customized, etc.).
    - All analysis methods return pandas DataFrame/Series or dicts.
    - For advanced customization, use the returned figure/axes objects directly.
    - The Visualizer is designed to be extended with new visualizations as needed.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.analyzer = DataAnalyzer(data)

    def report(self, path: Optional[str] = None, title: str = "Exploratory report - trainedml") -> str:
        """
        Generate a complete, self-contained HTML EDA report.

        The report gathers every analysis in the package: overview,
        descriptive statistics, missing values, correlations (with
        heatmap), distributions, outliers, normality, and VIF. Figures are
        embedded in the HTML (no external dependency).

        Parameters
        ----------
        path : str, optional
            Output HTML file path. If None, the HTML is only returned.
        title : str, optional
            Report title.

        Returns
        -------
        str
            The report's HTML content.

        Examples
        --------
        >>> viz = Visualizer(df)
        >>> viz.report("report.html")
        """
        from .report import generate_report
        return generate_report(self.data, path=path, title=title)

    def heatmap(self, features: Union[str, List[str]] = 'all', method: str = 'pearson',
                mask: bool = True, **kwargs: Any) -> Any:
        """
        Generate a correlation heatmap between variables.

        This method computes the correlation matrix for the selected features and displays
        it as a heatmap. Useful for quickly visualizing relationships and collinearities.

        Parameters
        ----------
        features : 'all' or list, default='all'
            Features to include in the correlation matrix. Use 'all' for all columns.
        method : str, default='pearson'
            Correlation method ('pearson', 'spearman', 'kendall').
        mask : bool, default=True
            Whether to mask the upper triangle (for symmetric matrices).
        **kwargs :
            Additional arguments for HeatmapViz (e.g., figsize, cmap).

        Returns
        -------
        matplotlib.figure.Figure
            The generated heatmap figure.

        Examples
        --------
        >>> fig = viz.heatmap()
        >>> fig.show()

        >>> fig = viz.heatmap(features=['A', 'B', 'C'], method='spearman', mask=False)
        >>> fig.show()

        >>> fig = viz.heatmap(cmap='viridis', figsize=(12, 8))
        >>> fig.show()
        """
        viz = HeatmapViz(self.data, features=features, method=method, mask=mask)
        viz.vizs()
        return viz.figure

    def histogram(self, columns: Union[str, List[str]] = 'all', legend: bool = False,
                bins: int = 10, **kwargs: Any) -> Figure:
        """
        Generate one or more histograms for selected columns.

        This method plots the distribution of one or more columns as histograms.
        Useful for visualizing the shape, skewness, and outliers of numeric variables.

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to plot. Use 'all' for all numeric columns.
        legend : bool, default=False
            Show legend if multiple columns.
        bins : int, default=10
            Number of bins for the histogram.
        **kwargs :
            Additional arguments for HistogramViz (e.g., color, alpha).

        Returns
        -------
        matplotlib.figure.Figure
            The generated histogram figure.

        Examples
        --------
        >>> fig = viz.histogram()
        >>> fig.show()

        >>> fig = viz.histogram(columns=['A', 'B'], bins=30, legend=True)
        >>> fig.show()

        >>> fig = viz.histogram(columns=['A'], bins=20, color='red', alpha=0.5)
        >>> fig.show()
        """
        viz = HistogramViz(self.data, columns=columns, legend=legend, bins=bins)
        viz.vizs()
        return viz.figure

    def line(self, x_column: str, y_column: str, **kwargs: Any) -> Any:
        """
        Generate a line plot between two columns.

        This method creates a line plot of y_column versus x_column. Useful for time series,
        trends, or any ordered relationship between two variables.

        Parameters
        ----------
        x_column : str
            Column for the x-axis.
        y_column : str
            Column for the y-axis.
        **kwargs :
            Additional arguments for LineViz (e.g., marker, linestyle).

        Returns
        -------
        matplotlib.figure.Figure
            The generated line plot figure.

        Examples
        --------
        >>> fig = viz.line(x_column='A', y_column='B')
        >>> fig.show()

        >>> fig = viz.line(x_column='A', y_column='B', marker='o', linestyle='--')
        >>> fig.show()
        """
        viz = LineViz(self.data, x_column=x_column, y_column=y_column)
        viz.vizs()
        return viz.figure

    def get_features(self) -> List[str]:
        """
        Return the list of feature columns in the DataFrame.

        Returns
        -------
        list
            List of column names.

        Examples
        --------
        >>> features = viz.get_features()
        >>> print(features)
        """
        return self.data.columns.tolist()

    # Exploratory analyses (via DataAnalyzer)
    def distribution(self, columns: Union[str, List[str]] = 'all', **kwargs: Any) -> Dict[str, Any]:
        """
        Distribution of variables (histograms).

        This method computes summary statistics and histograms for the selected columns.
        Useful for quick EDA and for checking variable distributions before modeling.

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to analyze.
        **kwargs :
            Additional arguments for the analyzer.

        Returns
        -------
        dict
            Summary statistics and figures.

        Examples
        --------
        >>> dist = viz.distribution()
        >>> print(dist)
        >>> dist = viz.distribution(columns=['A', 'B'])
        >>> print(dist)
        """
        return self.analyzer.distribution(columns=columns, **kwargs)

    def correlation(self, features: Union[str, List[str]] = 'all', method: str = 'pearson',
                    mask: bool = True, **kwargs: Any) -> pd.DataFrame:
        """
        Correlation matrix (heatmap).

        This method computes the correlation matrix for the selected features.
        Returns a pandas DataFrame (not a plot).

        Parameters
        ----------
        features : 'all' or list, default='all'
            Features to include.
        method : str, default='pearson'
            Correlation method.
        mask : bool, default=True
            Whether to mask the upper triangle.
        **kwargs :
            Additional arguments for the analyzer.

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.

        Examples
        --------
        >>> corr = viz.correlation()
        >>> print(corr)
        """
        return self.analyzer.correlation(features=features, method=method, mask=mask, **kwargs)

    def missing(self, **kwargs: Any) -> pd.DataFrame:
        """
        Missing values analysis.

        This method returns the count of missing values per column.
        Useful for data cleaning and preprocessing.

        Returns
        -------
        pandas.Series
            Count of missing values per column.

        Examples
        --------
        >>> missing = viz.missing()
        >>> print(missing)
        """
        return self.analyzer.missing(**kwargs)

    def outliers(self, **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        """
        Outlier analysis.

        This method detects outliers in the dataset using IQR or Z-score.
        Returns a dictionary with outlier values per column.

        Returns
        -------
        dict
            Outlier summary per column.

        Examples
        --------
        >>> out = viz.outliers()
        >>> print(out)
        """
        return self.analyzer.outliers(**kwargs)

    def plot_outliers(self) -> Figure:
        """
        Boxplot grid highlighting outliers, one panel per numeric column.

        A complement to :meth:`outliers` (which returns counts and bounds
        as data): this draws the same IQR-based outliers visually.

        Returns
        -------
        matplotlib.figure.Figure
            The generated boxplot grid.

        Examples
        --------
        >>> fig = viz.plot_outliers()
        >>> fig.show()
        """
        viz = OutliersViz(self.data)
        viz.vizs()
        return viz.figure

    def target(self, target_column: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Target variable analysis.

        This method analyzes the target variable (distribution, imbalance, etc.).
        Returns a dictionary with summary statistics and plots.

        Parameters
        ----------
        target_column : str
            Name of the target column.
        **kwargs :
            Additional arguments for the analyzer.

        Returns
        -------
        dict
            Target analysis summary.

        Examples
        --------
        >>> target = viz.target(target_column='species')
        >>> print(target)
        """
        return self.analyzer.target(target_column=target_column, **kwargs)

    def plot_target(self, target_column: str) -> Figure:
        """
        Plot the target variable's distribution (bar chart for a categorical
        target, histogram for a numeric one).

        A complement to :meth:`target` (which returns counts as data).

        Parameters
        ----------
        target_column : str
            Name of the target column.

        Returns
        -------
        matplotlib.figure.Figure
            The generated distribution plot.

        Examples
        --------
        >>> fig = viz.plot_target(target_column='species')
        >>> fig.show()
        """
        viz = TargetViz(self.data, target_column=target_column)
        viz.vizs()
        return viz.figure

    def boxplot(self, columns: Union[str, List[str]] = 'all', by: Optional[str] = None,
                **kwargs: Any) -> Figure:
        """
        Boxplots by variable.

        This method generates boxplots for the selected columns, optionally grouped by another variable.
        Returns a matplotlib Figure.

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to plot.
        by : str or None, default=None
            Grouping variable.
        **kwargs :
            Additional arguments for the analyzer.

        Returns
        -------
        matplotlib.figure.Figure
            The generated boxplot figure.

        Examples
        --------
        >>> fig = viz.boxplot(columns=['A', 'B'], by='Group')
        >>> fig.show()
        """
        return self.analyzer.boxplot(columns=columns, by=by, **kwargs)

    def bivariate(self, x: str, y: str, **kwargs: Any) -> Figure:
        """
        Bivariate analysis (scatter, etc.).

        This method generates a scatter plot or other bivariate visualization between two variables.
        Returns a matplotlib Figure.

        Parameters
        ----------
        x : str
            First variable.
        y : str
            Second variable.
        **kwargs :
            Additional arguments for the analyzer.

        Returns
        -------
        matplotlib.figure.Figure
            The generated bivariate plot.

        Examples
        --------
        >>> fig = viz.bivariate(x='A', y='B')
        >>> fig.show()
        """
        return self.analyzer.bivariate(x=x, y=y, **kwargs)

    def normality(self, columns: Union[str, List[str]] = 'all',
                **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        """
        Normality analysis (tests, QQ-plots, etc.).

        This method tests the normality of the selected columns using Shapiro, D'Agostino, Anderson, etc.
        Returns a dictionary of test results per column.

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to test.
        **kwargs :
            Additional arguments for the analyzer.

        Returns
        -------
        dict
            Normality test results per column.

        Examples
        --------
        >>> norm = viz.normality()
        >>> print(norm)
        """
        return self.analyzer.normality(columns=columns, **kwargs)

    def plot_normality(self, columns: Union[str, List[str]] = 'all') -> Figure:
        """
        QQ-plots (quantile-quantile against the normal distribution), one
        panel per numeric column.

        A complement to :meth:`normality` (which returns test statistics as
        data): a QQ-plot makes departures from normality visually obvious
        (curvature, heavy tails) in a way p-values alone do not.

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated QQ-plot grid.

        Examples
        --------
        >>> fig = viz.plot_normality(columns=['A', 'B'])
        >>> fig.show()
        """
        viz = NormalityViz(self.data, columns=columns)
        viz.vizs()
        return viz.figure

    def multicollinearity(self) -> pd.DataFrame:
        """
        Multicollinearity analysis (VIF, etc.).

        This method computes the Variance Inflation Factor (VIF) for each feature.
        Returns a pandas Series with VIF values.

        Returns
        -------
        pandas.Series
            VIF per feature.

        Examples
        --------
        >>> vif = viz.multicollinearity()
        >>> print(vif)
        """
        return self.analyzer.multicollinearity()

    def plot_multicollinearity(self) -> Figure:
        """
        Bar chart of the Variance Inflation Factor (VIF) per feature.

        A complement to :meth:`multicollinearity` (which returns the same
        values as data).

        Returns
        -------
        matplotlib.figure.Figure
            The generated VIF bar chart.

        Examples
        --------
        >>> fig = viz.plot_multicollinearity()
        >>> fig.show()
        """
        viz = MulticollinearityViz(self.data)
        viz.vizs()
        return viz.figure

    def profiling(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Automatic profiling (global report).

        This method generates a global profiling report (summary statistics, missing, outliers, correlation).
        Returns a dictionary with all results.

        Returns
        -------
        dict
            Profiling report (summary statistics, missing, outliers, correlation).

        Examples
        --------
        >>> report = viz.profiling()
        >>> print(report['describe'])
        """
        return self.analyzer.profiling(**kwargs)