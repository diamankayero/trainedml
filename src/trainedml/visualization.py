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

Examples
--------
>>> from trainedml.visualization import Visualizer
>>> viz = Visualizer(df)
>>> fig = viz.heatmap()
>>> fig.show()
"""

from trainedml.viz.heatmap import HeatmapViz
from trainedml.viz.histogram import HistogramViz
from trainedml.viz.line import LineViz
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
    def __init__(self, data):
        self.data = data
        self.analyzer = DataAnalyzer(data)

    def report(self, path=None, title="Rapport exploratoire - trainedml"):
        """
        Génère un rapport EDA HTML complet et auto-contenu.

        Le rapport rassemble toutes les analyses du package : aperçu,
        statistiques descriptives, valeurs manquantes, corrélations (avec
        heatmap), distributions, outliers, normalité et VIF. Les figures
        sont embarquées dans le HTML (aucune dépendance externe).

        Parameters
        ----------
        path : str, optional
            Chemin du fichier HTML de sortie. Si None, le HTML est seulement retourné.
        title : str, optional
            Titre du rapport.

        Returns
        -------
        str
            Le contenu HTML du rapport.

        Examples
        --------
        >>> viz = Visualizer(df)
        >>> viz.report("rapport.html")
        """
        from .report import generate_report
        return generate_report(self.data, path=path, title=title)

    def heatmap(self, features='all', method='pearson', mask=True, **kwargs):
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

    def histogram(self, columns='all', legend=False, bins=10, **kwargs):
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

    def line(self, x_column, y_column, **kwargs):
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

    def get_features(self):
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
    def distribution(self, columns='all', **kwargs):
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

    def correlation(self, features='all', method='pearson', mask=True, **kwargs):
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

    def missing(self, **kwargs):
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

    def outliers(self, **kwargs):
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

    def target(self, target_column, **kwargs):
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

    def boxplot(self, columns='all', by=None, **kwargs):
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

    def bivariate(self, x, y, **kwargs):
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

    def normality(self, columns='all', **kwargs):
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

    def multicollinearity(self, **kwargs):
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
        return self.analyzer.multicollinearity(**kwargs)

    def profiling(self, **kwargs):
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