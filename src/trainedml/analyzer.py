"""
Data analysis and exploratory statistics for trainedml.

This module provides the DataAnalyzer class, which offers a suite of methods for
descriptive statistics, distribution analysis, correlation, missing values, outliers,
target analysis, boxplots, bivariate analysis, normality, multicollinearity, and profiling.

Mathematical context
--------------------
- Correlation: Pearson, Spearman, Kendall
- Outlier detection: IQR, Z-score
- Normality: Shapiro-Wilk, D'Agostino, Anderson-Darling
- Multicollinearity: Variance Inflation Factor (VIF)

Examples
--------
>>> from trainedml.analyzer import DataAnalyzer
>>> analyzer = DataAnalyzer(df)
>>> analyzer.correlation()
>>> analyzer.outliers()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


class DataAnalyzer:
    r"""
    Exploratory data analysis and statistics.

    Provides a suite of methods for descriptive statistics, distribution analysis, correlation,
    missing values, outliers, target analysis, boxplots, bivariate analysis, normality,
    multicollinearity, and profiling.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to analyze.

    Attributes
    ----------
    data : pandas.DataFrame
        The underlying data.

    Examples
    --------
    Basic usage:
    >>> from trainedml.analyzer import DataAnalyzer
    >>> analyzer = DataAnalyzer(df)
    >>> stats = analyzer.distribution()
    >>> print(stats)

    Correlation matrix:
    >>> corr = analyzer.correlation(method='spearman')
    >>> print(corr)

    Outlier detection:
    >>> out = analyzer.outliers(method='zscore', threshold=3)
    >>> print(out)

    Normality tests:
    >>> norm = analyzer.normality()
    >>> print(norm)

    Profiling report:
    >>> report = analyzer.profiling()
    >>> print(report['describe'])

    Notes
    -----
    - All methods return pandas objects or dicts for easy integration with pandas workflows.
    - For plotting, returned objects are matplotlib figures.
    """

    def __init__(self, data):
        self.data = data

    def _select_numeric(self, columns='all'):
        """Return numeric columns from the DataFrame."""
        if columns == 'all':
            return self.data.select_dtypes(include=[np.number])
        if isinstance(columns, str):
            columns = [columns]
        return self.data[columns].select_dtypes(include=[np.number])

    def distribution(self, columns='all', **kwargs):
        """
        Compute and plot the distribution of variables.

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to analyze.
        **kwargs :
            Additional arguments for plotting.

        Returns
        -------
        dict
            Dictionary with keys 'describe' (summary stats) and 'figure' (histogram grid).

        Examples
        --------
        >>> stats = analyzer.distribution()
        >>> print(stats['describe'])
        >>> stats['figure'].show()
        """
        df_num = self._select_numeric(columns)
        cols = df_num.columns.tolist()
        n = len(cols)
        if n == 0:
            return {'describe': pd.DataFrame(), 'figure': None}

        ncols_plot = min(n, 3)
        nrows_plot = (n + ncols_plot - 1) // ncols_plot
        fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(5 * ncols_plot, 4 * nrows_plot))
        axes = np.atleast_1d(axes).flatten()

        for i, col in enumerate(cols):
            ax = axes[i]
            ax.hist(df_num[col].dropna(), bins=kwargs.get('bins', 30), edgecolor='black', alpha=0.7)
            ax.set_title(col)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()

        return {'describe': df_num.describe(), 'figure': fig}

    def correlation(self, features='all', method='pearson', mask=True, **kwargs):
        r"""
        Compute the correlation matrix between features.

        Parameters
        ----------
        features : 'all' or list, default='all'
            Features to include.
        method : str, default='pearson'
            Correlation method ('pearson', 'spearman', 'kendall').
        mask : bool, default=True
            Whether to mask the upper triangle.
        **kwargs :
            Additional arguments for plotting.

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.

        Notes
        -----
        Pearson correlation:
        $r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$

        Examples
        --------
        >>> corr = analyzer.correlation()
        >>> print(corr)
        >>> corr = analyzer.correlation(features=['A', 'B'], method='kendall')
        >>> print(corr)
        """
        df_num = self._select_numeric(features)
        corr = df_num.corr(method=method)

        if mask:
            mask_arr = np.triu(np.ones_like(corr, dtype=bool), k=1)
            corr = corr.where(~mask_arr)

        return corr

    def missing(self, **kwargs):
        """
        Analyze missing values in the dataset.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns 'count', 'percent' for each column with missing values.

        Examples
        --------
        >>> missing = analyzer.missing()
        >>> print(missing)
        """
        total = self.data.isnull().sum()
        percent = (total / len(self.data)) * 100
        result = pd.DataFrame({'count': total, 'percent': percent})
        result = result[result['count'] > 0].sort_values('count', ascending=False)
        return result

    def outliers(self, method='iqr', threshold=1.5, **kwargs):
        r"""
        Detect outliers in the dataset.

        Parameters
        ----------
        method : str, default='iqr'
            Outlier detection method ('iqr', 'zscore').
        threshold : float, default=1.5
            Threshold for outlier detection (IQR multiplier or Z-score cutoff).
        **kwargs :
            Additional arguments.

        Returns
        -------
        dict
            Dictionary per column with keys 'count', 'indices', 'lower_bound', 'upper_bound'.

        Notes
        -----
        IQR method:
        $Q_1 = 25\%$ percentile, $Q_3 = 75\%$ percentile
        $IQR = Q_3 - Q_1$
        Outlier if $x < Q_1 - k \cdot IQR$ or $x > Q_3 + k \cdot IQR$

        Examples
        --------
        >>> out = analyzer.outliers()
        >>> print(out)
        >>> out = analyzer.outliers(method='zscore', threshold=3)
        >>> print(out)
        """
        df_num = self._select_numeric()
        results = {}

        for col in df_num.columns:
            x = df_num[col].dropna()
            if method == 'iqr':
                q1 = x.quantile(0.25)
                q3 = x.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                mask = (x < lower) | (x > upper)
            elif method == 'zscore':
                z = np.abs((x - x.mean()) / x.std())
                lower = x.mean() - threshold * x.std()
                upper = x.mean() + threshold * x.std()
                mask = z > threshold
            else:
                raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

            outlier_idx = x[mask].index.tolist()
            results[col] = {
                'count': int(mask.sum()),
                'indices': outlier_idx,
                'lower_bound': float(lower),
                'upper_bound': float(upper),
            }

        return results

    def target(self, target_column, **kwargs):
        """
        Analyze the target variable (distribution, imbalance, etc.).

        Parameters
        ----------
        target_column : str
            Name of the target column.
        **kwargs :
            Additional arguments.

        Returns
        -------
        dict
            Target analysis summary with keys 'value_counts', 'percent', 'n_unique', 'dtype'.

        Examples
        --------
        >>> target = analyzer.target(target_column='species')
        >>> print(target)
        """
        y = self.data[target_column]
        counts = y.value_counts()
        percent = (counts / len(y)) * 100
        return {
            'value_counts': counts,
            'percent': percent,
            'n_unique': int(y.nunique()),
            'dtype': str(y.dtype),
            'missing': int(y.isnull().sum()),
        }

    def boxplot(self, columns='all', by=None, **kwargs):
        """
        Generate boxplots for selected columns.

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to plot.
        by : str or None, default=None
            Grouping variable.
        **kwargs :
            Additional arguments for plotting.

        Returns
        -------
        matplotlib.figure.Figure
            The generated boxplot figure.

        Examples
        --------
        >>> fig = analyzer.boxplot(columns=['A', 'B'], by='Group')
        >>> fig.show()
        """
        df_num = self._select_numeric(columns)
        cols = df_num.columns.tolist()

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        if by is not None and by in self.data.columns:
            self.data.boxplot(column=cols, by=by, ax=ax)
            ax.set_title(f'Boxplot grouped by {by}')
        else:
            df_num[cols].boxplot(ax=ax)
            ax.set_title('Boxplot')
        ax.set_ylabel('Value')
        fig.tight_layout()
        return fig

    def bivariate(self, x, y, **kwargs):
        """
        Bivariate analysis between two variables (scatter plot + regression line).

        Parameters
        ----------
        x : str
            First variable.
        y : str
            Second variable.
        **kwargs :
            Additional arguments for plotting.

        Returns
        -------
        matplotlib.figure.Figure
            The generated bivariate plot.

        Examples
        --------
        >>> fig = analyzer.bivariate(x='A', y='B')
        >>> fig.show()
        """
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        x_data = self.data[x].dropna()
        y_data = self.data[y].dropna()
        # Align indices
        common = x_data.index.intersection(y_data.index)
        x_data = x_data.loc[common]
        y_data = y_data.loc[common]

        ax.scatter(x_data, y_data, alpha=0.6, edgecolors='k', linewidths=0.5)

        # Linear regression line
        if np.issubdtype(x_data.dtype, np.number) and np.issubdtype(y_data.dtype, np.number):
            slope, intercept, r_value, _, _ = stats.linregress(x_data, y_data)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, 'r--',
                    label=f'y={slope:.3f}x+{intercept:.3f} (R²={r_value**2:.3f})')
            ax.legend()

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f'{x} vs {y}')
        fig.tight_layout()
        return fig

    def normality(self, columns='all', **kwargs):
        """
        Test normality of variables (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling).

        Parameters
        ----------
        columns : 'all' or list, default='all'
            Columns to test.
        **kwargs :
            Additional arguments.

        Returns
        -------
        dict
            Normality test results per column with p-values and statistics.

        Examples
        --------
        >>> norm = analyzer.normality()
        >>> print(norm)
        """
        df_num = self._select_numeric(columns)
        results = {}

        for col in df_num.columns:
            x = df_num[col].dropna()
            col_result = {}

            # Shapiro-Wilk (max 5000 samples)
            sample = x if len(x) <= 5000 else x.sample(5000, random_state=42)
            stat_sw, p_sw = stats.shapiro(sample)
            col_result['shapiro'] = {'statistic': float(stat_sw), 'p_value': float(p_sw)}

            # D'Agostino-Pearson (requires n >= 20)
            if len(x) >= 20:
                stat_da, p_da = stats.normaltest(x)
                col_result['dagostino'] = {'statistic': float(stat_da), 'p_value': float(p_da)}
            else:
                col_result['dagostino'] = {'statistic': None, 'p_value': None}

            # Anderson-Darling
            ad_result = stats.anderson(x, dist='norm', method='interpolate')
            col_result['anderson'] = {
                'statistic': float(ad_result.statistic),
                'p_value': float(ad_result.pvalue),
            }

            # Skewness & kurtosis
            col_result['skewness'] = float(x.skew())
            col_result['kurtosis'] = float(x.kurtosis())

            results[col] = col_result

        return results

    def multicollinearity(self, **kwargs):
        """
        Analyze multicollinearity using Variance Inflation Factor (VIF).

        Returns
        -------
        pandas.DataFrame
            VIF per feature with columns 'feature' and 'VIF'.

        Notes
        -----
        $VIF_j = \frac{1}{1 - R_j^2}$
        where $R_j^2$ is the $R^2$ of regressing feature $j$ on all others.

        Examples
        --------
        >>> vif = analyzer.multicollinearity()
        >>> print(vif)
        """
        df_num = self._select_numeric().dropna()
        if df_num.shape[1] < 2:
            return pd.DataFrame(columns=['feature', 'VIF'])

        X = df_num.values
        vif_data = pd.DataFrame({
            'feature': df_num.columns,
            'VIF': [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        })
        return vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

    def profiling(self, **kwargs):
        """
        Generate a global profiling report (summary statistics, missing, outliers, etc.).

        Returns
        -------
        dict
            Profiling report with keys 'describe', 'dtypes', 'missing', 'outliers',
            'correlation', 'shape'.

        Examples
        --------
        >>> report = analyzer.profiling()
        >>> print(report['describe'])
        """
        return {
            'shape': self.data.shape,
            'dtypes': self.data.dtypes,
            'describe': self.data.describe(include='all'),
            'missing': self.missing(),
            'outliers': self.outliers(),
            'correlation': self.correlation(),
        }
