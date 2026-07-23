"""
Self-contained HTML EDA report generation for trainedml.

This module assembles the package's exploratory analyses (descriptive
statistics, missing values, correlations, distributions, outliers,
normality, multicollinearity) into a **self-contained** HTML report:
matplotlib figures are embedded as base64, the file opens in any browser
with no dependency.

Entry point: :func:`generate_report`, also accessible via
:meth:`trainedml.visualization.Visualizer.report`.

Example
-------
>>> from trainedml.report import generate_report
>>> generate_report(df, "report.html", title="My dataset")
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import pandas as pd


_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 2rem auto;
       max-width: 1100px; padding: 0 1rem; color: #1a1a2e; }
h1 { border-bottom: 3px solid #4361ee; padding-bottom: .4rem; }
h2 { color: #4361ee; margin-top: 2.2rem; }
table { border-collapse: collapse; font-size: .85rem; margin: .8rem 0; }
th, td { border: 1px solid #d0d0e0; padding: .35rem .6rem; text-align: right; }
th { background: #eef0fb; }
img { max-width: 100%; height: auto; border: 1px solid #e0e0ea; border-radius: 6px; }
.note { color: #777; font-style: italic; }
.meta { color: #555; font-size: .9rem; }
"""


def _fig_to_html(fig) -> str:
    """Convert a matplotlib figure into a self-contained base64 <img> tag."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="figure"/>'


def _safe_section(title: str, builder) -> str:
    """Build an HTML section; on error, insert a note instead of crashing."""
    try:
        body = builder()
    except Exception as e:  # the report must still be generated even if an analysis fails
        body = f'<p class="note">Section unavailable: {e}</p>'
    return f"<h2>{title}</h2>\n{body}"


def generate_report(data: pd.DataFrame, path: Optional[str] = None,
                    title: str = "Exploratory report - trainedml") -> str:
    """
    Generate a complete, self-contained HTML EDA report for a DataFrame.

    The report contains: dataset overview, descriptive statistics, missing
    values, correlation matrix and heatmap, distributions, outliers (IQR),
    normality tests, and variance inflation factors (VIF).

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to analyze.
    path : str, optional
        Output HTML file path. If None, the HTML is only returned.
    title : str, default="Exploratory report - trainedml"
        Report title.

    Returns
    -------
    str
        The report's HTML content.

    Examples
    --------
    >>> from trainedml.report import generate_report
    >>> html = generate_report(df, "report.html")
    """
    from .analyzer import DataAnalyzer
    from .viz.heatmap import HeatmapViz
    from .viz.missing import missing_summary
    from .viz.multicollinearity import vif_summary
    from .viz.normality import normality_tests
    from .viz.outliers import outlier_summary

    analyzer = DataAnalyzer(data)
    numeric = data.select_dtypes(include=[np.number])
    sections = []

    # --- Overview ---
    def overview():
        dtypes = data.dtypes.astype(str).rename("dtype").to_frame()
        return (
            f'<p class="meta">{data.shape[0]} rows x {data.shape[1]} columns - '
            f'{data.duplicated().sum()} duplicate(s)</p>'
            + dtypes.to_html()
            + "<h3>First rows</h3>" + data.head(10).to_html()
        )
    sections.append(_safe_section("Dataset overview", overview))

    # --- Descriptive statistics ---
    sections.append(_safe_section(
        "Descriptive statistics", lambda: data.describe().round(3).to_html()))

    # --- Missing values ---
    def missing():
        miss = missing_summary(data).rename("missing values").to_frame()
        miss["%"] = (miss["missing values"] / len(data) * 100).round(2)
        return miss.to_html()
    sections.append(_safe_section("Missing values", missing))

    # --- Correlation ---
    def correlation():
        viz = HeatmapViz(numeric, features=list(numeric.columns))
        viz.vizs()
        return numeric.corr().round(3).to_html() + _fig_to_html(viz.figure)
    if numeric.shape[1] >= 2:
        sections.append(_safe_section("Correlations", correlation))

    # --- Distributions ---
    def distributions():
        result = analyzer.distribution()
        return _fig_to_html(result["figure"]) if result["figure"] is not None else ""
    if not numeric.empty:
        sections.append(_safe_section("Distributions", distributions))

    # --- Outliers ---
    def outliers():
        summary = outlier_summary(data)
        return pd.DataFrame(summary).T.to_html()
    if not numeric.empty:
        sections.append(_safe_section("Outliers (IQR method)", outliers))

    # --- Normality ---
    def normality():
        results = normality_tests(numeric, columns=list(numeric.columns))
        rows = {}
        for col, res in results.items():
            stat, pval = res["shapiro"]
            rows[col] = {"shapiro_stat": round(float(stat), 4),
                         "p_value": round(float(pval), 4),
                         "normal (alpha=5%)": "yes" if pval > 0.05 else "no"}
        return pd.DataFrame(rows).T.to_html()
    if not numeric.empty:
        sections.append(_safe_section("Normality tests (Shapiro-Wilk)", normality))

    # --- Multicollinearity ---
    def vif():
        return vif_summary(numeric).round(2).rename("VIF").to_frame().to_html()
    if numeric.shape[1] >= 2:
        sections.append(_safe_section("Multicollinearity (VIF)", vif))

    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_CSS}</style></head><body>"
        f"<h1>{title}</h1>"
        f'<p class="meta">Generated by trainedml - pandas {pd.__version__}</p>'
        + "\n".join(sections)
        + "</body></html>"
    )

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    return html
