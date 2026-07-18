"""
Génération de rapport EDA HTML autonome pour trainedml.

Ce module assemble les analyses exploratoires du package (statistiques
descriptives, valeurs manquantes, corrélations, distributions, outliers,
normalité, multicolinéarité) en un rapport HTML **auto-contenu** : les
figures matplotlib sont embarquées en base64, le fichier s'ouvre dans
n'importe quel navigateur sans dépendance.

Point d'entrée : :func:`generate_report`, aussi accessible via
:meth:`trainedml.visualization.Visualizer.report`.

Exemple
-------
>>> from trainedml.report import generate_report
>>> generate_report(df, "rapport.html", title="Mon dataset")
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
    """Convertit une figure matplotlib en balise <img> base64 auto-contenue."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="figure"/>'


def _safe_section(title: str, builder) -> str:
    """Construit une section HTML ; en cas d'erreur, insère une note au lieu de planter."""
    try:
        body = builder()
    except Exception as e:  # le rapport doit rester généré même si une analyse échoue
        body = f'<p class="note">Section indisponible : {e}</p>'
    return f"<h2>{title}</h2>\n{body}"


def generate_report(data: pd.DataFrame, path: Optional[str] = None,
                    title: str = "Rapport exploratoire - trainedml") -> str:
    """
    Génère un rapport EDA HTML complet et auto-contenu pour un DataFrame.

    Le rapport contient : aperçu du dataset, statistiques descriptives,
    valeurs manquantes, matrice et heatmap de corrélation, distributions,
    outliers (IQR), tests de normalité et facteurs d'inflation de la
    variance (VIF).

    Parameters
    ----------
    data : pandas.DataFrame
        Le dataset à analyser.
    path : str, optional
        Chemin du fichier HTML de sortie. Si None, le HTML est seulement retourné.
    title : str, default="Rapport exploratoire - trainedml"
        Titre du rapport.

    Returns
    -------
    str
        Le contenu HTML du rapport.

    Examples
    --------
    >>> from trainedml.report import generate_report
    >>> html = generate_report(df, "rapport.html")
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

    # --- Aperçu ---
    def overview():
        dtypes = data.dtypes.astype(str).rename("dtype").to_frame()
        return (
            f'<p class="meta">{data.shape[0]} lignes × {data.shape[1]} colonnes - '
            f'{data.duplicated().sum()} doublon(s)</p>'
            + dtypes.to_html()
            + "<h3>Premières lignes</h3>" + data.head(10).to_html()
        )
    sections.append(_safe_section("Aperçu du dataset", overview))

    # --- Statistiques descriptives ---
    sections.append(_safe_section(
        "Statistiques descriptives", lambda: data.describe().round(3).to_html()))

    # --- Valeurs manquantes ---
    def missing():
        miss = missing_summary(data).rename("valeurs manquantes").to_frame()
        miss["%"] = (miss["valeurs manquantes"] / len(data) * 100).round(2)
        return miss.to_html()
    sections.append(_safe_section("Valeurs manquantes", missing))

    # --- Corrélation ---
    def correlation():
        viz = HeatmapViz(numeric, features=list(numeric.columns))
        viz.vizs()
        return numeric.corr().round(3).to_html() + _fig_to_html(viz.figure)
    if numeric.shape[1] >= 2:
        sections.append(_safe_section("Corrélations", correlation))

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
        sections.append(_safe_section("Outliers (méthode IQR)", outliers))

    # --- Normalité ---
    def normality():
        results = normality_tests(numeric, columns=list(numeric.columns))
        rows = {}
        for col, res in results.items():
            stat, pval = res["shapiro"]
            rows[col] = {"shapiro_stat": round(float(stat), 4),
                         "p_value": round(float(pval), 4),
                         "normale (α=5%)": "oui" if pval > 0.05 else "non"}
        return pd.DataFrame(rows).T.to_html()
    if not numeric.empty:
        sections.append(_safe_section("Tests de normalité (Shapiro-Wilk)", normality))

    # --- Multicolinéarité ---
    def vif():
        return vif_summary(numeric).round(2).rename("VIF").to_frame().to_html()
    if numeric.shape[1] >= 2:
        sections.append(_safe_section("Multicolinéarité (VIF)", vif))

    html = (
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_CSS}</style></head><body>"
        f"<h1>{title}</h1>"
        f'<p class="meta">Généré par trainedml - pandas {pd.__version__}</p>'
        + "\n".join(sections)
        + "</body></html>"
    )

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    return html
