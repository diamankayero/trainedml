# viz : visualisations spécialisées

Chaque visualisation est une classe héritant de `Vizs` (`vizs.py`), avec une
méthode `vizs()` qui construit la figure matplotlib (accessible ensuite via
`.figure`). Certains modules exposent aussi une fonction d'analyse pure qui
retourne des objets pandas.

| Module | Classe | Fonction d'analyse |
|---|---|---|
| `heatmap.py` | `HeatmapViz` | - |
| `histogram.py` | `HistogramViz` | - |
| `line.py` | `LineViz` | - |
| `boxplot.py` | `BoxplotViz` | - |
| `bivariate.py` | `BivariateViz` | - |
| `distribution.py` | `DistributionViz` | `distribution_summary` |
| `correlation.py` | `CorrelationViz` | `correlation_matrix` |
| `missing.py` | `MissingValuesViz` | `missing_summary` |
| `outliers.py` | `OutliersViz` | `outlier_summary` (IQR ou z-score) |
| `normality.py` | `NormalityViz` | `normality_tests` (Shapiro, D'Agostino, Anderson) |
| `multicollinearity.py` | `MulticollinearityViz` | `vif_summary` |
| `target.py` | `TargetViz` | - |
| `profiling.py` | `ProfilingViz` | `profiling_report` |

## Points d'attention

- L'accès utilisateur recommandé est la façade `Visualizer`
  (`src/trainedml/visualization.py`), pas ces classes directement.
- `HeatmapViz` avec `features='all'` ne retient que les colonnes numériques
  (corrigé en 0.2.0 : plantait sur les colonnes texte).
- Compatibilité anciennes versions gérée par des replis try/except :
  `stats.anderson` sans `method` (scipy ancien), `boxplot(vert=False)`
  (matplotlib < 3.10). Conserver ce pattern pour tout nouvel usage d'API
  récente de scipy/matplotlib.
- Ces modules ne sont pas encore typés (exclus de mypy) ; toute contribution
  d'annotation est bienvenue, penser à retirer l'exclusion dans pyproject.toml.
