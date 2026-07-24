# viz : visualisations spécialisées

Chaque visualisation est une classe héritant de `Vizs` (`vizs.py`), avec une
méthode `vizs()` qui construit la figure matplotlib (accessible ensuite via
`.figure`). Certains modules exposent aussi une fonction d'analyse pure qui
retourne des objets pandas. `confusion.py` et `roc.py` sortent de ce moule :
ce sont de simples fonctions (`plot_confusion_matrix`, `plot_roc_curve`)
opérant sur des prédictions (`y_true`/`y_pred`/`y_score`), pas sur un
DataFrame de features.

| Module | Classe | Fonction d'analyse |
|---|---|---|
| `heatmap.py` | `HeatmapViz` | - |
| `histogram.py` | `HistogramViz` | - |
| `line.py` | `LineViz` | - |
| `missing.py` | `MissingValuesViz` | `missing_summary` |
| `outliers.py` | `OutliersViz` | `outlier_summary` (IQR ou z-score) |
| `normality.py` | `NormalityViz` | `normality_tests` (Shapiro, D'Agostino, Anderson) |
| `multicollinearity.py` | `MulticollinearityViz` | `vif_summary` |
| `target.py` | `TargetViz` | - |
| `confusion.py` | - | `plot_confusion_matrix` |
| `roc.py` | - | `plot_roc_curve` |

## Points d'attention

- L'accès utilisateur recommandé est la façade `Visualizer`
  (`src/trainedml/visualization.py`), pas ces classes directement.
- Toutes les stats affichées par `Visualizer` (missing, outliers, normality,
  multicollinearity, target) sont calculées par `DataAnalyzer`
  (`src/trainedml/analyzer.py`), la seule source de vérité pour ces
  analyses ; les fonctions `*_summary`/`*_tests`/`vif_summary` de ce dossier
  sont une seconde implémentation indépendante, utilisée uniquement en
  interne par certains modules (ex. `report.py` utilisait ces fonctions
  avant consolidation ; il appelle désormais `DataAnalyzer` lui aussi). Ne
  pas ajouter de nouvelle logique de calcul ici : les classes `*Viz` de ce
  dossier ne doivent que dessiner un graphique à partir de ce que
  `DataAnalyzer` calcule.
- `HeatmapViz` avec `features='all'` ne retient que les colonnes numériques
  (corrigé en 0.2.0 : plantait sur les colonnes texte).
- Compatibilité anciennes versions gérée par des replis try/except :
  `stats.anderson` sans `method` (scipy ancien), `boxplot(vert=False)`
  (matplotlib < 3.10). Conserver ce pattern pour tout nouvel usage d'API
  récente de scipy/matplotlib.
- `boxplot.py`, `bivariate.py`, `distribution.py`, `correlation.py` et
  `profiling.py` ont été retirés (2026-07-24) : ils dupliquaient
  `DataAnalyzer.boxplot/bivariate/distribution/correlation/profiling` sans
  jamais être utilisés par `Visualizer`, avec des formes de retour
  divergentes de surcroît (risque de bug latent). `DataAnalyzer` reste seul
  responsable de ces analyses.
