# Changelog

Toutes les évolutions notables de ce projet sont documentées ici.
Le format suit [Keep a Changelog](https://keepachangelog.com/fr/) et le projet
adhère au [versionnement sémantique](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté
- Un README de documentation dans chaque dossier du projet ;
  DOC_UTILISATION.md restructuré avec journal du projet.
- Galerie d'exemples dans la doc (Sphinx-Gallery, façon
  scikit-learn/matplotlib) : onze scripts `examples/*/plot_*.py`, organisés
  en trois sections (Bases, Données et modèles, Production), réellement
  exécutés au moment du build avec sorties et graphiques inclus, et
  téléchargeables en `.py` ou en notebook `.ipynb`.

### Modifié
- La démo web (API FastAPI + page HTML/JS, déployée sur
  https://trainedml.onrender.com) vit désormais dans son propre dépôt
  https://github.com/diamankayero/trainedml-webapp, qui consomme trainedml
  depuis PyPI. Ce dépôt ne contient plus que le package ; l'extra
  d'installation `[web]`, render.yaml et le Dockerfile sont partis avec elle.
- Docstrings, commentaires et sorties CLI entièrement en anglais (le mélange
  FR/EN hérité des phases précédentes est résorbé).
- `figure.py`, `analyzer.py`, `visualization.py` et tout `trainedml.viz.*`
  sont désormais annotés en type hints ; l'exclusion mypy correspondante a
  été retirée de `pyproject.toml`.
- Doc Sphinx : thème Shibuya (remplace sphinx_rtd_theme), version affichée
  lue depuis pyproject.toml, labels de section préfixés par document.
- Page d'accueil de la doc refondue : installation depuis PyPI (l'ancienne
  page documentait une installation depuis les sources), badges réels
  (PyPI, Python, CI, licence), logo monogramme SVG, liens vers l'écosystème
  (démo web, ModeLmL), et navigation latérale groupée en trois sections
  (cœur, modèles, exploration) au lieu d'une liste plate de 26 pages.
- Identité visuelle verte (inspiration NVIDIA, #76B900) : icône monogramme
  Tm et mot-symbole trainedml dessinés en SVG monoligne (aucune police
  requise), bannière en tête du README (visible sur GitHub et PyPI),
  accent vert et lien GitHub dans l'en-tête de la doc, images du README
  en URLs absolues pour un rendu correct sur PyPI.
- Build de la doc sans aucun warning : sections Methods/Attributes
  redondantes retirées des docstrings (elles dupliquaient les entrées
  générées par autodoc).

### Retiré
- Les scripts historiques à plat `examples/quickstart.py`,
  `compare_models.py`, `exemple_regression.py`, `rapport_eda.py`, ainsi que
  `examples/notebooks/`, remplacés par la galerie d'exemples ci-dessus
  qu'ils recoupaient entièrement.

### Corrigé
- `Visualizer.multicollinearity()` ne transmet plus de `**kwargs` vers
  `DataAnalyzer.multicollinearity()`, qui n'en accepte pas.
- `viz/boxplot.py` et `viz/correlation.py` : `from __future__ import
  annotations` était placé avant le docstring de module, ce qui le
  désactivait silencieusement.
- `viz/normality.py` : docstring de module dupliqué et mort supprimé.
- Docstrings LaTeX non-raw (`evaluation.py`, `models/logistic.py`,
  `viz/multicollinearity.py`) : `\f` et `\b` y étaient interprétés comme
  séquences d'échappement Python et corrompaient le rendu Sphinx.
- Formules `$...$` remplacées par le rôle reST `:math:` (la syntaxe dollar
  n'est pas du reST et s'affichait littéralement dans la doc), et lignes
  vides ajoutées entre les intros et les blocs `>>>` pour que Sphinx les
  rende en blocs doctest colorisés au lieu de texte courant.

## [0.2.0] - 2026-07-18

### Ajouté
- `trainedml.compare()` : comparaison de tous les modèles adaptés à un dataset
  en une ligne, avec validation croisée, retournant un DataFrame trié.
- `Benchmark.run_cv()` : benchmark par validation croisée K-fold (stratifiée en
  classification), avec moyennes et écarts-types des métriques.
- `Benchmark.to_dataframe()` : résultats du benchmark en DataFrame pandas trié.
- Prétraitement automatique (`trainedml.preprocessing`) : imputation,
  standardisation des colonnes numériques, encodage one-hot des colonnes
  catégorielles. Activé par défaut dans `Trainer` (`preprocess=True`) et
  `compare()` ; réentraîné à chaque pli en CV (aucune fuite d'information).
- `Trainer` accepte des hyperparamètres (`model_params={"n_neighbors": 7}`)
  et **n'importe quel estimateur scikit-learn** (`model=SVC()`).
- `Trainer.save()` / `Trainer.load()` : persistance du modèle entraîné et de
  son préprocesseur (joblib).
- `Trainer.fit(seed=..., test_size=...)` et `load_data(seed=...)` : re-split
  sans recréer l'objet.
- CLI : `--cv N` (benchmark par validation croisée), `--save`, `--load`,
  `--input`, `--output` (prédiction sur un CSV avec un modèle sauvegardé).
- Rapport EDA HTML auto-contenu : `Visualizer.report("rapport.html")` /
  `trainedml.report.generate_report`.
- `trainedml.tasks` : détection centralisée du type de tâche
  (classification/régression), partagée par Trainer, CLI, Benchmark et compare.
- `trainedml.__version__`.
- Type hints sur toute l'API publique (Trainer, DataLoader, Evaluator,
  Benchmark, modèles) + configuration mypy (`[tool.mypy]` dans pyproject).
- Notebooks d'exemples exécutables dans `examples/notebooks/`
  (quickstart, comparaison de modèles + rapport EDA).
- CI : matrice Python 3.9→3.13 + Windows, lint ruff, mypy, tests bloquants ;
  publication PyPI uniquement sur tag `v*`.

### Modifié
- Datasets intégrés (iris, wine) chargés **localement** via `sklearn.datasets` :
  plus aucun accès réseau ni cache nécessaire (mêmes noms de colonnes qu'avant).
- La CLI passe par `DataLoader.split` (API du package) au lieu d'appeler
  scikit-learn directement.
- `Trainer.evaluate()` et `Benchmark` choisissent les métriques selon la tâche :
  classification (accuracy, precision, recall, f1) ou régression (r2, mse,
  rmse, mae).

### Corrigé
- `Trainer.evaluate()` appliquait des métriques de classification aux
  régresseurs.
- Détection de tâche et sélection des colonnes numériques compatibles avec les
  pandas récents (`StringDtype`).
- Un nouveau split (`load_data`) invalide désormais l'entraînement précédent
  (`is_fitted=False`).
- `Visualizer.heatmap()` / `HeatmapViz` avec `features='all'` plantait sur un
  DataFrame contenant des colonnes non numériques ; elles sont désormais
  ignorées (la corrélation n'est définie que pour les colonnes numériques).

### Supprimé
- Factories redondantes `trainedml.models.factory` et `trainedml.utils.factory`
  (utiliser `trainedml.models.get_model`).
- Fichiers `__pycache__` retirés du suivi git.

## [0.1.4] - 2026-07

- Version publiée sur PyPI (correctif de déploiement).
