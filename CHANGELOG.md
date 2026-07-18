# Changelog

Toutes les évolutions notables de ce projet sont documentées ici.
Le format suit [Keep a Changelog](https://keepachangelog.com/fr/) et le projet
adhère au [versionnement sémantique](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté
- API web FastAPI (`webapp_api/`) : routes /api/train, /api/predict,
  /api/compare, /api/models, avec page de démo HTML/JS servie à la racine
  et documentation interactive /docs. Extra d'installation : `trainedml[web]`.
- Tests de l'API (`tests/test_api.py`), ignorés si fastapi absent.
- Fichiers de déploiement : `render.yaml` (Render, gratuit), `Dockerfile`
  et `.dockerignore` (tout hébergeur Docker) ; marche à suivre dans
  webapp_api/README.md.
- Un README de documentation dans chaque dossier du projet ;
  DOC_UTILISATION.md restructuré avec journal du projet.

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
