# src/trainedml : le package

Cœur du package. Chaque module a une responsabilité unique ; le point d'entrée
utilisateur est la classe `Trainer` et la fonction `compare()`, exposées dans
`__init__.py`.

## Modules

| Module | Rôle |
|---|---|
| `__init__.py` | API principale : classe `Trainer` (fit, evaluate, predict, save/load), exports publics, `__version__` |
| `compare.py` | `compare()` : comparaison de tous les modèles adaptés en une ligne, validation croisée, retour DataFrame trié |
| `preprocessing.py` | Préprocesseur standard (imputation, standardisation, one-hot) + `PreprocessedModel` (enveloppe sans fuite en CV) |
| `tasks.py` | Détection du type de tâche (classification vs régression), partagée par tout le package |
| `benchmark.py` | `Benchmark` : comparaison de modèles sur un split (`run`) ou en CV (`run_cv`), export `to_dataframe()` |
| `evaluation.py` | `Evaluator` : métriques de classification et de régression, routage par tâche (`evaluate_for`) |
| `report.py` | Rapport EDA HTML auto-contenu (`generate_report`), figures embarquées en base64 |
| `cli.py` | Interface en ligne de commande : entraînement, benchmark, visualisation, sauvegarde, prédiction sur CSV |
| `analyzer.py` | `DataAnalyzer` : analyses exploratoires (distribution, corrélation, outliers, normalité, VIF, profiling) |
| `visualization.py` | `Visualizer` : façade unique de toutes les visualisations et analyses, dont `report()` |
| `figure.py` | Encapsulation de figures multi-backend (matplotlib, plotly) |
| `data/` | Chargement des données (voir son README) |
| `models/` | Modèles ML (voir son README) |
| `viz/` | Visualisations spécialisées (voir son README) |

## Flux typique

```
DataLoader (data/) -> split -> preprocessing -> models -> Evaluator
                                     |                        |
                                  Trainer  ------------- evaluate/predict/save
compare() = DataLoader + PreprocessedModel + Benchmark.run_cv + to_dataframe
```

## Conventions

- Docstrings NumPy-style, avec formules mathématiques pour Sphinx quand pertinent.
- Type hints sur toute l'API publique ; mypy doit passer (`mypy` à la racine).
- Les modules historiques de visualisation (`viz/`, `figure.py`, `analyzer.py`,
  `visualization.py`) sont exclus de mypy (voir `[tool.mypy]` dans pyproject.toml),
  à annoter progressivement.
- Détection de tâche : toujours passer par `tasks.py`, jamais de duplication locale.
- Pas de tirets longs dans les textes ; ponctuation simple.
