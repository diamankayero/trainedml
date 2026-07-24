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
| `analyzer.py` | `DataAnalyzer` : seule source de vérité pour les analyses exploratoires (distribution, corrélation, missing, outliers, target, boxplot, bivariate, normalité, VIF, profiling) |
| `visualization.py` | `Visualizer` : façade unique de toutes les visualisations et analyses, dont `report()` ; délègue les stats à `DataAnalyzer` et les graphiques à `viz/*` |
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
- Type hints sur toute l'API publique ; mypy doit passer (`mypy` à la racine, aucune exclusion).
- Détection de tâche : toujours passer par `tasks.py`, jamais de duplication locale.
- Une seule source de vérité par analyse : les stats vivent dans `DataAnalyzer`
  (`analyzer.py`), jamais dupliquées dans `viz/*` ou `report.py`. Les classes
  de `viz/` ne font que dessiner un graphique à partir de ce que
  `DataAnalyzer` calcule (voir `viz/README.md`).
- Pas de tirets longs dans les textes ; ponctuation simple.
