# tests : les tests unitaires

78 tests en `unittest`, exécutables aussi avec pytest (utilisé par la CI).

```bash
pytest tests/            # ou
python -m unittest discover tests
```

## Organisation

| Fichier | Ce qu'il couvre |
|---|---|
| `test_trainer.py` | Trainer : classification, régression, model_params, estimateur sklearn arbitraire, prétraitement (NaN, catégorielles), save/load, re-split par seed |
| `test_compare.py` | `compare()` (classification, régression, modèles personnalisés) et `Benchmark.run_cv` / `to_dataframe` |
| `test_report.py` | Rapport EDA HTML (contenu, écriture fichier, façade Visualizer) |
| `test_cli.py` | Détection de tâche et exécution du CLI de bout en bout |
| `test_benchmark.py` | Benchmark sur split simple |
| `test_evaluation.py` | Métriques de classification et de régression |
| `test_knn.py`, `test_logistic.py`, `test_random_forest.py`, `test_regressors.py` | Chaque modèle individuellement |
| `test_analyzer.py`, `test_heatmap.py`, `test_histogram.py`, `test_line.py`, `test_multicollinearity.py`, `test_normality.py`, `test_outliers.py`, `test_profiling.py` | Analyses et visualisations |

## Règles

- Aucun test ne doit dépendre du réseau : les datasets intégrés (iris, wine)
  sont chargés localement via scikit-learn.
- Le split se fait via `DataLoader.split`, jamais via scikit-learn directement
  (cohérence avec l'API du package).
- Toute correction de bug s'accompagne d'un test de non-régression
  (exemple : `test_heatmap_all_with_mixed_columns`).
- Les tests de figures utilisent le backend matplotlib `Agg` quand nécessaire.
