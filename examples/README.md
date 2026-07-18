# examples : exemples exécutables

Tous les scripts se lancent depuis la racine du projet et n'ont besoin que du
package installé (`pip install trainedml` ou `pip install -e .`).

## Scripts

| Script | Ce qu'il montre |
|---|---|
| `quickstart.py` | Workflow complet du Trainer : fit, evaluate, variation de seed, predict, save/load |
| `compare_models.py` | `compare()` sur Wine, puis avec des modèles personnalisés (dont un SVC scikit-learn) |
| `exemple_regression.py` | Régression avec données en mémoire (X, y), métriques adaptées, comparatif des régresseurs en CV |
| `rapport_eda.py` | Génération d'un rapport EDA HTML auto-contenu sur Iris |

```bash
python examples/quickstart.py
```

## Notebooks (`notebooks/`)

| Notebook | Contenu |
|---|---|
| `01_quickstart.ipynb` | Prise en main pas à pas : Trainer, seeds, estimateurs sklearn, persistance, données en mémoire |
| `02_comparaison_et_eda.ipynb` | `compare()` (classification, régression, modèles personnalisés) et rapport EDA |

Les notebooks sont validés en CI locale en exécutant toutes leurs cellules de
code ; si vous les modifiez, vérifiez qu'ils s'exécutent de bout en bout.
