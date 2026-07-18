# models : les modèles ML

## Structure

- `base.py` : `BaseModel` (ABC, task='classification') et `BaseRegressor`
  (task='regression'). Interface imposée : `fit`, `predict`, `evaluate`.
  L'attribut `task` sert au routage automatique des métriques.
- `knn.py`, `logistic.py`, `random_forest.py` : classificateurs.
- `regressors.py` : régresseurs (Linear, Ridge, Lasso, KNN, RandomForest).
- `__init__.py` : registres et factories.

## Registres

- `CLASSIFIER_MAP` : nom -> classe de classificateur
- `REGRESSOR_MAP` : nom -> classe de régresseur
- `MODEL_MAP` : union des deux

Factories : `get_model(name, **kwargs)`, `get_classifier`, `get_regressor`.
C'est LE point d'entrée unique ; les anciennes factories dupliquées
(`models/factory.py`, `utils/`) ont été supprimées en 0.2.0.

## Ajouter un modèle

1. Créer une classe héritant de `BaseModel` ou `BaseRegressor` qui enveloppe
   un estimateur scikit-learn dans `self.model`.
2. L'ajouter au registre correspondant dans `__init__.py` (et à `__all__`).
3. Ajouter un test dans `tests/`.

Note : le `Trainer` accepte aussi n'importe quel estimateur scikit-learn
directement (`Trainer(model=SVC())`), sans passer par un registre.
