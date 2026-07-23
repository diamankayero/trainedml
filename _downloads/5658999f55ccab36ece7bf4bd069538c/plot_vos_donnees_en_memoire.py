"""
Vos données en mémoire
=========================

Entraîner directement sur un DataFrame maison (pas un dataset intégré, pas
une URL), avec des colonnes catégorielles et des valeurs manquantes : le
prétraitement automatique s'en charge.
"""

import numpy as np
import pandas as pd

from trainedml import Trainer

# %%
# Un jeu de données synthétique de résiliation client (churn), avec des
# colonnes numériques, catégorielles, et quelques valeurs manquantes
# réalistes, comme un export CRM typique.

rng = np.random.default_rng(42)
n = 400
anciennete = rng.integers(1, 72, n).astype(float)
mensualite = rng.normal(70, 20, n).round(2)
type_contrat = rng.choice(["mensuel", "un_an", "deux_ans"], n, p=[0.5, 0.3, 0.2])
support_technique = rng.choice(["oui", "non"], n, p=[0.3, 0.7])

score_risque = (
    (type_contrat == "mensuel").astype(float) * 0.4
    + (support_technique == "non").astype(float) * 0.2
    + (anciennete < 12).astype(float) * 0.3
    + rng.normal(0, 0.15, n)
)
churn = np.where(score_risque > np.quantile(score_risque, 0.7), "oui", "non")

df = pd.DataFrame({
    "anciennete_mois": anciennete,
    "mensualite": mensualite,
    "type_contrat": type_contrat,
    "support_technique": support_technique,
    "churn": churn,
})
df.loc[rng.random(n) < 0.05, "mensualite"] = np.nan

X = df.drop(columns=["churn"])
y = df["churn"]
print(X.isnull().sum())

# %%
# Prétraitement automatique
# -----------------------------
# ``preprocess=True`` (par défaut) impute les valeurs manquantes puis
# encode les colonnes catégorielles en one-hot avant d'entraîner : aucune
# préparation manuelle n'est nécessaire.

trainer = Trainer(X=X, y=y, model="random_forest", seed=42)
trainer.fit()
print(trainer.evaluate())

# %%
# Importance des variables
# -----------------------------
# ``trainer.model`` est le modèle trainedml ; l'estimateur scikit-learn
# sous-jacent (qui porte ``feature_importances_``) est ``trainer.model.model``.

X_num = pd.get_dummies(X, columns=["type_contrat", "support_technique"], drop_first=True)
X_num["mensualite"] = X_num["mensualite"].fillna(X_num["mensualite"].median())

trainer_num = Trainer(X=X_num, y=y, model="random_forest", seed=42, preprocess=False)
trainer_num.fit()
importances = pd.Series(
    trainer_num.model.model.feature_importances_, index=X_num.columns
).sort_values(ascending=False)
print(importances)
