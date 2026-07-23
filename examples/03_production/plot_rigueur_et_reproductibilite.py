"""
Rigueur et reproductibilité
==============================

Un score parfait est un signal d'alarme, pas une victoire. Cet exemple
mesure la variance d'un score selon la graine de découpage, puis reproduit
une fuite de données volontaire pour en montrer l'effet.
"""

import statistics

import pandas as pd

from trainedml import Trainer
from trainedml.data.loader import DataLoader

# %%
# Variance selon la graine
# -----------------------------
# Un score unique, sur une seule graine, ne suffit pas à juger un modèle :
# il dépend en partie du hasard du découpage train/test.

scores = []
for seed in range(10):
    t = Trainer(dataset="wine", model="knn", seed=seed)
    t.fit()
    scores.append(t.evaluate()["accuracy"])

print("Scores sur 10 graines :", [round(s, 3) for s in scores])
print(f"Min={min(scores):.3f}  Max={max(scores):.3f}  "
      f"Ecart-type={statistics.pstdev(scores):.3f}")

# %%
# Une fuite de données par duplication
# -----------------------------------------
# Dupliquer des lignes avant le split train/test (au lieu d'après, ou
# jamais) fait fuiter des jumeaux exacts entre train et test : avec
# ``n_neighbors=1``, un point de test retrouve alors son propre jumeau
# comme plus proche voisin, sans avoir rien appris de généralisable.

X, y = DataLoader().load_dataset(name="wine")

t_sans = Trainer(X=X, y=y, model="knn", model_params={"n_neighbors": 1}, seed=42)
t_sans.fit()
print(f"\nSans duplication : {t_sans.evaluate()['accuracy']:.3f}")

X_leak = pd.concat([X, X], ignore_index=True)
y_leak = pd.concat([y, y], ignore_index=True)
t_avec = Trainer(X=X_leak, y=y_leak, model="knn", model_params={"n_neighbors": 1}, seed=42)
t_avec.fit()
print(f"Avec duplication : {t_avec.evaluate()['accuracy']:.3f}")

# %%
# Règle à retenir
# --------------------
# Toute transformation qui "regarde" les données (augmentation, imputation,
# standardisation, sélection de variables sur leur lien avec la cible) doit
# être calculée après le split, et seulement à partir du train. C'est
# exactement ce que fait le prétraitement automatique de trainedml.
