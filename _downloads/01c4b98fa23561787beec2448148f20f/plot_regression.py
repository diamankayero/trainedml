"""
Régression : estimer une note continue
=========================================

Forcer explicitement un modèle de régression, lire ses métriques (R², MSE,
RMSE, MAE), et comparer un modèle linéaire à un modèle non linéaire.
"""

from trainedml.data.loader import DataLoader
from trainedml import Trainer

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
X, y = DataLoader().load_dataset(url=url, target="quality")

trainer = Trainer(X=X, y=y, model="random_forest_regressor", seed=42)
trainer.fit()
print("task détecté :", trainer.task)
print(trainer.evaluate())

# %%
# Modèle linéaire contre modèle non linéaire
# ---------------------------------------------
# Un écart de R² de 0.1 à 0.15 entre les deux n'est pas négligeable : c'est
# 10 à 15 points de variance expliquée en plus, une part substantielle
# quand le R² plafonne déjà autour de 0.5.

trainer_lin = Trainer(X=X, y=y, model="linear", seed=42)
trainer_lin.fit()
print(f"R² random_forest_regressor : {trainer.evaluate()['r2']:.3f}")
print(f"R² linear                  : {trainer_lin.evaluate()['r2']:.3f}")

# %%
# Interpréter un R² modeste
# -----------------------------
# La cible est une note donnée par des dégustateurs humains, un jugement
# intrinsèquement bruité : le modèle ne peut pas expliquer une variance qui
# ne vient pas des variables chimiques mesurées mais du jugement lui-même.
# Un R² autour de 0.5 sur ce type de cible n'est donc pas un échec.

# %%
# Modèles de régression disponibles
# --------------------------------------

for nom in ["linear", "ridge", "lasso", "random_forest_regressor", "knn_regressor"]:
    t = Trainer(X=X, y=y, model=nom, seed=42)
    t.fit()
    print(f"{nom:25s} -> R²={t.evaluate()['r2']:.3f}")
# lasso ressort nettement en retrait ici (R² proche de 0, voire négatif) :
# sa régularisation par défaut est trop forte pour ce jeu de données et
# annule presque tous les coefficients. Un modèle n'est jamais bon "par
# défaut" dans l'absolu, le TP 8 de la série pédagogique (voir le dépôt
# trainedml-tp) creuse justement le réglage des hyperparamètres.
