"""
Premiers pas : entraîner et évaluer un modèle
==============================================

Le strict nécessaire pour entraîner un modèle avec trainedml : charger un
dataset, entraîner, lire les métriques, prédire sur une donnée nouvelle.
"""

# %%
# On charge le dataset Iris intégré (aucun téléchargement, aucun fichier à
# fournir) et on entraîne une forêt aléatoire pour reconnaître l'espèce
# d'une fleur à partir de quatre mesures.

from trainedml import Trainer

trainer = Trainer(dataset="iris", model="random_forest", seed=42)
trainer.fit()
print(trainer.evaluate())

# %%
# Prédire une donnée nouvelle
# ----------------------------
# ``predict`` prend une liste de listes : une ligne par échantillon.

fleur = [[5.1, 3.5, 1.4, 0.2]]
print("Espèce prédite :", trainer.predict(fleur)[0])

# %%
# Comparer plusieurs modèles
# ----------------------------
# Le nom du modèle est le seul paramètre qui change : l'API d'entraînement,
# d'évaluation et de prédiction reste identique quel que soit l'algorithme.

for nom_modele in ["random_forest", "knn", "logistic"]:
    t = Trainer(dataset="iris", model=nom_modele, seed=42)
    t.fit()
    print(f"{nom_modele:15s} -> {t.evaluate()}")
