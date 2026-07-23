"""
Charger des données distantes
================================

Charger un CSV public par URL avec ``DataLoader``, sans jamais le
télécharger à la main, et gérer le séparateur CSV.
"""

from trainedml.data.loader import DataLoader
from trainedml import Trainer

loader = DataLoader()
url_rouge = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

# %%
# trainedml détecte déjà automatiquement le séparateur pour les URLs
# ``winequality*`` (``;`` au lieu de ``,``) ; sur un autre CSV public,
# précisez ``sep=";"`` explicitement si le chargement produit une seule
# colonne géante.

X, y = loader.load_dataset(url=url_rouge, target="quality")
print("Colonnes :", X.columns.tolist())
print("Forme :", X.shape)

# %%
# Classification ou régression ?
# ---------------------------------
# ``quality`` est un entier entre 3 et 8 : on peut le traiter comme une
# classification (chaque note est une classe) ou comme une régression (la
# note est une quantité continue). ``Trainer`` déduit une heuristique à
# partir du nombre de valeurs distinctes.

print(y.value_counts().sort_index())

trainer = Trainer(X=X, y=y, model="random_forest", seed=42)
trainer.fit()
print("task détecté :", trainer.task)
print(trainer.evaluate())

# %%
# Comparer avec un second fichier
# ----------------------------------

url_blanc = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
X_blanc, y_blanc = loader.load_dataset(url=url_blanc, target="quality")
print(f"Échantillons rouges : {X.shape[0]}  -  blancs : {X_blanc.shape[0]}")
