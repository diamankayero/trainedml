"""
Découper son dataset
=====================

Ce que fait vraiment le découpage train/test, et pourquoi le ratio compte,
en le pilotant explicitement via ``DataLoader.split()`` plutôt que de
laisser ``Trainer`` le faire tout seul.
"""

from trainedml.data.loader import DataLoader
from trainedml import Trainer

loader = DataLoader()
X, y = loader.load_dataset(name="iris")
print(f"Dataset complet : {len(X)} fleurs")

# %%
# Un ratio train/test explicite
# -------------------------------
# ``DataLoader.split`` renvoie ``X_train, X_test, y_train, y_test``, comme
# ``train_test_split`` de scikit-learn.

X_train, X_test, y_train, y_test = loader.split(X, y, test_size=0.2)
print(f"test_size=0.2 -> train={len(X_train)}  test={len(X_test)}")

# %%
# Effet du ratio sur la taille de chaque ensemble
# --------------------------------------------------
# Sur un dataset de 150 lignes, un ``test_size`` trop petit ne laisse que
# quelques dizaines d'exemples pour évaluer le modèle : une seule erreur
# suffit alors à faire bouger le score de plusieurs points.

for test_size in [0.1, 0.2, 0.3, 0.5]:
    X_tr, X_te, y_tr, y_te = loader.split(X, y, test_size=test_size)
    print(f"  test_size={test_size} -> train={len(X_tr):3d}  test={len(X_te):3d}")

# %%
# Trainer fait le même découpage en interne
# --------------------------------------------
# ``Trainer.fit(test_size=..., seed=...)`` n'a rien de magique : il pilote
# le même découpage aléatoire reproductible que ``DataLoader.split()``.

trainer = Trainer(dataset="iris", model="knn", seed=42)
trainer.fit(test_size=0.3)

X_train_manuel, X_test_manuel, _, _ = loader.split(X, y, test_size=0.3, random_state=42)

print(f"\ntrainer.X_test (via Trainer.fit)      : {len(trainer.X_test)} lignes")
print(f"X_test_manuel (via DataLoader.split)  : {len(X_test_manuel)} lignes")
print("Mêmes lignes exactement ?", list(trainer.X_test.index) == list(X_test_manuel.index))
