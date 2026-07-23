"""
Projet complet : de bout en bout
===================================

Un pipeline complet sur un cas réel : classer des vins en trois catégories
commerciales à partir de leur analyse chimique, avec exploration,
comparaison de modèles, réglage, audit de rigueur, et sauvegarde pour la
production.
"""

import statistics
import tempfile
from pathlib import Path

from trainedml import Trainer, compare
from trainedml.data.loader import DataLoader
from trainedml.visualization import Visualizer

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
X, y = DataLoader().load_dataset(url=url, target="quality")

print(f"Valeurs manquantes : {len(Visualizer(X).missing())} colonne(s)")

# %%
# Construire la cible métier
# -------------------------------
# Trois catégories commerciales plutôt que la note brute de 0 à 10.

y_classe = y.apply(lambda q: "bas" if q <= 4 else ("moyen" if q <= 6 else "haut"))
repartition = y_classe.value_counts()
baseline = repartition.max() / repartition.sum()
print(repartition)
print(f"Baseline naïve (toujours la classe majoritaire) : {baseline:.3f}")

# %%
# Comparer plusieurs modèles
# -------------------------------
# Les classes sont déséquilibrées : l'accuracy seule peut être trompeuse,
# on la lit à côté de la baseline naïve et de precision/recall.

resultats = compare(X=X, y=y_classe, cv=5, seed=42)
print(resultats[["accuracy", "accuracy_std", "precision", "recall"]])

# %%
# Affiner et auditer avant de livrer
# -----------------------------------------
# Le gain d'un réglage est généralement modeste : la majorité de la
# performance vient du modèle et des données, pas du nombre d'arbres.

scores = []
for seed in range(10):
    t = Trainer(X=X, y=y_classe, model="random_forest", seed=seed,
                model_params={"n_estimators": 300})
    t.fit()
    scores.append(t.evaluate()["accuracy"])
print(f"Score sur 10 graines : min={min(scores):.3f} max={max(scores):.3f} "
      f"moyenne={statistics.mean(scores):.3f} (baseline={baseline:.3f})")

# %%
# Sauvegarder pour la production
# -------------------------------------

trainer = Trainer(X=X, y=y_classe, model="random_forest", seed=42,
                   model_params={"n_estimators": 300})
trainer.fit()
chemin_modele = Path(tempfile.mkdtemp()) / "modele_qualite_vin.joblib"
trainer.save(chemin_modele)

restaure = Trainer.load(chemin_modele)
print("Prédictions sur un nouveau lot :", list(restaure.predict(X.head(5))))
