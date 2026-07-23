"""
Sauvegarder et déployer
==========================

Entraîner une fois, sauvegarder, puis recharger un modèle pour noter de
nouveaux lots sans réentraîner, comme pour un scoring planifié en
production.
"""

import tempfile
from pathlib import Path

import pandas as pd

from trainedml import Trainer
from trainedml.data.loader import DataLoader

# Un dossier temporaire : cet exemple doit pouvoir tourner (et être rejoué
# tel quel après téléchargement) sans laisser de fichiers derrière lui.
dossier = Path(tempfile.mkdtemp())

trainer = Trainer(dataset="wine", model="random_forest", seed=42)
trainer.fit()
chemin_modele = dossier / "modele_vin.joblib"
trainer.save(chemin_modele)
print(f"Modèle sauvegardé : {chemin_modele}")

# %%
# Recharger et prédire
# -------------------------

restaure = Trainer.load(chemin_modele)
echantillon = [[13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]]
print("Prédiction :", restaure.predict(echantillon))

# %%
# Ce qu'un modèle rechargé ne peut plus faire
# --------------------------------------------------
# ``Trainer.load`` ne recharge pas les données d'origine : ``X_test`` et
# ``y_test`` valent ``None``. Ce n'est pas un bug : garder le fichier léger
# et ne pas trimballer le dataset d'entraînement jusqu'en production est un
# choix documenté. Seul ``predict`` fonctionne après rechargement.

try:
    restaure.evaluate()
except ValueError as e:
    print(f"evaluate() échoue après rechargement : {type(e).__name__}")

# %%
# Prédire un lot depuis un CSV
# ----------------------------------

X, y = DataLoader().load_dataset(name="wine")
chemin_lot = dossier / "lot_a_scorer.csv"
X.head(5).to_csv(chemin_lot, index=False)
lot = pd.read_csv(chemin_lot)
print("Prédictions sur le lot :", list(restaure.predict(lot)))

# %%
# L'équivalent en ligne de commande
# ----------------------------------------
# Pour un scoring planifié (cron, tâche planifiée Windows), la CLI évite
# d'écrire du code :
#
# .. code-block:: bash
#
#     trainedml --dataset wine --model random_forest --save modele.joblib
#     trainedml --load modele.joblib --input lot_a_scorer.csv --output resultats.csv
