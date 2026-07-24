"""
Diagnostiquer un modèle de classification
============================================

Une fois un modèle entraîné, l'accuracy seule ne suffit pas : matrice de
confusion, courbe ROC/AUC et importance des variables donnent une image
bien plus complète de son comportement.
"""

from trainedml import Trainer

trainer = Trainer(dataset="wine", model="random_forest", seed=42)
trainer.fit()
print(trainer.evaluate())

# %%
# Matrice de confusion
# ------------------------
# Où le modèle se trompe-t-il, et entre quelles classes ?

trainer.confusion_matrix(normalize="true");

# %%
# Courbe ROC (une par classe, one-vs-rest)
# ---------------------------------------------
# L'AUC mesure la capacité du modèle à bien classer les exemples positifs
# avant les négatifs, indépendamment du seuil de décision choisi.

trainer.roc_curve();

# %%
# Importance des variables
# -----------------------------
# ``feature_importances()`` fonctionne quel que soit le modèle : natif pour
# les arbres (``feature_importances_``) et les modèles linéaires (``coef_``),
# et par permutation pour les autres (KNN, SVM...).

print(trainer.feature_importances().head())
trainer.plot_feature_importances(top_n=8);
