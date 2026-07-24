"""
Comparer et choisir un modèle
===============================

``compare()`` entraîne et évalue plusieurs modèles en une ligne, avec
validation croisée, et renvoie un tableau trié prêt à lire.
"""

from trainedml import compare

resultats = compare(dataset="wine", cv=5, seed=42)
print(resultats)

# %%
# Score moyen et stabilité
# --------------------------
# Le comparatif inclut un écart-type par métrique (``*_std``) entre les
# plis de validation croisée : un modèle en tête avec un écart-type
# nettement supérieur aux autres est moins fiable qu'il n'y paraît.

meilleur = resultats.index[0]
print(f"\nMeilleur score moyen : {meilleur} ({resultats.loc[meilleur, 'accuracy']:.3f})")
print(resultats[["accuracy", "accuracy_std", "fit_time"]])

# %%
# Un estimateur scikit-learn arbitraire
# ----------------------------------------
# ``compare()`` accepte aussi un dictionnaire ``models`` personnalisé,
# mélangeant modèles nommés trainedml et estimateurs scikit-learn.

from sklearn.svm import SVC

resultats_perso = compare(
    dataset="wine",
    models={"svm_rbf": SVC(kernel="rbf", C=1.0), "svm_lineaire": SVC(kernel="linear")},
    cv=5,
    seed=42,
)
print(resultats_perso)

# %%
# Classes déséquilibrées : l'accuracy peut cacher un problème
# ------------------------------------------------------------------
# Sur un jeu de données où une classe est rare, ``Trainer``/``compare()``
# émettent un avertissement. Ici, la classe "rare" ne représente que 10 %
# des exemples.

import numpy as np
import pandas as pd

from trainedml import Trainer

rng = np.random.default_rng(0)
n = 300
a, b = rng.normal(size=n), rng.normal(size=n)
score = a + 0.5 * b + rng.normal(0, 0.5, n)
y_desequilibre = pd.Series(np.where(score > np.quantile(score, 0.90), "rare", "frequent"))
X_desequilibre = pd.DataFrame({"a": a, "b": b})

trainer = Trainer(X=X_desequilibre, y=y_desequilibre, model="logistic", seed=42)
trainer.fit()  # émet un UserWarning : classes déséquilibrées
print("Sans class_weight :", trainer.evaluate())

# %%
# ``class_weight="balanced"`` (déjà supporté par random_forest et logistic
# via ``model_params``) pénalise davantage les erreurs sur la classe rare :
# l'accuracy globale baisse, mais la classe rare est mieux détectée. Un vrai
# compromis, pas une amélioration gratuite : à activer quand rater un cas
# rare coûte plus cher qu'une fausse alerte (fraude, panne, diagnostic...).

trainer_pondere = Trainer(X=X_desequilibre, y=y_desequilibre, model="logistic", seed=42,
                          model_params={"class_weight": "balanced"})
trainer_pondere.fit()
print("Avec class_weight='balanced' :", trainer_pondere.evaluate())
