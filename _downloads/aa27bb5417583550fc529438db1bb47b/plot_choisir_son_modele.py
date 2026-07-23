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
