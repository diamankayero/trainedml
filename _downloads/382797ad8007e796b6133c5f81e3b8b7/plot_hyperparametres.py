"""
Hyperparamètres : réglage manuel et recherche automatique
=============================================================

Affiner un modèle nommé avec ``model_params``, brancher un estimateur
scikit-learn quelconque, puis laisser ``grid_search``/``random_search``
chercher la meilleure combinaison par validation croisée.
"""

from sklearn.svm import SVC

from trainedml import Trainer

# %%
# ``model_params`` avec un modèle nommé
# ------------------------------------------

t_petit = Trainer(dataset="wine", model="random_forest", seed=42,
                   model_params={"n_estimators": 10, "max_depth": 3})
t_petit.fit()
t_grand = Trainer(dataset="wine", model="random_forest", seed=42,
                   model_params={"n_estimators": 300, "max_depth": 3})
t_grand.fit()
print("n_estimators=10  :", t_petit.evaluate())
print("n_estimators=300 :", t_grand.evaluate())

# %%
# Un estimateur scikit-learn arbitraire
# ------------------------------------------
# Aucun changement dans le reste du code : ``fit``, ``evaluate``,
# ``predict`` fonctionnent à l'identique, que le modèle soit nommé ou un
# estimateur passé directement.

t_svm = Trainer(dataset="wine", model=SVC(kernel="rbf", C=1.0), seed=42)
t_svm.fit()
print("SVC(kernel='rbf', C=1.0) :", t_svm.evaluate())

# %%
# ``model_params`` est incompatible avec un estimateur déjà construit
# --------------------------------------------------------------------------
# Un objet déjà instancié porte déjà ses propres hyperparamètres :
# configurez-le directement (``SVC(C=2)``) plutôt que de passer
# ``model_params``.

try:
    Trainer(dataset="wine", model=SVC(), model_params={"C": 2}, seed=42)
except ValueError as e:
    print(f"Erreur attendue : {e}")

# %%
# Recherche exhaustive : grid_search
# ---------------------------------------
# Essaie toutes les combinaisons du grillage et réentraîne automatiquement
# avec la meilleure trouvée : le Trainer est prêt pour ``evaluate()``/
# ``predict()`` juste après.

trainer = Trainer(dataset="wine", model="knn", seed=42)
resultats = trainer.grid_search({"n_neighbors": [1, 3, 5, 7, 9]}, cv=5)
print(resultats)
print("Meilleurs paramètres :", trainer.best_params_)
print("Score CV associé :", round(trainer.best_score_, 3))

# %%
# Recherche aléatoire : random_search
# ------------------------------------------
# Pour un espace de recherche trop grand à explorer entièrement,
# ``random_search`` tire ``n_iter`` combinaisons au hasard plutôt que de
# toutes les essayer.

trainer_rf = Trainer(dataset="wine", model="random_forest", seed=42)
trainer_rf.random_search(
    {"n_estimators": [50, 100, 200, 300], "max_depth": [None, 3, 5, 10]},
    n_iter=6, cv=5,
)
print("Meilleurs paramètres (random_forest) :", trainer_rf.best_params_)
print(trainer_rf.evaluate())
