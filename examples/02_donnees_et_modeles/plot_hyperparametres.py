"""
Hyperparamètres et modèles externes
======================================

Affiner un modèle nommé avec ``model_params``, puis brancher un estimateur
scikit-learn quelconque directement dans ``Trainer``.
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
