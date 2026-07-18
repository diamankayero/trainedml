"""
Comparer tous les modèles d'une tâche en une ligne avec trainedml.compare().

Exécution :
    python examples/compare_models.py
"""

from trainedml import compare

# Classification : tous les classificateurs sur Wine, validation croisée 5 plis.
# Le prétraitement (standardisation...) est appliqué à chaque pli.
df = compare(dataset="wine", cv=5)
print("\n=== Classification (wine) ===")
print(df.round(3).to_string())

# Comparer des modèles personnalisés, y compris scikit-learn
from sklearn.svm import SVC
from trainedml.models import KNNModel, RandomForestModel

df = compare(
    dataset="iris",
    models={
        "svc_rbf": SVC(kernel="rbf"),
        "knn_3": KNNModel(n_neighbors=3),
        "rf_100": RandomForestModel(n_estimators=100),
    },
    cv=5,
)
print("\n=== Modèles personnalisés (iris) ===")
print(df.round(3).to_string())
