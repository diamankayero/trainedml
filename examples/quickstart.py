"""
Quickstart trainedml : entraîner, évaluer, prédire et sauvegarder en quelques lignes.

Exécution :
    python examples/quickstart.py
"""

from trainedml import Trainer

# 1. Entraîner un KNN sur Iris (dataset intégré, chargé localement)
trainer = Trainer(dataset="iris", model="knn", model_params={"n_neighbors": 5})
trainer.fit()

# 2. Évaluer (métriques adaptées à la tâche automatiquement)
print("Scores :", trainer.evaluate())

# 3. Vérifier la stabilité sur plusieurs splits (sans recréer le Trainer)
for seed in range(3):
    scores = trainer.fit(seed=seed).evaluate()
    print(f"seed={seed} : accuracy={scores['accuracy']:.3f}")

# 4. Prédire sur de nouvelles données
print("Prédiction :", trainer.predict([[5.1, 3.5, 1.4, 0.2]]))

# 5. Sauvegarder puis recharger le modèle
trainer.save("model_iris.joblib")
restored = Trainer.load("model_iris.joblib")
print("Prédiction (modèle rechargé) :", restored.predict([[6.2, 2.8, 4.8, 1.8]]))
