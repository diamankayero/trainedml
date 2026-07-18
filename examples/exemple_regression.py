"""
Pipeline de régression complet avec trainedml.

Montre : données en mémoire (X, y), détection automatique de la tâche,
métriques de régression, et comparatif de tous les régresseurs par
validation croisée.

Exécution :
    python examples/exemple_regression.py
"""

import numpy as np
import pandas as pd

from trainedml import Trainer, compare

# Jeu de données synthétique : prix en fonction de la surface et du nombre de pièces
rng = np.random.default_rng(42)
n = 200
surface = rng.uniform(20, 150, n)
pieces = np.clip((surface / 25).round() + rng.integers(-1, 2, n), 1, 8)
prix = 2000 * surface + 5000 * pieces + rng.normal(0, 8000, n)

X = pd.DataFrame({"surface": surface, "pieces": pieces})
y = pd.Series(prix, name="prix")

# Régression avec le Trainer, données en mémoire.
# La tâche est détectée automatiquement → métriques r2, mse, rmse, mae.
trainer = Trainer(X=X, y=y, model="ridge")
trainer.fit()
print("Scores régression :", {k: round(v, 3) for k, v in trainer.evaluate().items()})

# Prédiction sur un logement de 75 m² avec 3 pièces
print("Prix prédit (75 m², 3 pièces) :", trainer.predict([[75, 3]]).round(0))

# Comparer tous les régresseurs par validation croisée, en une ligne
df = compare(X=X, y=y, cv=5)
print("\n=== Comparatif des régresseurs (CV 5 plis) ===")
print(df.round(3).to_string())
