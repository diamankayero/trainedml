"""
Générer un rapport exploratoire (EDA) HTML complet avec trainedml.

Le rapport contient : aperçu, statistiques descriptives, valeurs manquantes,
corrélations avec heatmap, distributions, outliers, tests de normalité et VIF.
Il est auto-contenu (figures embarquées) et s'ouvre dans un navigateur.

Exécution :
    python examples/rapport_eda.py
"""

import pandas as pd

from trainedml.data.loader import DataLoader
from trainedml.visualization import Visualizer

# Charger un dataset et reconstituer le DataFrame complet
loader = DataLoader()
X, y = loader.load_dataset(name="iris")
data = pd.concat([X, y], axis=1)

# Générer le rapport
viz = Visualizer(data)
viz.report("rapport_iris.html", title="Rapport EDA — Iris")
print("Rapport écrit dans rapport_iris.html")
