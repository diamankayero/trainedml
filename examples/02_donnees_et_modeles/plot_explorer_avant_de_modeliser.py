"""
Explorer avant de modéliser
=============================

Avant tout modèle : un rapport EDA complet, une carte de corrélation, la
distribution des variables, les valeurs manquantes, les outliers et la
multicolinéarité, le tout depuis ``Visualizer``.
"""

from trainedml.data.loader import DataLoader
from trainedml.visualization import Visualizer

X, y = DataLoader().load_dataset(name="wine")
viz = Visualizer(X)

# %%
# Rapport EDA en une ligne
# --------------------------
# ``report()`` génère un rapport HTML auto-contenu (statistiques
# descriptives, corrélations, distributions, outliers, normalité) ; passez
# un ``path`` pour l'écrire sur disque, ou récupérez directement le HTML.

html = viz.report(title="Rapport EDA - wine")
print(f"Rapport généré : {len(html)} caractères de HTML auto-contenu.")

# %%
# Carte de corrélation
# -----------------------
# Deux variables corrélées à plus de 0.8 en valeur absolue apportent
# largement la même information au modèle.

viz.heatmap()

# %%
# Distribution des variables
# -----------------------------

viz.histogram(columns=["alcohol", "magnesium", "color_intensity"], legend=True)

# %%
# Valeurs manquantes
# ---------------------

manquantes = viz.missing()
print(f"Colonnes avec valeurs manquantes : {len(manquantes)}")

# %%
# Outliers (méthode IQR)
# --------------------------

outliers = viz.outliers()
for colonne, info in outliers.items():
    if info["count"] > 0:
        print(f"  {colonne:20s} : {info['count']} valeur(s) hors "
              f"[{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")

# %%
# Multicolinéarité (VIF)
# --------------------------
# Cette implémentation ne recentre pas les données avant le calcul, ce qui
# gonfle mécaniquement le VIF des variables dont la moyenne est loin de
# zéro : lisez-le en classement relatif entre variables, pas au seuil
# manuel de 10.

print(viz.multicollinearity().to_string(index=False))
