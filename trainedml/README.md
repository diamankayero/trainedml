# trainedml

Ce package propose des outils pour charger des jeux de données publics, entraîner et comparer des modèles de machine learning, et visualiser les résultats.

## Installation

```bash
pip install -r requirements.txt
```

## Fonctionnalités principales
- Chargement de jeux de données publics (ex : Iris)
- Modèles : KNN, Régression Logistique, Random Forest
- Visualisations : heatmap, histogramme, courbe
- API simple pour l'entraînement, l'évaluation et la comparaison

## Exemple d'utilisation

```python
from trainedml.data.loader import DataLoader
from trainedml.models.knn import KNNModel
from trainedml.visualization import Visualizer

# Chargement des données
iris = DataLoader().load_iris()

# Entraînement d'un modèle
X = iris.drop(columns=['species'])
y = iris['species']
model = KNNModel()
model.fit(X, y)

# Visualisation
viz = Visualizer(iris)
fig = viz.heatmap()
fig.show()
```

## Tests

```bash
python -m unittest discover tests
```
