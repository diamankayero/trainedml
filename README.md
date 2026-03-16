# trainedml

> Framework modulaire de machine learning en Python

<p align="left">
   <a href="https://diamankayero.github.io/trainedml/"><img src="https://img.shields.io/badge/Documentation-GitHub%20Pages-blue?logo=github" alt="Documentation"></a>
   <a href="https://trainedml.streamlit.app"><img src="https://img.shields.io/badge/Webapp-Streamlit-ff4b4b?logo=streamlit" alt="Webapp"></a>
   <a href="https://github.com/diamankayero/trainedml"><img src="https://img.shields.io/badge/GitHub-Repo-333?logo=github" alt="GitHub"></a>
   <img src="https://img.shields.io/badge/tests-passing-brightgreen" alt="Tests">
   <img src="https://img.shields.io/badge/coverage-100%25-success" alt="Coverage">
</p>

---


## Sommaire

- [Présentation](#présentation)
- [Diagramme de Gantt](#diagramme-de-gantt)
- [Architecture du projet](#architecture-du-projet)
- [Détail du package principal](#détail-du-package-principal)
- [Installation](#installation)
- [Utilisation rapide](#utilisation-rapide)
- [Tests](#tests)
- [Documentation](#documentation)
- [Contribution](#contribution)
- [Licence](#licence)
- [Contact](#contact)

<p align="center">
   <img src="public/matric_corre.png" alt="Matrice de corrélation" width="260"/>
> Modular machine learning framework for Python

[![PyPI version](https://img.shields.io/pypi/v/trainedml)](https://pypi.org/project/trainedml/)
[![Python](https://img.shields.io/pypi/pyversions/trainedml)](https://pypi.org/project/trainedml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-blue?logo=github)](https://diamankayero.github.io/trainedml/)
[![Webapp](https://img.shields.io/badge/Webapp-Streamlit-ff4b4b?logo=streamlit)](https://trainedml.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-333?logo=github)](https://github.com/diamankayero/trainedml)


## Summary

- [Overview](#overview)
- [Installation](#installation)
- [Features](#features)
- [API Usage](#api-usage)
- [CLI Usage](#cli-usage)
- [Architecture](#architecture)
- [Examples](#examples)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contribution](#contribution)
- [License](#license)
- [Contact](#contact)
- [Screenshots](#screenshots)
### ✨ Fonctionnalités principales
- **Code modulaire et documenté** : Implémentations claires de KNN, régression logistique, random forest, etc.
## Overview

**trainedml** is a modular Python package for training, benchmarking, and visualizing machine learning models on classic or custom datasets. Inspired by the clarity and extensibility of numpy, pandas, and matplotlib, it provides a unified API, CLI, and web interface for ML workflows.

---
- **Interface en ligne de commande** : Script CLI pour automatiser les pipelines ML (entraînement, benchmark, visualisation)
## Installation

Install from PyPI:
```bash
pip install trainedml
```

Or from source:
```bash
git clone https://github.com/diamankayero/trainedml.git
cd trainedml
pip install -e .
```

---
- **Application web interactive** : Interface Streamlit pour la démonstration et l'exploration
## Features

- Unified API for training, evaluation, and prediction
- Supports public datasets (Iris, Wine) and remote CSVs
- Modular models: KNN, Logistic Regression, Random Forest, etc.
- Standard metrics: accuracy, precision, recall, F1, R², MSE, RMSE, MAE
- Benchmarking with timing and parallelization
- Exploratory data analysis: distribution, correlation, missing values, outliers, normality, multicollinearity, profiling
- Visualization tools: heatmaps, histograms, line plots, boxplots, bivariate
- CLI and Streamlit webapp for interactive use
- Complete Sphinx documentation and unit tests

---
- **Outils de visualisation** : Heatmaps, histogrammes, courbes de performance
## API Usage

### Trainer class
```python
from trainedml import Trainer

trainer = Trainer(dataset="iris", model="random_forest")
trainer.fit()
- **Documentation complète** : Documentation Sphinx et tests unitaires

---

## Diagramme de Gantt

### DataLoader
```python
from trainedml.data.loader import DataLoader
X, y = DataLoader().load_dataset(name="wine")


### Model API
```python
from trainedml.models.knn import KNNModel
model = KNNModel(n_neighbors=3)
model.fit(X_train, y_train)

<p align="center">
   <img src="public/TrainedML_7days_diagram_gantt.png" alt="Diagramme de Gantt du projet" width="600"/>

### Visualizer
```python
from trainedml.visualization import Visualizer
viz = Visualizer(X)
fig = viz.heatmap()
fig.show()
</p>

---

## CLI Usage

Show help:
```bash
python -m trainedml --help
```

Benchmark on Iris dataset:
```bash
python -m trainedml --benchmark --dataset iris
```

Train KNN on Wine dataset:
```bash
python -m trainedml --model knn --dataset wine
```

Visualize correlation heatmap:
```bash
python -m trainedml --dataset iris --show
```

---
---
## Architecture
```
trainedml/
├── src/trainedml/           # Python package source
│   ├── __init__.py          # Trainer API
│   ├── analyzer.py          # Exploratory analysis
│   ├── benchmark.py         # Model benchmarking
│   ├── cli.py               # CLI interface
│   ├── evaluation.py        # Evaluation metrics
│   ├── figure.py            # Figure generation
│   ├── visualization.py     # Visualization tools
│   ├── data/                # Data loading
│   ├── models/              # ML models
│   ├── utils/               # Utilities
│   ├── viz/                 # Advanced visualizations
├── doc/                     # Sphinx documentation
├── tests/                   # Unit tests
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
```

---

## Examples

### Classification
```python
trainer = Trainer(dataset="iris", model="knn")
trainer.fit()


## Architecture du projet

### Regression
```python
trainer = Trainer(url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", target="quality", model="logistic")
trainer.fit()

Chaque dossier important contient un fichier markdown (`README.md`, `DOC_UTILISATION.md`, `streamlit_app.md`, etc.) détaillant commandes, usage et bonnes pratiques spécifiques.


### Benchmark
```python
from trainedml.benchmark import Benchmark
models = {'knn': KNNModel(), 'rf': RandomForestModel()}
bench = Benchmark(models)
results = bench.run(X_train, y_train, X_test, y_test)
print(results)
```

---
trainedml/
## Testing
Run unit tests:
```bash
pytest tests/
```

---
├── .github/               # Workflows CI/CD GitHub Actions
## Documentation
- [Online documentation (GitHub Pages)](https://diamankayero.github.io/trainedml/)
- [Local Sphinx docs](trainedml/doc/build/html/index.html)
- [Usage guide](DOC_UTILISATION.md)
- [Webapp docs](trainedml_webapp/doc/streamlit_app.md)

---
├── docs/                  # Documents PDF, rapports, etc. (voir docs/README.md)
## Contribution
Contributions are welcome!
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
├── GESTION_PROJET.md      # Guide de gestion de projet, déploiement, CI/CD
## License
MIT — see [LICENSE](LICENSE)

---
├── LICENSE                # Licence du projet
## Contact
- Open an [issue](https://github.com/diamankayero/trainedml/issues) on GitHub
- Submit a [pull request](https://github.com/diamankayero/trainedml/pulls)

---
├── public/                # Images et ressources publiques (voir public/README.md)
## Screenshots
<p align="center">
   <img src="public/matric_corre.png" alt="Correlation matrix" width="260"/>
   <img src="public/histogram.png" alt="Histogram" width="260"/>
   <img src="public/line.png" alt="Line plot" width="260"/>
   <img src="public/comparaison_de_model.jpeg" alt="Model comparison" width="260"/>
</p>
├── pyproject.toml         # Configuration du projet Python
├── README.md              # Ce fichier (documentation racine)
├── requirements.txt       # Dépendances Python
├── slides/                # Slides de présentation (voir slides/README.md)
├── src/                   # Code source pour la webapp Streamlit et la CLI (voir src/README.md)
├── tests/                 # Tests unitaires (voir tests/README.md)
├── trainedml/             # Package principal (voir détail ci-dessous)
├── trainedml_webapp/      # Application Streamlit (voir trainedml_webapp/README.md)
├── venv/                  # Environnement virtuel Python
```

Chaque dossier important contient un fichier markdown (`README.md`, `GESTION_PROJET.md`, `streamlit_app.md`, etc.) détaillant commandes, usage et bonnes pratiques spécifiques.
> 💡 **Remarque** :
> - Le vrai package Python (avec code, doc Sphinx, tests) se trouve dans `trainedml/trainedml`.
- **[Guide de gestion de projet](GESTION_PROJET.md)**
> - Les autres dossiers (public, docs, slides, etc.) servent à la documentation, aux ressources et à la présentation du projet.


## Détail du package principal

Le cœur du framework se trouve dans le dossier `trainedml/trainedml`, qui contient :

```
trainedml/
├── src/trainedml/           # Code source du package Python (voir src/trainedml/README.md)
│   ├── __init__.py          # API principale (Trainer)
│   ├── analyzer.py          # Analyse exploratoire
│   ├── benchmark.py         # Benchmark des modèles
│   ├── cli.py               # Interface CLI
│   ├── evaluation.py        # Métriques d'évaluation
│   ├── figure.py            # Génération de figures
│   ├── visualization.py     # Outils de visualisation
│   ├── data/                # Chargement des données (voir data/README.md)
│   ├── models/              # Modèles ML (voir models/README.md)
│   ├── utils/               # Fonctions utilitaires (voir utils/README.md)
│   ├── viz/                 # Visualisations avancées (voir viz/README.md)
├── doc/                     # Documentation Sphinx (voir doc/README.md)
├── tests/                   # Tests unitaires pour chaque module (voir tests/README.md)
├── README.md                # Documentation du package (niveau package)
├── DOC_UTILISATION.md       # Guide d'utilisation détaillé
```

Chaque module/dossier important contient un fichier markdown détaillant son usage, ses commandes et ses bonnes pratiques. L'organisation facilite l'extension, la maintenance et la génération automatique de la documentation API (Sphinx).


## Installation

### Prérequis
- Python 3.9 ou supérieur
- pip

### Étapes d'installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/diamankayero/trainedml.git
   cd trainedml
   ```

   ```bash
   py -3.11 -m venv venv
   # Windows :
> Modular machine learning framework for Python

[![PyPI version](https://img.shields.io/pypi/v/trainedml)](https://pypi.org/project/trainedml/)
[![Python](https://img.shields.io/pipi/pyversions/trainedml)](https://pypi.org/project/trainedml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/Documentation-GitHub%20Pages-blue?logo=github)](https://diamankayero.github.io/trainedml/)
[![Webapp](https://img.shields.io/badge/Webapp-Streamlit-ff4b4b?logo=streamlit)](https://trainedml.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-333?logo=github)](https://github.com/diamankayero/trainedml)
   # oubien si ça marche pas tu fais les commandes suivantes pour activer venv
   .\venv\Scripts\Activate.ps1 # pour le cas de ma machine
   # pour desactiver le venv on fait 

Install from PyPI:
```bash
pip install trainedml
```

Or from source:
```bash
git clone https://github.com/diamankayero/trainedml.git
cd trainedml
pip install -e .
```

---
   deactivate
## Overview

**trainedml** is a modular Python package for training, benchmarking, and visualizing machine learning models on classic or custom datasets.

### Features
- Modular, documented code: KNN, logistic regression, random forest, etc.
- Command-line interface (CLI) for ML pipelines (training, benchmarking, visualization)
- Streamlit web application for interactive exploration
- Visualization tools: heatmaps, histograms, performance curves
- Complete documentation (Sphinx) and unit tests

---
   # pour suprimer le venv
## Quickstart

### CLI Usage
Show help:
```bash
python -m trainedml --help
```

Benchmark on Iris dataset:
```bash
python -m trainedml --benchmark --dataset iris
```

Train KNN on Wine dataset:
```bash
python -m trainedml --model knn --dataset wine
```

### Streamlit Webapp
Launch interactive app:
```bash
streamlit run trainedml_webapp/src/app.py
```

### Python API Example
```python
from trainedml import Trainer

trainer = Trainer(dataset="iris", model="random_forest")
trainer.fit()
scores = trainer.evaluate()
print(scores)
predictions = trainer.predict([[5.1, 3.5, 1.4, 0.2]])
print(predictions)
```

---
   Remove-Item -Recurse -Force venv
## Project Structure
```
trainedml/
├── src/trainedml/           # Python package source
│   ├── __init__.py          # Trainer API
│   ├── analyzer.py          # Exploratory analysis
│   ├── benchmark.py         # Model benchmarking
│   ├── cli.py               # CLI interface
│   ├── evaluation.py        # Evaluation metrics
│   ├── figure.py            # Figure generation
│   ├── visualization.py     # Visualization tools
│   ├── data/                # Data loading
│   ├── models/              # ML models
│   ├── utils/               # Utilities
│   ├── viz/                 # Advanced visualizations
├── doc/                     # Sphinx documentation
├── tests/                   # Unit tests
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
```

---
   
## Testing
Run unit tests:
```bash
pytest tests/
```

---
   # Linux/Mac :
## Documentation
- [Online documentation (GitHub Pages)](https://diamankayero.github.io/trainedml/)
- [Local Sphinx docs](trainedml/doc/build/html/index.html)
- [Usage guide](DOC_UTILISATION.md)
- [Webapp docs](trainedml_webapp/doc/streamlit_app.md)

---
   source venv/bin/activate
## Contribution
Contributions are welcome!
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
   ```
## License
MIT — see [LICENSE](LICENSE)

---

## Contact
- Open an [issue](https://github.com/diamankayero/trainedml/issues) on GitHub
- Submit a [pull request](https://github.com/diamankayero/trainedml/pulls)

---
3. **Installer les dépendances**
## Screenshots
<p align="center">
    <img src="public/matric_corre.png" alt="Correlation matrix" width="260"/>
    <img src="public/histogram.png" alt="Histogram" width="260"/>
    <img src="public/line.png" alt="Line plot" width="260"/>
    <img src="public/comparaison_de_model.jpeg" alt="Model comparison" width="260"/>
</p>
   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation rapide

### Interface en ligne de commande (CLI)

Afficher l'aide :
```bash
python src/trainedml/cli.py --help
```

**Exemples d'utilisation :**

Benchmark comparatif sur le dataset Iris :
```bash
python src/trainedml/cli.py --benchmark --dataset iris
```

Entraîner un modèle KNN sur le dataset Wine :
```bash
python src/trainedml/cli.py --model knn --dataset wine
```

### [Application web Streamlit](https://trainedml.streamlit.app)

Lancer l'interface interactive :
```bash
streamlit run trainedml_webapp/src/app.py
```

L'application permet de :
- Charger différents datasets
- Comparer les performances des modèles
- Visualiser les résultats avec des graphiques interactifs
- Effectuer des prédictions manuelles


### API Python

Utilisation programmatique du framework avec différents jeux de données :

#### Exemple 1 : Dataset iris
```python
from trainedml import Trainer

trainer = Trainer(dataset="iris", model="random_forest")
trainer.fit()
scores = trainer.evaluate()
print(scores)
# Prédiction sur de nouvelles données
predictions = trainer.predict([[5.1, 3.5, 1.4, 0.2]])
print(predictions)
```

#### Exemple 2 : Dataset wine
```python
from trainedml import Trainer

trainer = Trainer(dataset="wine", model="knn")
trainer.fit()
scores = trainer.evaluate()
print(scores)
```

#### Exemple 3 : Dataset personnalisé via URL
```python
from trainedml import Trainer

trainer = Trainer(
   url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
   target="quality",
   model="logistic"
)
trainer.fit()
scores = trainer.evaluate()
print(scores)
```

---


<!-- Structure détaillée fusionnée ci-dessus, voir section Architecture du projet et Détail du package principal. -->

---

## Tests

Exécuter les tests unitaires :
```bash
pytest tests/
```

ou

```bash
python -m unittest discover tests/
```

---

## Documentation

La documentation complète est disponible à plusieurs endroits :

- **[Documentation en ligne (GitHub Pages)](https://diamankayero.github.io/trainedml/)**
- **[Documentation Sphinx locale](trainedml/doc/build/html/index.html)**
- **[Guide d'utilisation général](DOC_UTILISATION.md)**
- **[Documentation de l'application web](trainedml_webapp/doc/streamlit_app.md)**

---

## Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## Contact

Pour toute question, suggestion ou problème :
- Ouvrez une [issue](https://github.com/diamankayero/trainedml/issues) sur GitHub
- Proposez une [pull request](https://github.com/diamankayero/trainedml/pulls)


