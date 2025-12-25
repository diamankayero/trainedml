# trainedml

> Framework pédagogique et modulaire de machine learning en Python

<p align="left">
   <a href="https://diamankayero.github.io/trainedml/"><img src="https://img.shields.io/badge/Documentation-GitHub%20Pages-blue?logo=github" alt="Documentation"></a>
   <a href="https://trainedml.streamlit.app"><img src="https://img.shields.io/badge/Webapp-Streamlit-ff4b4b?logo=streamlit" alt="Webapp"></a>
   <a href="https://github.com/diamankayero/trainedml"><img src="https://img.shields.io/badge/GitHub-Repo-333?logo=github" alt="GitHub"></a>
</p>

---

## 📋 Présentation

**trainedml** est un framework Python conçu pour l'apprentissage et la comparaison de modèles de machine learning sur des jeux de données classiques ou personnalisés. 

### ✨ Fonctionnalités principales

- **Code modulaire et documenté** : Implémentations claires de KNN, régression logistique, random forest, etc.
- **Interface en ligne de commande** : Script CLI pour automatiser les pipelines ML (entraînement, benchmark, visualisation)
- **Application web interactive** : Interface Streamlit pour la démonstration et l'exploration
- **Outils de visualisation** : Heatmaps, histogrammes, courbes de performance
- **Documentation complète** : Documentation Sphinx et tests unitaires

---

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Étapes d'installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/diamankayero/trainedml.git
   cd trainedml
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv venv
   
   # Windows :
   venv\Scripts\activate
   
   # Linux/Mac :
   source venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Utilisation rapide

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

### Application web Streamlit

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

Utilisation programmatique du framework :

```python
from trainedml import Trainer

# Créer un trainer
trainer = Trainer(dataset="iris", model="random_forest")

# Entraîner le modèle
trainer.fit()

# Évaluer les performances
scores = trainer.evaluate()
print(scores)
```

---

## 📁 Structure du projet

> 💡 **Note importante :** Chaque dossier clé contient un fichier markdown (`README.md`, `DOC_UTILISATION.md`, `streamlit_app.md`, etc.) détaillant les commandes, l'utilisation et les bonnes pratiques spécifiques. Consultez-les pour une prise en main rapide.

```
trainedml/
│
├── src/trainedml/              # Code source principal
│   ├── __init__.py             # API haut niveau (Trainer)
│   ├── cli.py                  # Interface ligne de commande
│   ├── benchmark.py            # Comparaison de modèles
│   ├── evaluation.py           # Métriques d'évaluation
│   ├── visualization.py        # Outils de visualisation
│   ├── data/                   # Chargement de données
│   ├── models/                 # Implémentations des modèles
│   │   ├── knn.py
│   │   ├── logistic.py
│   │   └── random_forest.py
│   └── viz/                    # Visualisations spécialisées
│
├── tests/                      # Tests unitaires
│
├── trainedml_webapp/           # Application Streamlit
│   ├── src/app.py
│   └── doc/                    # Documentation webapp
│
├── doc/                        # Documentation Sphinx
│
├── requirements.txt            # Dépendances Python
├── pyproject.toml             # Configuration du projet
└── README.md                  # Ce fichier
```

<details>
<summary>📂 Voir l'arborescence complète</summary>

```
trainedml/
│
├── .devcontainer/
├── .gitignore
├── DOC_UTILISATION.md
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── slides/
│
├── src/trainedml/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── evaluation.py
│   ├── figure.py
│   ├── visualization.py
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── viz/
│
├── tests/
│
├── trainedml/
│   ├── build/
│   ├── dist/
│   └── doc/build/html/
│
├── trainedml_webapp/
│   ├── doc/
│   └── src/app.py
│
└── venv/
```
</details>

---

## 🧪 Tests

Exécuter les tests unitaires :
```bash
pytest tests/
```

ou

```bash
python -m unittest discover tests/
```

---

## 📚 Documentation

La documentation complète est disponible à plusieurs endroits :

- **[Documentation en ligne (GitHub Pages)](https://diamankayero.github.io/trainedml/)**
- **[Documentation Sphinx locale](trainedml/doc/build/html/index.html)**
- **[Guide d'utilisation général](DOC_UTILISATION.md)**
- **[Documentation de l'application web](trainedml_webapp/doc/streamlit_app.md)**

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Poussez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 📧 Contact

Pour toute question, suggestion ou problème :
- Ouvrez une [issue](https://github.com/diamankayero/trainedml/issues) sur GitHub
- Proposez une [pull request](https://github.com/diamankayero/trainedml/pulls)

---

## 🌟 Remerciements

Merci à tous les contributeurs qui ont participé à ce projet éducatif !
