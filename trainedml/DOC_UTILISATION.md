## Erreurs fréquentes et solutions

### 1. Scores parfaits (1.000 partout)
**Cause possible :** Split trop facile, fuite de données, dataset trop simple.
**Solution :** Tester plusieurs seeds, augmenter la taille du jeu de test, essayer sur un autre dataset.

### 2. Erreur "could not convert string to float"
**Cause :** Tentative de calcul de corrélation ou d'entraînement sur une colonne non numérique (ex : 'species').
**Solution :** Sélectionner uniquement les colonnes numériques pour les visualisations ou l'entraînement.

### 3. AttributeError: 'Axes' object has no attribute 'show'
**Cause :** Utilisation de `fig.show()` sur une figure matplotlib/seaborn (au lieu de `plt.show()`).
**Solution :** Utiliser `import matplotlib.pyplot as plt; plt.show()` pour afficher la figure.

### 4. ImportError: No module named 'trainedml...'
**Cause :** Problème d'installation ou de chemin Python.
**Solution :** Vérifier que le package est bien installé (`pip install .`) et que le terminal est bien dans l'environnement Python correct.

### 5. Problèmes de seed ou de reproductibilité
**Cause :** Résultats différents à chaque exécution.
**Solution :** Fixer la seed avec l'option `--seed` pour garantir la reproductibilité.

### 6. Erreur de shape ou de colonnes manquantes
**Cause :** Mauvais nom de colonne, ou DataFrame mal préparé.
**Solution :** Vérifier les noms de colonnes et la préparation des données avant d'appeler les modèles ou visualisations.

---
# Documentation du projet trainedml

## Présentation
## Fonctionnalités principales
- Modèles : KNN, Régression Logistique, Random Forest
- Visualisations : heatmap, histogramme, courbe
- API simple pour l'entraînement, l'évaluation et la comparaison
- Interface CLI pour exécuter un pipeline complet sans écrire de code Python

---


## Installation des dépendances supplémentaires (pooch et tqdm)

Pour bénéficier du téléchargement et du cache automatique des datasets ou jeu de données, trainedml utilise la bibliothèque `pooch`. Pour afficher une barre de progression lors du téléchargement, pooch utilise aussi la bibliothèque `tqdm`.

**Commandes à exécuter :**

```bash
pip install pooch tqdm
```

- `pooch` : gère le téléchargement, le cache local et la vérification d'intégrité des fichiers de données.
- `tqdm` : permet d'afficher une barre de progression lors du téléchargement des fichiers (optionnelle mais recommandée pour le confort utilisateur).

Si `tqdm` n'est pas installé, pooch affichera une erreur lors du téléchargement avec barre de progression.

```bash
pip install -r requirements.txt
# ou
pip install .
```

---

## Utilisation via la CLI

### Commande de base

```bash
python -m trainedml.cli --model random_forest --show
```

### Options disponibles
- `--model` : Choix du modèle (`knn`, `logistic`, `random_forest`).
- `--seed` : Seed pour le split train/test (par défaut 42).
- `--show` : Affiche la visualisation générée (matplotlib).
- `--histogram` : Affiche un histogramme des colonnes numériques.


### Exemples

- Entraîner un modèle Random Forest et afficher la heatmap :
  ```bash
  python -m trainedml.cli --model random_forest --show
  ```

- Tester la robustesse des modèles avec différentes seeds (aléas du split train/test) :
  ```bash
  python -m trainedml.cli --benchmark --seed 1
  python -m trainedml.cli --benchmark --seed 123
  ```
  > Cela permet de vérifier que les scores ne sont pas toujours parfaits et d'observer la variabilité selon la répartition des données.

- Tester la robustesse avec une grande proportion de test (jeu d'entraînement plus petit) :
  ```bash
  python -m trainedml.cli --benchmark --test-size 0.5 --seed 1
  python -m trainedml.cli --benchmark --test-size 0.7 --seed 42
  ```
  > Plus la taille du jeu de test est grande, plus il est difficile pour les modèles d'obtenir des scores parfaits. Cela permet de mieux évaluer leur robustesse.

- Afficher un histogramme :
  ```bash
  python -m trainedml.cli --histogram --show
  ```

---


## Gestion des datasets avec pooch

Depuis la version X, trainedml utilise la bibliothèque [pooch](https://www.fatiando.org/pooch/latest/) pour télécharger et mettre en cache les jeux de données publics.

### Qu’est-ce que pooch ?

- pooch est une bibliothèque Python qui gère le téléchargement, le cache local et la vérification d’intégrité des fichiers de données.
- Elle évite de re-télécharger les fichiers à chaque exécution : le dataset est stocké localement et réutilisé automatiquement.
- Elle vérifie que le fichier n’a pas été corrompu (via un hash).
- Elle simplifie la gestion de datasets volumineux ou multiples.

### Avantages pour l’utilisateur

- Plus besoin de modifier le code pour utiliser un autre dataset : il suffit de passer l’URL et le nom de la colonne cible.
- Les téléchargements sont rapides et fiables, même en cas de coupure réseau.
- Le cache local évite de saturer la connexion ou de perdre du temps.


### Gestion automatique des séparateurs CSV (exemple Wine Quality)

Certains jeux de données publics utilisent un séparateur différent de la virgule (par exemple, le point-virgule `;`). C'est le cas du célèbre dataset Wine Quality de l'UCI.

**Exemple concret :**

- **URL :** https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
- **Colonne cible :** quality
- **Séparateur :** `;`

La commande suivante fonctionne directement (le module détecte automatiquement le séparateur pour ce dataset) :

```bash
python -m trainedml.cli --url https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv --target quality --model logistic --show
```

Le module trainedml détecte automatiquement le séparateur `;` pour les fichiers winequality. Pour d'autres jeux de données avec un séparateur particulier, une option `--sep` pourra être ajoutée si besoin.

**Si tu obtiens une erreur "colonne non trouvée" ou que toutes les données semblent dans une seule colonne, vérifie le séparateur du CSV !**

---

### Exemple d’utilisation avec un dataset personnalisé

Supposons que tu veuilles utiliser ton propre CSV hébergé en ligne :

```bash
python -m trainedml.cli --url https://mon-site.fr/mon-dataset.csv --target classe
```

- `--url` : lien direct vers le fichier CSV
- `--target` : nom de la colonne cible (celle à prédire)

Le module téléchargera le fichier (une seule fois grâce à pooch), le lira, et utilisera la colonne `classe` comme cible pour l’entraînement et la prédiction.

### Utilisation avancée en Python

Tu peux aussi utiliser l’API Trainer directement :

```python
from trainedml import Trainer

trainer = Trainer(
  url="https://mon-site.fr/mon-dataset.csv",
  target="classe",
  model="logistic"
)
trainer.fit()
print(trainer.evaluate())
print(trainer.predict([[1.2, 3.4, 5.6, 7.8]]))
```

### Remarque

- Si tu utilises un dataset public connu (ex : iris, wine), il suffit de passer `--dataset iris` ou `--dataset wine`.
- Pour tout autre dataset, utilise `--url` et `--target`.

---
## Architecture du code

- `src/trainedml/data/loader.py` : Chargement des jeux de données publics (ex : Iris).
- `src/trainedml/models/` : Modèles ML (KNN, Logistic, Random Forest).
- `src/trainedml/visualization.py` : Visualiseur central (heatmap, histogramme, courbe).
- `src/trainedml/cli.py` : Interface en ligne de commande (pipeline complet).
- `src/trainedml/tests/` : Tests unitaires pour chaque brique.

---

## Points importants de l'évolution du projet

- **Début** : Le projet était composé de modules indépendants (data, modèles, visualisation) mais sans point d'entrée global.
- **Constat** : Le projet "marche" (tests OK) mais n'est pas utilisable directement par un humain sans API ou CLI.
- **Ajout d'un CLI** : Création d'un fichier `cli.py` pour exécuter un pipeline complet depuis le terminal.
- **Correction heatmap** : Sélection automatique des colonnes numériques pour éviter les erreurs de conversion.
- **Ajout histogramme** : Option `--histogram` pour générer un histogramme des colonnes numériques.
- **Robustesse** : Ajout de l'option `--seed` pour tester la robustesse des modèles sur différents splits.
- **Vérification** : Affichage des tailles de splits pour s'assurer de l'absence de fuite de données.

---


## Conseils d'utilisation
- Les scores parfaits (1.000) sont rares et peuvent indiquer un split "trop facile" ou une fuite de données. Pour t'auto-évaluer et progresser, utilise les options `--seed` et `--test-size` pour tester la robustesse de tes modèles.
- Le CLI est le point d'entrée recommandé pour les utilisateurs non-développeurs ou pour automatiser des workflows.
- Pour des usages avancés, il est possible d'utiliser directement les modules Python (data, modèles, visualisation).

> **Note personnelle (pour l'apprentissage)** :
> Cette démarche (tester plusieurs seeds, augmenter la taille du jeu de test) est essentielle pour comprendre la robustesse de tes modèles et éviter de te faire piéger par des résultats trop beaux pour être vrais. C'est une bonne pratique à garder pour tous tes futurs projets de machine learning.

---

## Pour aller plus loin
- Ajouter d'autres jeux de données ou modèles.
- Ajouter d'autres visualisations (courbes ROC, scatter, etc.).
- Ajouter une API Python haut-niveau (`trainedml.run(...)`).
- Publier le package sur PyPI.

---

## Auteurs
- diamankayero
- Contributions et corrections par GitHub Copilot (GPT-4.1)

---

## Licence
MIT

---
## Partie 2 : Industrialisation et Documentation API avec Sphinx

### Pourquoi industrialiser et documenter ?
L'objectif est de rendre le projet utilisable, maintenable, partageable et compréhensible par tous (développeurs, utilisateurs, collègues, futurs contributeurs). Une documentation professionnelle et une structure de code claire sont essentielles pour tout projet open source ou d'entreprise.

### Environnement virtuel (venv)
- **Pourquoi ?** Isole les dépendances du projet pour éviter les conflits avec d'autres projets Python sur la machine.
- **Commande d'initialisation :**
  ```bash
  python -m venv venv
  # Activation sous Windows
  .\venv\Scripts\activate
  # Activation sous Linux/Mac
  source venv/bin/activate
  # oubien si ça marche pas tu fais les commandes suivantes pour activer venv
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1 #pour le cas de ma machine
  # pour desactiver le venv on fait
  deactivate
  # pour suprimer le venv
  Remove-Item -Recurse -Force venv
  # Et pour installer en fontion de la version python on utilise la commande suivante
  py -3.11 -m venv venv
  ```
- **Installation des dépendances :**
  ```bash
  pip install -r requirements.txt
  pip install .
  # ou bien tu fais la commande suivant
  pip install -e .
  #pour desinstaller le module trainedml tu peux juste faire
  pip uninstall -y trainedml
  ```

### Génération de la documentation API avec Sphinx
- **Initialisation de Sphinx :**
  ```bash
  cd doc
  sphinx-quickstart source
  ```
- **Extensions activées :**
  - `sphinx.ext.autodoc` : génère la doc API à partir des docstrings Python.
  - `sphinx.ext.napoleon` : supporte les docstrings au format Google/Numpy.
  - `sphinx_rtd_theme` : thème moderne et professionnel (comme sur ReadTheDocs).
  - `sphinx.ext.viewcode`, `sphinx.ext.autosectionlabel` : navigation et affichage du code source.
- **Installation du thème :**
  ```bash
  pip install sphinx_rtd_theme
  ```
- **Configuration dans conf.py :**
  ```python
  html_theme = 'sphinx_rtd_theme'
  extensions = [
      'sphinx.ext.autodoc',
      'sphinx.ext.napoleon',
      'sphinx.ext.viewcode',
      'sphinx.ext.autosectionlabel',
  ]
  ```
- **Ajout du chemin source :**
  ```python
  import os, sys
  sys.path.insert(0, os.path.abspath('../../src'))
  ```
- **Structuration de la doc :**
  - `index.rst` : page d’accueil, sommaire, guide rapide, FAQ, etc.
  - `modules.rst` : API détaillée, inclut tous les modules avec `.. automodule::`.
  - Utilisation du toctree pour la navigation.

### Commandes pour générer la documentation
- **Sous Windows :**
  ```bash
  cd doc
  .\make.bat html
  ```
- **Sous Linux/Mac :**
  ```bash
  cd doc
  make html
  ```
- **Résultat :**
  Ouvre `doc/build/html/index.html` dans ton navigateur.

### Conseils pour une doc parfaite
- Utilise des docstrings complètes et structurées (Google/Numpy) dans chaque classe/fonction.
- Ajoute des exemples d’utilisation dans la doc utilisateur et dans les docstrings.
- Corrige tous les warnings Sphinx (soulignement des titres, toctree, etc.).
- Utilise un thème moderne (sphinx_rtd_theme, furo, pydata-sphinx-theme).
- Ajoute une FAQ, une section contribution, une licence.
- Pour publier en ligne : ReadTheDocs (gratuit, facile à connecter à GitHub).

### Résumé des apports
- Projet modulaire, testé, documenté comme un vrai projet open source.
- Documentation API générée automatiquement et toujours à jour.
- Utilisation professionnelle de Sphinx et des outils Python modernes.
- Prêt à être partagé, maintenu, et enrichi par d’autres utilisateurs ou contributeurs.
