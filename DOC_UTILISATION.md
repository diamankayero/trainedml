# Documentation du projet trainedml

Guide d'utilisation, journal du projet et repères pour ne pas perdre
l'équilibre du projet. La documentation détaillée de chaque dossier est dans
le README du dossier concerné.

## Documentation par dossier

| Dossier | Contenu |
|---|---|
| [src/trainedml/](src/trainedml/README.md) | Architecture du package, rôle de chaque module, conventions |
| [src/trainedml/data/](src/trainedml/data/README.md) | Chargement des données (datasets intégrés, CSV distants, split) |
| [src/trainedml/models/](src/trainedml/models/README.md) | Modèles ML, registres, comment ajouter un modèle |
| [src/trainedml/viz/](src/trainedml/viz/README.md) | Visualisations spécialisées et compatibilité |
| [tests/](tests/README.md) | Organisation des tests et règles (pas de réseau, non-régression) |
| [examples/](examples/README.md) | Galerie d'exemples (source des pages Sphinx-Gallery de la doc) |
| [doc/](doc/README.md) | Documentation Sphinx : build local et publication |
| [public/](public/README.md) | Images du README |
| [.github/workflows/](.github/workflows/README.md) | CI, publication PyPI, procédure de release |
| [trainedml-webapp](https://github.com/diamankayero/trainedml-webapp) | Dépôt séparé : API HTTP (FastAPI) + page web de démo |

---

## Présentation

trainedml est un package Python modulaire pour entraîner, comparer et
visualiser des modèles de machine learning : du CSV au comparatif de modèles
et au rapport exploratoire, en une ligne.

## Fonctionnalités principales (v0.2.0)

- API `Trainer` : entraînement, évaluation, prédiction, sauvegarde/rechargement
- `compare()` : comparaison de tous les modèles adaptés, validation croisée, DataFrame trié
- Prétraitement automatique : imputation, standardisation, encodage one-hot
- Modèles intégrés (KNN, Logistique, Random Forest, Linear/Ridge/Lasso) et
  n'importe quel estimateur scikit-learn
- Détection automatique du type de tâche, métriques adaptées
- Rapport EDA HTML auto-contenu
- CLI complète pour exécuter un pipeline sans écrire de Python

## Installation

```bash
pip install trainedml
# ou depuis les sources
pip install -e .
```

Dépendances gérées par pyproject.toml (numpy, pandas, scikit-learn,
matplotlib, seaborn, plotly, tqdm, requests, pooch, statsmodels, joblib, scipy).

---

## Utilisation via la CLI

### Commande de base

```bash
python -m trainedml --model random_forest --dataset iris --show
```

### Options disponibles

- `--model` : modèle à utiliser (`knn`, `logistic`, `random_forest`, `linear`, `ridge`, `lasso`, `knn_regressor`, `random_forest_regressor`)
- `--dataset` : dataset intégré (`iris`, `wine`)
- `--url` / `--target` : CSV distant et nom de la colonne cible
- `--seed` : seed du split train/test (défaut 42)
- `--test-size` : proportion de test (défaut 0.3)
- `--benchmark` : comparer tous les modèles adaptés à la tâche
- `--cv N` : benchmark par validation croisée à N plis (0 = simple split)
- `--save fichier.joblib` : sauvegarder le modèle entraîné
- `--load fichier.joblib --input data.csv --output preds.csv` : prédire sur un CSV avec un modèle sauvegardé
- `--show` : afficher la figure générée
- `--histogram` : histogramme des colonnes numériques
- `--line X Y` : courbe entre deux colonnes

### Exemples

- Benchmark par validation croisée 5 plis sur Wine :
  ```bash
  python -m trainedml --dataset wine --benchmark --cv 5
  ```

- Tester la robustesse des modèles avec différentes seeds :
  ```bash
  python -m trainedml --benchmark --seed 1
  python -m trainedml --benchmark --seed 123
  ```
  > Cela permet de vérifier que les scores ne sont pas toujours parfaits et d'observer la variabilité selon la répartition des données.

- Tester la robustesse avec une grande proportion de test :
  ```bash
  python -m trainedml --benchmark --test-size 0.5 --seed 1
  ```

- Entraîner, sauvegarder, puis prédire sur de nouvelles données :
  ```bash
  python -m trainedml --dataset iris --model knn --save model.joblib
  python -m trainedml --load model.joblib --input nouvelles_donnees.csv --output predictions.csv
  ```

---

## Utilisation en Python

```python
from trainedml import Trainer, compare

# Workflow standard
trainer = Trainer(dataset="iris", model="knn", model_params={"n_neighbors": 5})
trainer.fit()
print(trainer.evaluate())
print(trainer.predict([[5.1, 3.5, 1.4, 0.2]]))

# Varier le seed sans recréer le Trainer
for s in range(5):
    print(trainer.fit(seed=s).evaluate())

# Données personnelles en mémoire
trainer = Trainer(X=mon_dataframe, y=ma_cible, model="ridge")

# N'importe quel estimateur scikit-learn
from sklearn.svm import SVC
trainer = Trainer(dataset="iris", model=SVC())

# Sauvegarde et rechargement
trainer.fit().save("model.joblib")
restored = Trainer.load("model.joblib")

# Comparer tous les modèles en une ligne
print(compare(dataset="wine", cv=5))

# Rapport EDA HTML
from trainedml.visualization import Visualizer
Visualizer(mon_dataframe).report("rapport.html")
```

---

## Gestion des datasets

### Datasets intégrés (hors-ligne)

`iris` et `wine` sont chargés localement via scikit-learn : aucun réseau
requis, tests et démos fonctionnent hors connexion.

### CSV distants avec pooch

Pour un CSV en ligne, trainedml utilise [pooch](https://www.fatiando.org/pooch/latest/) :
téléchargement, cache local (pas de re-téléchargement) et vérification
d'intégrité par hash.

```bash
python -m trainedml --url https://mon-site.fr/mon-dataset.csv --target classe
```

### Séparateurs CSV (exemple Wine Quality)

Certains CSV utilisent `;` au lieu de `,`. Le séparateur `;` est détecté
automatiquement pour les fichiers winequality :

```bash
python -m trainedml --url https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv --target quality --model logistic --show
```

Si vous obtenez une erreur "colonne non trouvée" ou que toutes les données
semblent dans une seule colonne, vérifiez le séparateur du CSV.

---

## Journal du projet

### Phase 1 : les fondations

- Début : modules indépendants (data, modèles, visualisation) sans point d'entrée global.
- Constat : le projet "marche" (tests OK) mais n'est pas utilisable sans API ou CLI.
- Ajout d'un CLI (`cli.py`) pour exécuter un pipeline complet depuis le terminal.
- Correction heatmap : sélection automatique des colonnes numériques.
- Ajout de `--histogram` et de `--seed` pour tester la robustesse.
- Affichage des tailles de splits pour vérifier l'absence de fuite de données.

### Phase 2 : industrialisation

- Environnement virtuel, packaging (pyproject.toml), publication PyPI (0.1.x).
- Documentation Sphinx (autodoc + napoleon + thème RTD) publiée sur GitHub Pages.
- Docstrings NumPy-style avec formules mathématiques.

### Phase 3 : la refonte 0.2.0 (2026-07-18)

Objectif : trouver la niche du package ("du CSV au comparatif de modèles et
au rapport EDA en une ligne") et le rendre digne de confiance.

- Rectification d'architecture : la CLI et les tests passent par `DataLoader.split`
  (API du package) au lieu d'appeler scikit-learn directement ; factories
  dupliquées supprimées ; `__pycache__` retirés du suivi git.
- Bugs corrigés : métriques de classification appliquées aux régresseurs,
  incompatibilités pandas récents (StringDtype), heatmap sur colonnes mixtes,
  incompatibilités scipy/matplotlib anciens.
- Nouvelles capacités : `compare()`, prétraitement automatique sans fuite,
  validation croisée, hyperparamètres, estimateurs sklearn arbitraires,
  données en mémoire, save/load, rapport EDA HTML, CLI enrichie.
- Qualité : type hints + mypy, ruff, CI honnête en matrice (3.9 à 3.13 + Windows),
  publication PyPI uniquement sur tag, notebooks validés, CHANGELOG.
- Publication : v0.2.0 sur PyPI, Release GitHub, doc à jour.
- Style : pas de tirets longs, ponctuation à la française ; pas de signature
  d'outil dans les commits.

### Phase 4 : ouverture web (2026-07-18)

- Documentation par dossier : un README dans chaque dossier du projet,
  DOC_UTILISATION.md restructuré avec ce journal.
- API HTTP : FastAPI expose train/predict/compare/models ; page de démo
  HTML/JS vanilla servie à la racine, doc interactive sur /docs. Le pattern :
  le frontend (HTML, React...) ne fait que des appels fetch, le Python reste
  côté serveur.
- Démo déployée et vérifiée en production : https://trainedml.onrender.com
  (train, predict et compare testés sur l'instance en ligne).
- Séparation des dépôts : la webapp a été extraite dans son propre repo
  https://github.com/diamankayero/trainedml-webapp qui consomme trainedml
  depuis PyPI (test d'intégration permanent du package publié). Le repo du
  package ne contient plus que la bibliothèque ; render.yaml, Dockerfile et
  l'extra [web] sont partis avec la webapp.
- ModeLmL : https://github.com/diamankayero/ModeLmL (nom choisi par
  l'utilisateur), **en ligne sur https://modelml.vercel.app**. La page HTML
  de trainedml-webapp est conservée volontairement : c'est la version la
  plus lisible pour apprendre ; ModeLmL est la version produit. Les deux
  consomment la même API.
- Migration Next.js + Tailwind : `/` = vitrine style Notion (clair,
  éditorial), `/app` = atelier repensé en produit avec navigation latérale
  à cinq écrans (Aperçu = dashboard du dataset, Données, Analyse native
  dessinée depuis /api/analysis avec heatmap de corrélation et histogrammes
  SVG maison, Comparaison, Prédiction). Le rapport HTML de trainedml est un
  téléchargement, pas un écran ; les tests de normalité y sont relégués
  (peu utiles en ML, retirés de l'Aperçu et de l'Analyse). L'écran
  Prédiction affiche l'importance des variables du modèle entraîné.
  L'API a été enrichie en conséquence (/api/dataset, /api/analysis,
  /api/report, importances dans /api/train, données uploadées).
- Hero de la vitrine : un vrai GIF de l'atelier en action (pas un mockup),
  capturé par un banc de test qui pilote un navigateur réel à travers les
  cinq écrans, frappe clavier comprise. Pas de ffmpeg disponible sur la
  machine : le pipeline est en JS pur (puppeteer pour les frames, sharp
  pour le redimensionnement, gifenc pour l'encodage GIF).
  Vérification par bancs de test visuels (scripts/visual-check.mjs et
  scripts/capture-hero-gif.mjs, Edge headless piloté).

### Phase 5 : fermeture des chantiers restants (2026-07-18)

- Docstrings, commentaires, messages d'erreur, sorties CLI et contenu HTML du
  rapport EDA : entièrement traduits en anglais (le mélange FR/EN de la
  phase 4 est résorbé). Deux passes de grep nécessaires : la première ne
  cherchait que les caractères accentués et a raté plusieurs modules `viz/`
  sans diacritiques (`target.py`, `missing.py`, `multicollinearity.py`) ;
  une seconde passe par mots-clés les a rattrapés.
- `figure.py`, `analyzer.py`, `visualization.py` et tout `viz/*` annotés en
  type hints ; l'exclusion mypy sur ces modules a été retirée de
  `pyproject.toml` et `mypy src/trainedml` passe sans erreur.
- Bugs trouvés en cours d'annotation : deux fichiers (`viz/boxplot.py`,
  `viz/correlation.py`) avaient `from __future__ import annotations` placé
  avant le docstring de module, ce qui le désactive silencieusement (Python
  n'assigne `__doc__` que si le docstring est la toute première instruction) ;
  `viz/normality.py` avait un second docstring orphelin, mort, laissé par une
  fusion précédente ; `Visualizer.multicollinearity()` transmettait `**kwargs`
  vers `DataAnalyzer.multicollinearity()` qui n'en accepte pas (échouerait à
  l'exécution si un appelant passait un argument).

### Phase 6 : identité visuelle et galerie d'exemples (2026-07-23)

- Doc Sphinx passée au thème Shibuya (accent vert `grass`, logo et
  mot-symbole SVG dessinés en tracés géométriques, aucune police requise),
  page d'accueil refondue (installation PyPI, badges réels, écosystème),
  bannière reprise dans le README GitHub et sur PyPI. Détail complet dans
  le CHANGELOG (plusieurs bugs de rendu Sphinx corrigés au passage :
  formules `$...$` invalides en reST, docstrings LaTeX non-raw, blocs
  `>>>` fondus dans le texte faute de ligne vide).
- Galerie d'exemples façon scikit-learn/matplotlib, via
  [Sphinx-Gallery](https://sphinx-gallery.github.io/) : onze scripts dans
  `examples/{01_bases,02_donnees_et_modeles,03_production}/plot_*.py`,
  réellement exécutés au moment du build (code, sorties, graphiques
  inclus), téléchargeables en `.py`/`.ipynb`, organisés en trois
  sous-galeries liées depuis la sidebar. Contenu repris et adapté de la
  série pédagogique du dépôt `trainedml-tp` (mêmes 11 étapes, du premier
  modèle au projet complet), reformaté au format cellules Sphinx-Gallery
  (`# %%`) et enrichi de vrais graphiques (`viz.heatmap()`,
  `viz.histogram()`) sur l'exemple d'exploration.
- Deux pièges rencontrés : le motif par défaut de Sphinx-Gallery
  (`'/plot_'`) ne matche jamais sur les chemins Windows à antislashs, donc
  aucun exemple n'était réellement exécuté (juste affiché statiquement)
  tant que le motif n'a pas été rendu indépendant du séparateur ; et les
  scripts qui sauvegardent un fichier (`Trainer.save()`, export CSV)
  s'exécutent avec le dossier source de l'exemple comme répertoire courant,
  polluant `examples/` à chaque build tant qu'ils n'écrivent pas dans un
  dossier temporaire (`tempfile.mkdtemp()`).
- Les scripts historiques à plat de `examples/` (`quickstart.py`,
  `compare_models.py`, `exemple_regression.py`, `rapport_eda.py`) et le
  dossier `examples/notebooks/` ont été retirés : ils recoupaient
  entièrement la nouvelle galerie, qui les remplace avec un meilleur
  résultat (sorties et graphiques réels, notebook généré automatiquement
  pour chaque exemple). `examples/` n'a plus qu'une seule source de
  vérité : `01_bases/`, `02_donnees_et_modeles/`, `03_production/`.

### Chantiers restants

- Consolider les couches de visualisation (`figure.py`, `visualization.py`,
  `analyzer.py`, `viz/`) autour de la façade `Visualizer`.
- Ajouter d'autres datasets intégrés et d'autres visualisations (ROC, scatter).

---

## Conseils d'utilisation

- Les scores parfaits (1.000) sont rares et peuvent indiquer un split "trop
  facile" ou une fuite de données. Utilisez `--seed` et `--test-size` (ou
  `trainer.fit(seed=...)`) pour tester la robustesse, et préférez `--cv` pour
  une évaluation fiable.
- La CLI est le point d'entrée recommandé pour automatiser des workflows ;
  l'API Python pour les usages avancés.

> Note personnelle (pour l'apprentissage) :
> tester plusieurs seeds et augmenter la taille du jeu de test est essentiel
> pour comprendre la robustesse des modèles et éviter de se faire piéger par
> des résultats trop beaux pour être vrais.

---

## Erreurs fréquentes et solutions

### 1. Scores parfaits (1.000 partout)
**Cause possible :** split trop facile, fuite de données, dataset trop simple.
**Solution :** tester plusieurs seeds, augmenter la taille du jeu de test, utiliser `--cv`.

### 2. Erreur "could not convert string to float"
**Cause :** calcul de corrélation ou entraînement sur une colonne non numérique.
**Solution :** depuis la 0.2.0, la heatmap ignore les colonnes non numériques et le
prétraitement du Trainer encode les colonnes catégorielles ; mettre à jour le package.

### 3. AttributeError: 'Axes' object has no attribute 'show'
**Cause :** `fig.show()` sur un objet Axes.
**Solution :** `import matplotlib.pyplot as plt; plt.show()`.

### 4. ImportError: No module named 'trainedml...'
**Cause :** problème d'installation ou d'environnement Python.
**Solution :** vérifier `pip install trainedml` (ou `pip install -e .`) dans le bon environnement.

### 5. Problèmes de reproductibilité
**Cause :** résultats différents à chaque exécution.
**Solution :** fixer la seed (`--seed` ou `seed=` dans le Trainer).

### 6. Erreur de shape ou de colonnes manquantes
**Cause :** mauvais nom de colonne, DataFrame mal préparé.
**Solution :** vérifier les noms de colonnes ; pour `predict`, fournir les mêmes
features que l'entraînement (le préprocesseur sauvegardé s'occupe du reste).

---

## Environnement virtuel (venv)

- **Pourquoi ?** Isoler les dépendances du projet.
- **Commandes :**
  ```bash
  python -m venv venv
  # Activation sous Windows
  .\venv\Scripts\activate
  # Activation sous Linux/Mac
  source venv/bin/activate
  # ou bien si ça ne marche pas :
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  # désactiver
  deactivate
  # supprimer
  Remove-Item -Recurse -Force venv
  # créer avec une version précise de Python
  py -3.11 -m venv venv
  ```
- **Installation :**
  ```bash
  pip install -e ".[dev]"
  # désinstaller
  pip uninstall -y trainedml
  ```

## Documentation Sphinx

Voir [doc/README.md](doc/README.md) pour le build local et la structure.
La doc est publiée automatiquement sur GitHub Pages à chaque push sur main.

Conseils : docstrings complètes NumPy-style avec exemples, corriger tous les
warnings Sphinx, ajouter chaque nouveau module dans `doc/source/modules.rst`.

---

## Auteurs

- diamankayero

## Licence

MIT
