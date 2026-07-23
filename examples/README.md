# examples : galerie d'exemples

Onze scripts exécutables et narrés, organisés en trois sections, du premier
modèle à la mise en production. C'est la source de la **galerie d'exemples**
de la documentation (https://diamankayero.github.io/trainedml/), générée par
[Sphinx-Gallery](https://sphinx-gallery.github.io/) : chaque script `plot_*.py`
y est réellement exécuté au moment du build (sorties et graphiques inclus),
puis proposé au téléchargement en `.py` et en notebook `.ipynb` généré
automatiquement.

Ils se lancent aussi directement depuis la racine du projet, sans rien
d'autre que le package installé (`pip install trainedml` ou `pip install -e .`) :

```bash
python examples/01_bases/plot_premiers_pas.py
```

## Sections

| Dossier | Contenu |
|---|---|
| [`01_bases/`](01_bases/) | Premier modèle, découpage train/test |
| [`02_donnees_et_modeles/`](02_donnees_et_modeles/) | Comparer des modèles, explorer un dataset, données distantes, régression, données en mémoire, hyperparamètres |
| [`03_production/`](03_production/) | Rigueur et reproductibilité, sauvegarde/déploiement, projet complet de bout en bout |

Le sommaire détaillé de chaque section est dans son `README.txt` (repris tel
quel comme en-tête de la sous-galerie dans la doc).
