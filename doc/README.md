# doc : documentation Sphinx

Documentation API générée automatiquement depuis les docstrings (autodoc +
napoleon), publiée sur GitHub Pages à chaque push sur main par le workflow
`.github/workflows/docs.yml` : https://diamankayero.github.io/trainedml/

## Construire en local

```bash
pip install sphinx sphinx_rtd_theme
cd doc
make html          # Linux/Mac
.\make.bat html    # Windows
# resultat : doc/build/html/index.html
```

## Structure de source/

- `conf.py` : configuration (thème, extensions, chemin vers src/)
- `index.rst` : page d'accueil, quickstart, FAQ
- `modules.rst` : sommaire de l'API détaillée
- `trainedml/*.rst` : une page par module (automodule). Les pages ajoutées en
  0.2.0 : trainer, loader, preprocessing, tasks, benchmark, evaluation, report

## Ajouter un module à la doc

1. Créer `source/trainedml/<module>.rst` avec un bloc `.. automodule::`.
2. L'ajouter au toctree de `source/modules.rst`.
3. Vérifier le build local sans warning avant de pousser.
