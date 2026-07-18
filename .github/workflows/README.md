# workflows : intégration continue et publication

## workflows/workflow.yml : CI + publication PyPI

Déclenché sur push/PR vers main et dev, et sur les tags `v*`.

- **Job test** (matrice) : Ubuntu avec Python 3.9, 3.10, 3.11, 3.12, 3.13
  et Windows avec Python 3.11. Étapes : install (`pip install -e ".[dev]"`),
  lint bloquant (`ruff check --select E9,F src tests`), typage (`mypy`),
  tests (`pytest tests/ -v`). Un échec de test fait échouer le job (ne jamais
  remettre de `|| echo` qui avale les erreurs).
- **Job publish-to-pypi** : uniquement sur les tags `v*` et si les tests
  passent. Build (`python -m build`), vérification (`twine check`), upload
  vers PyPI avec le secret `PYPI_TOKEN` (Settings > Secrets du repo).

## workflows/docs.yml : documentation

Sur chaque push vers main : build Sphinx (`doc/`) et déploiement sur la
branche gh-pages, servie par GitHub Pages.

## Publier une nouvelle version

1. Mettre à jour `version` dans `pyproject.toml` et `__version__` dans
   `src/trainedml/__init__.py`, compléter `CHANGELOG.md`.
2. Commit + push, attendre que la CI soit verte.
3. `git tag vX.Y.Z && git push --tags` : la publication PyPI est automatique.
4. Créer la Release GitHub à partir du tag avec la section du CHANGELOG.
