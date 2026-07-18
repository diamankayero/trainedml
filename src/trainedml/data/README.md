# data : chargement des données

Un seul module, `loader.py`, qui isole toute la logique d'accès aux données :
le reste du package ne connaît jamais la provenance (locale, URL, open data).

## DataLoader

- `load_dataset(name=...)` : datasets intégrés `iris` et `wine`, chargés
  **localement** via `sklearn.datasets` (aucun réseau requis). Les noms de
  colonnes d'iris sont au format seaborn (`sepal_length`, ...) pour rester
  compatibles avec les versions antérieures qui téléchargeaient le CSV.
- `load_dataset(url=..., target=...)` : CSV distant, téléchargé et mis en
  cache par `pooch` ; le séparateur `;` est détecté pour les fichiers
  winequality, sinon `sep=` est passable.
- `split(X, y, test_size, random_state)` : séparation train/test. C'est le
  point de passage unique du package ; ni la CLI ni les tests n'appellent
  scikit-learn directement pour splitter.

## Ajouter un dataset intégré

Ajouter un bloc dans `load_dataset` qui retourne `(X: DataFrame, y: Series)`,
et documenter le nom dans le README racine.
