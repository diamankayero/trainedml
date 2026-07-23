trainedml
=========

.. image:: https://img.shields.io/pypi/v/trainedml.svg
   :target: https://pypi.org/project/trainedml/
   :alt: Version PyPI

.. image:: https://img.shields.io/pypi/pyversions/trainedml.svg
   :target: https://pypi.org/project/trainedml/
   :alt: Versions Python

.. image:: https://github.com/diamankayero/trainedml/actions/workflows/workflow.yml/badge.svg
   :target: https://github.com/diamankayero/trainedml/actions
   :alt: CI

.. image:: https://img.shields.io/badge/licence-MIT-green.svg
   :target: https://github.com/diamankayero/trainedml/blob/main/LICENCE
   :alt: Licence MIT

**Du CSV au comparatif de modèles et au rapport EDA en une ligne.**

trainedml est une bibliothèque Python d'apprentissage automatique supervisé
pensée pour l'enseignement, la recherche et le prototypage rapide : chargement
de données, prétraitement automatique, entraînement, comparaison de modèles
par validation croisée et exploration visuelle, le tout derrière une API
unifiée et une CLI.

Installation
------------

Le package est publié sur `PyPI <https://pypi.org/project/trainedml/>`_ :

.. code-block:: bash

   pip install trainedml

Démarrage rapide
----------------

.. code-block:: python

   from trainedml import Trainer, compare

   # Entraîner, évaluer, prédire
   trainer = Trainer(dataset="iris", model="random_forest")
   trainer.fit()
   print(trainer.evaluate())
   y_pred = trainer.predict([[5.1, 3.5, 1.4, 0.2]])

   # Sauvegarder / recharger
   trainer.save("model.joblib")
   restored = Trainer.load("model.joblib")

   # Comparer tous les modèles (validation croisée) en une ligne
   print(compare(dataset="wine", cv=5))

La même chose en ligne de commande :

.. code-block:: bash

   trainedml --dataset iris --model random_forest --show
   trainedml --dataset wine --benchmark --cv 5

Voir aussi la :doc:`galerie d'exemples <auto_examples/01_bases/index>` : des scripts
complets, exécutables et téléchargeables, pour chaque étape d'un projet
(premiers pas, comparaison de modèles, régression, production...).

Fonctionnalités
---------------

- **Données** : datasets intégrés (Iris, Wine...), CSV distants avec cache
  local, données en mémoire ; voir :doc:`DataLoader <trainedml/loader>`.
- **Prétraitement automatique** : imputation, standardisation, encodage
  one-hot, sans fuite d'information en validation croisée ; voir
  :doc:`Prétraitement <trainedml/preprocessing>`.
- **Modèles** : Random Forest, kNN, régression logistique, régresseurs, et
  n'importe quel estimateur scikit-learn ; voir :doc:`Trainer <trainedml/trainer>`.
- **Comparaison** : benchmark de tous les modèles adaptés au dataset, avec
  validation croisée ; voir :doc:`Benchmark <trainedml/benchmark>`.
- **Exploration** : statistiques descriptives, corrélations, valeurs
  manquantes, outliers, normalité, VIF, et rapport EDA HTML auto-contenu ;
  voir :doc:`Visualizer <trainedml/visualization>` et
  :doc:`Rapport EDA <trainedml/report>`.

Écosystème
----------

- `Code source sur GitHub <https://github.com/diamankayero/trainedml>`_
- `Package sur PyPI <https://pypi.org/project/trainedml/>`_
- `Démo web interactive <https://trainedml.onrender.com>`_ : l'API du package
  exposée en HTTP, avec une page d'essai dans le navigateur
- `ModeLmL <https://modelml.vercel.app>`_ : l'atelier web complet construit
  sur trainedml (dashboard, analyse, comparaison, prédiction)

Questions fréquentes
--------------------

**Comment charger mon propre dataset ?**
    Passez l'URL de votre CSV et le nom de la colonne cible à
    :doc:`DataLoader <trainedml/loader>`, ou donnez directement un DataFrame
    en mémoire à ``Trainer``.

**Puis-je utiliser mes propres modèles ?**
    Oui : tout estimateur scikit-learn se passe tel quel
    (``Trainer(model=SVC())``), et un modèle maison s'intègre en héritant de
    ``BaseModel``.

Contribuer et licence
---------------------

Les contributions sont les bienvenues : issues et pull requests sur
`GitHub <https://github.com/diamankayero/trainedml>`_. Le projet est
distribué sous licence MIT.

.. toctree::
   :maxdepth: 2
   :caption: Exemples
   :hidden:

   auto_examples/01_bases/index
   auto_examples/02_donnees_et_modeles/index
   auto_examples/03_production/index

.. toctree::
   :maxdepth: 1
   :caption: Cœur du package
   :hidden:

   trainedml/trainer
   trainedml/loader
   trainedml/preprocessing
   trainedml/tasks
   trainedml/benchmark
   trainedml/evaluation
   trainedml/report

.. toctree::
   :maxdepth: 1
   :caption: Modèles
   :hidden:

   trainedml/models/knn
   trainedml/models/logistic
   trainedml/models/random_forest

.. toctree::
   :maxdepth: 1
   :caption: Exploration et visualisation
   :hidden:

   trainedml/visualization
   trainedml/analyzer
   trainedml/vizs
   trainedml/heatmap
   trainedml/histogram
   trainedml/line
   trainedml/distribution
   trainedml/correlation
   trainedml/missing
   trainedml/outliers
   trainedml/target
   trainedml/boxplot
   trainedml/bivariate
   trainedml/normality
   trainedml/multicollinearity
   trainedml/profiling
