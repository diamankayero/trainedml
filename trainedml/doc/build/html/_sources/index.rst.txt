.. trainedml documentation master file, created by
   sphinx-quickstart on Mon Dec 22 02:19:59 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

trainedml
=========

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/
   :alt: Python Version

**trainedml** est une bibliothèque Python modulaire pour l'apprentissage automatique supervisé, conçue pour l'enseignement, la recherche et le prototypage rapide.

- Chargement flexible de jeux de données publics ou personnalisés
- Modèles classiques (Random Forest, KNN, Régression Logistique)
- Visualisations intégrées (heatmap, histogramme, courbe)
- API simple pour l'intégration web/applications
- Documentation API complète et tests unitaires

.. contents:: Sommaire
   :depth: 2
   :local:

Guide de démarrage rapide
=========================

.. code-block:: python

   from trainedml import Trainer
   trainer = Trainer(dataset="iris", model="random_forest")
   trainer.fit()
   print(trainer.evaluate())
   y_pred = trainer.predict([[5.1, 3.5, 1.4, 0.2]])

Installation
============

.. code-block:: bash

   pip install -r requirements.txt
   pip install .

Utilisation en ligne de commande
================================

.. code-block:: bash

   python -m trainedml --dataset iris --model random_forest --show


Sections de la documentation
============================

.. toctree::
   :maxdepth: 2
   :caption: API

   modules

Explications mathématiques des méthodes
=======================================


.. rubric:: Explications mathématiques des méthodes


.. note::
   Les explications mathématiques détaillées pour chaque méthode sont disponibles dans la section API détaillée :

   - :doc:`k-Nearest Neighbors (kNN) <trainedml/models/knn>`
   - :doc:`Régression Logistique <trainedml/models/logistic>`
   - :doc:`Random Forest <trainedml/models/random_forest>`

   (voir aussi "modules" dans le menu de gauche)

.. comment:: Les blocs mathématiques sont désormais déplacés dans les pages dédiées de l'API.


FAQ
---

**Q : Comment ajouter un nouveau dataset ?**

R : Utilisez la classe DataLoader avec l'URL de votre CSV et le nom de la colonne cible.

**Q : Puis-je utiliser mes propres modèles ?**

R : Oui, il suffit d'implémenter la classe BaseModel et de l'ajouter à MODEL_MAP.

Contribuer
----------

Les contributions sont les bienvenues ! Merci de soumettre vos issues et pull requests sur GitHub.

Licence
-------

Ce projet est distribué sous licence MIT.

Partie 2 : Industrialisation et documentation avancée
======================================================

Cette section détaille les étapes d’industrialisation et d’amélioration de la documentation du projet trainedml, réalisées avec Sphinx.

Refactorisation et flexibilité
-------------------------------
- Refactorisation du code pour permettre le chargement de n’importe quel dataset (pas seulement iris).
- Ajout d’une classe Trainer centralisant le workflow ML (chargement, split, fit, évaluation, prédiction).
- Intégration de pooch pour le cache et le téléchargement des datasets distants.
- Séparation claire entre les modules : data, models, viz, tests, etc.

Tests et robustesse
--------------------
- Correction et harmonisation de tous les tests unitaires pour la nouvelle API.
- Prise en charge des datasets personnalisés dans les tests.
- Ajout de tests pour la visualisation (heatmap, histogramme, courbe).

Documentation Sphinx professionnelle
-------------------------------------
- Initialisation de la doc Sphinx dans `doc/source`.
- Activation des extensions autodoc et napoleon pour générer la doc API à partir des docstrings.
- Ajout d’un sommaire API (toctree) incluant tous les modules : data, models, viz, etc.
- Passage au thème sphinx_rtd_theme pour un rendu moderne et professionnel.
- Correction des titres et soulignements pour éliminer tous les warnings Sphinx.
- Structuration de la page d’accueil : badge, résumé, guide rapide, installation, CLI, FAQ, contribution, licence.

Conseils pour une doc parfaite
------------------------------
- Utiliser le style Google ou NumPy pour les docstrings (meilleure lisibilité dans Sphinx).
- Ajouter des exemples d’utilisation dans chaque module.
- Générer la doc avec : `make.bat html` puis ouvrir `build/html/index.html`.
- Pour publier en ligne : utiliser ReadTheDocs ou GitHub Pages.

Résumé
------
Le projet trainedml est maintenant industrialisé, modulaire, testé et documenté comme un vrai projet open source professionnel. La documentation Sphinx est complète, moderne et extensible.