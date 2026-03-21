k-Nearest Neighbors (kNN)
=========================

.. currentmodule:: trainedml.models.knn

Principe (kNN)
--------------

Le kNN est un algorithme de classification supervisée basé sur la proximité des exemples dans l’espace des caractéristiques. Pour une observation :math:`x`, on recherche les :math:`k` voisins les plus proches selon une distance (souvent euclidienne) :

.. math::
   d(x, x_i) = \sqrt{\sum_{j=1}^p (x_j - x_{i,j})^2}

La classe prédite est la plus fréquente parmi les :math:`k` voisins :

.. math::
   \hat{y} = \operatorname{mode}\left(\{y_i\}_{i \in V_k(x)}\right)

.. math::

   V_k(x)\ \text{ est l’ensemble des } k \text{ plus proches voisins de } x.

**Avantages** : simple, non paramétrique, efficace pour les petits jeux de données.

**Limites** : sensible à l’échelle des variables, coûteux pour de grands jeux de données.

Illustration (kNN)
------------------

.. image:: ../../_static/figures/knn.png
   :alt: Schéma kNN
   :width: 400px
   :align: center

Exemple illustré (kNN)
----------------------

Supposons un jeu de données avec trois points :math:`A=(1,2)`, :math:`B=(2,3)`, :math:`C=(4,2)` et une nouvelle observation :math:`X=(2,2)`. Pour :math:`k=2`, les deux plus proches voisins de :math:`X` sont :math:`A` et :math:`B`. Si :math:`A` est de classe 0 et :math:`B` de classe 1, la classe prédite sera la plus fréquente parmi ces deux voisins.

Pour aller plus loin (kNN)
--------------------------

- Documentation scikit-learn : https://scikit-learn.org/stable/modules/neighbors.html
- Cours OpenClassrooms : https://openclassrooms.com/fr/courses/4425111-initiez-vous-au-machine-learning/5028281-classifiez-avec-les-k-plus-proches-voisins-knn
