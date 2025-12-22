Régression logistique
=====================

.. currentmodule:: trainedml.models.logistic

Principe (Régression logistique)
--------------------------------

La régression logistique est un modèle linéaire pour la classification binaire. Elle modélise la probabilité d’appartenance à la classe 1 par la fonction sigmoïde :

.. math::
   P(y=1\mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}

Les paramètres :math:`(w, b)` sont estimés par maximisation de la log-vraisemblance :

.. math::
   \ell(w, b) = \sum_{i=1}^n \left[ y_i \log \sigma(w^T x_i + b) + (1-y_i) \log (1-\sigma(w^T x_i + b)) \right]

**Avantages** : interprétable, rapide, probabiliste.

**Limites** : linéaire, sensible aux outliers, suppose l’indépendance des variables explicatives.

Illustration (Régression logistique)
------------------------------------

.. image:: ../../_static/figures/building-predictive-models-logistic-regression-in-python_01.png
   :alt: Courbe sigmoïde
   :width: 400px
   :align: center

Exemple illustré (Régression logistique)
----------------------------------------

Supposons que l’on veuille prédire si un étudiant réussit un examen (1) ou non (0) en fonction de son nombre d’heures de révision :math:`x`. La régression logistique modélise la probabilité de réussite en fonction de :math:`x` et ajuste la courbe sigmoïde pour séparer les deux classes.

Pour aller plus loin (Régression logistique)
--------------------------------------------

- Documentation scikit-learn : https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Article Wikipedia : https://fr.wikipedia.org/wiki/R%C3%A9gression_logistique
