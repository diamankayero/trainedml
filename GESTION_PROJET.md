# Guide de gestion de projet trainedml

Ce document explique comment gérer le cycle de vie du projet, collaborer efficacement, déployer l’application et automatiser les tests.

## Sommaire
- [Branches Git et workflow](#branches-git-et-workflow)
- [Cycle de travail collaboratif](#cycle-de-travail-collaboratif)
- [Déploiement Streamlit Cloud](#déploiement-streamlit-cloud)
- [Automatisation CI/CD (GitHub Actions)](#automatisation-cicd-github-actions)
- [Bonnes pratiques et conventions](#bonnes-pratiques-et-conventions)
- [Fichiers de workflow disponibles](#fichiers-de-workflow-disponibles)

---

## Branches Git et workflow

- **main** : version stable, production, déploiement.
- **dev** : développement courant, intégration de nouvelles fonctionnalités.
- **feature/xxx** : une branche par fonctionnalité ou correction.

## Cycle de travail collaboratif

1. Crée une branche à partir de `dev` :
   ```bash
   git checkout dev
   git checkout -b feature/ma-fonctionnalite
   ```
2. Développe, committe régulièrement :
   ```bash
   git add .
   git commit -m "Ajout fonctionnalité"
   git push -u origin feature/ma-fonctionnalite
   ```
3. Ouvre une Pull Request vers `dev`.
4. Après validation, fusionne dans `dev` puis dans `main` pour publier.

## Déploiement Streamlit Cloud

- Connecte le dépôt à [Streamlit Cloud](https://streamlit.io/cloud).
- Déploie la branche `main`.
- Vérifie que `requirements.txt` est à jour.
- Chemin d’entrée recommandé : `trainedml_webapp/src/app.py`.

## Automatisation CI/CD (GitHub Actions)

- Utilise `.github/workflows/ci.yml` pour automatiser tests et déploiement.
- Exemple de pipeline :
  - Installation des dépendances
  - Exécution des tests
  - Déploiement automatique si succès

## Bonnes pratiques et conventions

- Respecte la structure du projet et les conventions de nommage.
- Documente chaque module/fonction avec des docstrings.
- Mets à jour les README.md de chaque dossier si tu ajoutes/modifies des fonctionnalités.
- Utilise des messages de commit clairs et explicites.
- Pour toute question, consulte le README principal ou ouvre une issue.

## Fichiers de workflow disponibles

Le dossier `.github/workflows/` contient plusieurs workflows GitHub Actions :

- **ci.yml** :
  - Exécute les tests unitaires à chaque push ou pull request sur `main` ou `dev`.
  - Installe les dépendances et lance `pytest`.
- **docs.yml** :
  - Génère et déploie automatiquement la documentation Sphinx sur GitHub Pages à chaque push sur `main`.
  - Installe Sphinx et les thèmes nécessaires, puis publie le dossier HTML généré.

Adapte ou complète ces workflows selon tes besoins (tests avancés, linting, build, etc.).

---

Pour l’utilisation fonctionnelle du package (exemples, API, visualisations), voir le fichier `trainedml/trainedml/DOC_UTILISATION.md`.
