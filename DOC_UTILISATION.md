---

## Gestion des branches Git et déploiement

### Workflow recommandé

- **Branche principale (`main`)** : contient la version stable et validée du projet. C'est celle à utiliser pour le déploiement ou la présentation.
- **Branche de développement (`dev`)** : sert à développer de nouvelles fonctionnalités, corriger des bugs ou expérimenter sans impacter la version stable.
- **Branches de fonctionnalités (`feature/nom`)** : pour chaque nouvelle fonctionnalité ou amélioration, crée une branche dédiée à partir de `dev`.

#### Exemple de cycle de travail
1. **Développement** :
   - Travaille sur la branche `dev` ou sur une branche `feature/ma-fonctionnalite`.
   - Commits réguliers pour suivre l'avancement.
2. **Fusion** :
   - Une fois la fonctionnalité testée et validée, fusionne la branche `feature/ma-fonctionnalite` dans `dev`.
   - Quand la version sur `dev` est stable, fusionne `dev` dans `main` pour publier.
3. **Déploiement** :
   - Déploie la branche `main` sur Streamlit Cloud ou configure GitHub Actions pour automatiser les tests et le déploiement.

#### Commandes utiles
```bash
git checkout dev                # Passe sur la branche de développement
git checkout -b feature/ma-fonctionnalite  # Crée une branche de fonctionnalité
git add . && git commit -m "Ajout fonctionnalité"
git push -u origin feature/ma-fonctionnalite
# Fusionne la fonctionnalité dans dev
git checkout dev
git merge feature/ma-fonctionnalite
git push
# Fusionne dev dans main pour publier
git checkout main
git merge dev
git push
```

#### Déploiement Streamlit Cloud
- Connecte ton dépôt GitHub à [Streamlit Cloud](https://streamlit.io/cloud).
- Choisis la branche `main` pour le déploiement.
- Configure le fichier `requirements.txt` pour installer les dépendances.
- Lance l'app (exemple : `trainedml_webapp/src/app.py`).

#### Automatisation avec GitHub Actions
- Ajoute un fichier `.github/workflows/ci.yml` pour automatiser les tests et le déploiement.
- Exemple de workflow : installation des dépendances, exécution des tests, déploiement automatique.

---
