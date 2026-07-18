# webapp_api : API HTTP + page de démo

Le pattern standard de l'industrie : le Python (trainedml) tourne côté
serveur derrière une API HTTP (FastAPI) ; n'importe quel frontend (la page
HTML/JS fournie, React, mobile...) consomme cette API. Le frontend peut
changer, l'API ne bouge pas.

## Lancement

```bash
pip install -e ".[web]"        # fastapi + uvicorn
uvicorn webapp_api.api:app --reload
# puis ouvrir http://localhost:8000
```

## Fichiers

- `api.py` : l'application FastAPI et ses routes
- `static/index.html` : page de démo autonome (HTML + CSS + JS vanilla,
  aucune dépendance ni build) qui appelle l'API avec `fetch`

## Routes

| Route | Corps | Retour |
|---|---|---|
| `GET /api/models` | - | classificateurs et régresseurs disponibles |
| `POST /api/train` | dataset ou url+target, model, model_params, test_size, seed | scores, tâche, features |
| `POST /api/predict` | features (lignes à prédire) | prédictions du dernier modèle entraîné |
| `POST /api/compare` | dataset ou url+target, cv, seed | tableau comparatif (validation croisée) |
| `GET /docs` | - | documentation interactive générée par FastAPI (Swagger) |
| `GET /` | - | page de démo |

## Limites assumées (démo)

- Un seul modèle en mémoire à la fois (dictionnaire `STATE`) ; pas de
  persistance entre redémarrages. Pour un vrai déploiement : `Trainer.save()`
  au train et `Trainer.load()` au démarrage, ou un stockage par utilisateur.
- CORS ouvert à tous les domaines ; à restreindre en production.
- Pas d'authentification.

## Passer à React plus tard

L'API reste identique. Un frontend React (Vite) en dev sur
http://localhost:5173 peut appeler http://localhost:8000/api/... directement
(le CORS est déjà ouvert) ; en production, builder le frontend et servir le
dossier de build à la place de `static/`.

## Mettre en ligne

Le principe : la même commande uvicorn, mais sur une machine allumée 24h/24
avec une URL publique. Deux fichiers à la racine du repo préparent ça.

### Render (recommandé, gratuit)

Le fichier `render.yaml` est détecté automatiquement :

1. Créer un compte sur https://render.com (connexion via GitHub).
2. New > Blueprint > choisir le repo `diamankayero/trainedml`.
3. Render lit render.yaml et propose le service `trainedml-api` : valider.
4. À la fin du build, l'URL publique (https://trainedml-api.onrender.com ou
   proche) sert la page de démo et l'API.

Chaque push sur main redéploie automatiquement. Plan gratuit : le service
s'endort après 15 min d'inactivité, la première visite suivante prend ~30 s.

### Docker (tout hébergeur)

Le `Dockerfile` à la racine fonctionne sur Fly.io, Railway, Hugging Face
Spaces (Space de type Docker), ou un VPS :

```bash
docker build -t trainedml-api .
docker run -p 8000:8000 trainedml-api
```

L'hébergeur fournit la variable d'environnement `PORT` ; l'image l'utilise
automatiquement (8000 par défaut en local).

## Tests

`tests/test_api.py` couvre toutes les routes avec le TestClient FastAPI
(ignorés si fastapi n'est pas installé).
