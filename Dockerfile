# Image de déploiement de l'API web trainedml.
# Fonctionne sur tout hébergeur acceptant Docker (Fly.io, Railway,
# Hugging Face Spaces, VPS...).
#
#   docker build -t trainedml-api .
#   docker run -p 8000:8000 trainedml-api
#   # puis ouvrir http://localhost:8000

FROM python:3.11-slim

WORKDIR /app

# Dépendances d'abord (couche mise en cache tant que pyproject ne change pas)
COPY pyproject.toml README.md LICENCE ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"

COPY webapp_api ./webapp_api

EXPOSE 8000
# $PORT est fourni par la plupart des hébergeurs ; 8000 par défaut en local
CMD ["sh", "-c", "uvicorn webapp_api.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
