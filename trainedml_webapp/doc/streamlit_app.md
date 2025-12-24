# Documentation de l'application Streamlit `trainedml_webapp`

## Présentation générale

Cette application Streamlit permet de comparer plusieurs modèles de machine learning (KNN, Régression Logistique, Random Forest) sur des jeux de données classiques (Iris, Wine) ou des jeux de données personnalisés (CSV uploadé ou par URL). Elle propose :
- Benchmark multi-modèles avec visualisation des scores et temps d'entraînement
- Prédiction manuelle sur un échantillon personnalisé
- Visualisations automatiques (heatmap de corrélation, histogrammes)

## Structure du code principal (`app.py`)

### Imports et configuration
- **Import des librairies** : matplotlib, pandas, numpy, scikit-learn, streamlit
- **Import du package trainedml** : data loader, modèles, visualisation, évaluation
- **Ajout du chemin src/** : permet d'importer le package trainedml en mode développement

### Interface utilisateur
- **Sidebar** :
  - Choix du dataset (iris, wine)
  - Upload ou URL CSV
  - Sélection des modèles à comparer
  - Paramètres de split train/test
  - Bouton "Entraîner et comparer"
- **Colonne principale** :
  - Infos sur le dataset (shape, classes, preview)
  - Résultats du benchmark (tableau, barplots)
  - Formulaire de prédiction manuelle
  - Visualisations (heatmap, histogrammes)

### Fonctionnement détaillé

#### Chargement des données
- Si un CSV est uploadé ou une URL fournie, on charge ce fichier (la dernière colonne est la cible).
- Sinon, on charge un dataset intégré via `DataLoader` (iris ou wine).

#### Encodage des labels
- Si la cible est catégorielle, on utilise `LabelEncoder` pour transformer les labels en entiers (utile pour scikit-learn).
- On garde le `LabelEncoder` pour pouvoir réafficher le label original lors de la prédiction.

#### Sélection et entraînement des modèles
- L'utilisateur choisit un ou plusieurs modèles à comparer.
- Les modèles sont instanciés via une **factory** (`get_model`) : cela permet d'ajouter facilement de nouveaux modèles sans modifier le code principal.
- Pour chaque modèle sélectionné :
  - Entraînement sur X_train, y_train
  - Prédiction sur X_test
  - Calcul des scores (accuracy, precision, recall, f1, temps)
- Les résultats sont affichés dans un tableau et sous forme de barplots.

#### Prédiction manuelle
- L'utilisateur saisit les valeurs d'un échantillon via un formulaire dynamique.
- Le modèle sélectionné prédit la classe correspondante.
- Si un `LabelEncoder` a été utilisé, la prédiction est reconvertie en label original pour l'affichage.
- Le code est générique : il fonctionne pour tout dataset, même inconnu.

#### Visualisations
- Heatmap de corrélation sur toutes les features
- Histogrammes pour chaque variable numérique

### Pourquoi une factory pour les modèles ?
La factory (`get_model`) permet d'instancier dynamiquement un modèle à partir de son nom. Cela rend le code principal plus simple, évite les if/else à rallonge, et facilite l'ajout de nouveaux modèles. Exemple :
```python
from trainedml.utils.factory import get_model
model = get_model("KNN")
```

## Exemple de code clé expliqué

```python
# Sélection des modèles
models_selected = st.sidebar.multiselect(
    "Choisir les modèles",
    ["KNN", "Logistic Regression", "Random Forest"],
    default=["KNN"]
)

# Entraînement et benchmark
if st.sidebar.button("Entraîner et comparer"):
    for model_name in models_selected:
        model = get_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # ... calcul des scores ...
```

- `get_model(model_name)` : retourne une instance du modèle demandé (KNN, Logistic, Random Forest, etc.)
- `model.fit(X_train, y_train)` : entraîne le modèle
- `model.predict(X_test)` : prédit les classes sur le jeu de test

## Bonnes pratiques
- Le code est générique et s'adapte à tout dataset compatible (pas seulement iris ou wine)
- L'affichage des prédictions reconvertit toujours l'encodage en label original pour une meilleure lisibilité
- L'utilisation de la factory rend l'ajout de nouveaux modèles très simple

---

Pour toute question ou contribution : [https://github.com/diamankayero/trainedml](https://github.com/diamankayero/trainedml)

---

# Explication détaillée du code `app.py`

## 1. Imports et configuration
```python
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from trainedml.data.loader import DataLoader
from trainedml.models.knn import KNNModel
from trainedml.models.logistic import LogisticModel
from trainedml.models.random_forest import RandomForestModel
from trainedml.visualization import Visualizer
from trainedml.evaluation import Evaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
```
- **matplotlib.pyplot** : pour les graphiques/barplots
- **streamlit** : framework web interactif
- **pandas, numpy** : manipulation de données
- **trainedml.*** : modules internes (data, modèles, visualisation, évaluation)
- **scikit-learn** : split, encodage, modèles
- **sys.path.insert...** : permet d'importer le package trainedml en mode développement

## 2. Mise en forme et logo
```python
st.markdown("""<style>...</style>""", unsafe_allow_html=True)
st.image(...)
st.title(...)
st.markdown(...)
```
- Ajoute du CSS custom pour le style
- Affiche un logo et un titre explicite

## 3. Sidebar : configuration utilisateur
```python
st.sidebar.header("Configuration")
st.sidebar.markdown(...)
dataset_name = st.sidebar.selectbox(...)
user_url = st.sidebar.text_input(...)
sep = st.sidebar.selectbox(...)
user_file = st.sidebar.file_uploader(...)
```
- **selectbox** : choix du dataset intégré (iris, wine)
- **text_input**/**file_uploader** : possibilité de charger un CSV distant ou local
- **selectbox** : choix du séparateur CSV

## 4. Chargement des données
```python
if user_url:
    ... # charge le CSV distant
elif user_file is not None:
    ... # charge le CSV uploadé
if df_user is not None:
    X = df_user.iloc[:, :-1]
    y = df_user.iloc[:, -1]
    dataset_name = ...
else:
    loader = DataLoader()
    X, y = loader.load_dataset(name=dataset_name)
```
- Si un CSV est fourni, on le charge (la dernière colonne est la cible)
- Sinon, on charge un dataset intégré via DataLoader

## 5. Encodage des labels
```python
le = None
if y.dtype == object or y.dtype.name == "category" or (hasattr(y, 'dtype') and str(y.dtype).startswith('float')):
    le = LabelEncoder()
    y = le.fit_transform(y)
    class_names = [str(c) for c in le.classes_]
else:
    class_names = [str(c) for c in np.unique(y)]
```
- Si la cible est catégorielle, on encode les labels (LabelEncoder)
- On garde la correspondance pour l'affichage ultérieur

## 6. Sélection des modèles et split
```python
from trainedml.utils.factory import get_model
models_selected = st.sidebar.multiselect(...)
seed = st.sidebar.number_input(...)
test_size = st.sidebar.slider(...)
X_train, X_test, y_train, y_test = train_test_split(...)
```
- **multiselect** : choix des modèles à comparer
- **number_input/slider** : paramètres de split
- **train_test_split** : division des données en train/test

## 7. Layout principal (2 colonnes)
```python
col1, col2 = st.columns([1, 2])
```
- **col1** : infos dataset (shape, classes, preview)
- **col2** : benchmark, prédiction, visualisations

### Colonne 1 : infos dataset
```python
with col1:
    st.markdown(...)
    st.write(...)
    st.dataframe(X.head())
```
- Affiche les infos principales du dataset

### Colonne 2 : benchmark, prédiction, visualisations

#### a) Entraînement et benchmark
```python
if 'trained_models' not in st.session_state:
    st.session_state['trained_models'] = {}
trained_models = st.session_state['trained_models']

if st.sidebar.button("Entraîner et comparer"):
    results = []
    new_trained_models = {}
    with st.spinner(...):
        for model_name in models_selected:
            model = get_model(model_name)
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            t1 = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - t1
            scores = Evaluator.evaluate_all(y_test, y_pred)
            results.append({...})
            new_trained_models[model_name] = model
    st.session_state['trained_models'] = new_trained_models
    trained_models = new_trained_models
    st.success(...)
    df_results = pd.DataFrame(results)
    st.markdown(...)
    st.dataframe(df_results)
    # Barplots
    fig_acc, ax_acc = plt.subplots()
    df_results.plot(...)
    st.pyplot(fig_acc)
    fig_time, ax_time = plt.subplots()
    df_results.plot(...)
    st.pyplot(fig_time)
```
- **st.session_state** : stocke les modèles entraînés pour la session (permet la prédiction manuelle ensuite)
- **for model_name in models_selected** : boucle sur chaque modèle choisi
- **get_model(model_name)** : instancie dynamiquement le modèle (factory)
- **fit/predict** : entraînement et prédiction
- **Evaluator.evaluate_all** : calcule tous les scores pertinents
- **results.append** : stocke les résultats pour affichage
- **Barplots** : visualisation des scores et temps d'entraînement

#### b) Prédiction manuelle
```python
st.subheader("Prédiction manuelle")
st.markdown(...)
input_features = list(X.columns)
user_inputs = {}
for feat in input_features:
    val = st.number_input(f"{feat}", value=float(X[feat].mean()))
    user_inputs[feat] = val
model_pred_name = st.selectbox(...)
if st.button("Prédire"):
    trained_models = st.session_state.get('trained_models', {})
    if model_pred_name in trained_models:
        model_pred = trained_models[model_pred_name]
        arr = np.array([list(user_inputs.values())]).reshape(1, -1)
        try:
            pred = model_pred.predict(arr)
            if le is not None:
                label = le.inverse_transform([pred[0]])[0]
            else:
                label = str(pred[0])
            st.success(f"Prédiction : {label}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
    else:
        st.warning("Veuillez d'abord entraîner et comparer les modèles.")
```
- **for feat in input_features** : crée dynamiquement un champ pour chaque feature du dataset
- **st.number_input** : permet de saisir une valeur numérique pour chaque feature
- **model_pred_name = st.selectbox(...)** : choix du modèle pour la prédiction
- **if st.button("Prédire")** : lance la prédiction manuelle
- **model_pred.predict(arr)** : effectue la prédiction sur l'échantillon saisi
- **le.inverse_transform** : reconvertit l'encodage en label original si besoin

#### c) Visualisations
```python
viz = Visualizer(pd.concat([X, pd.Series(y, name='target')], axis=1))
st.subheader("Heatmap de corrélation")
fig = viz.heatmap(features=list(X.columns))
if hasattr(fig, 'figure'):
    st.pyplot(fig.figure)
else:
    st.pyplot(fig)
st.subheader("Histogramme des variables numériques")
fig2 = viz.histogram(columns=list(X.columns), legend=True)
if hasattr(fig2, 'figure'):
    st.pyplot(fig2.figure)
else:
    st.pyplot(fig2)
```
- **Visualizer** : outil interne pour générer heatmap et histogrammes
- **pd.concat([X, y])** : fusionne features et cible pour la visualisation
- **st.pyplot** : affiche les graphiques matplotlib

## 8. Footer et aide
```python
st.sidebar.markdown("---")
st.sidebar.info(...)
st.sidebar.markdown(...)
```
- Affiche un encadré d'aide pour l'utilisateur (ex : comment récupérer un lien GitHub Raw)

---

## Pourquoi la factory ?
La factory (`get_model`) permet d'instancier dynamiquement un modèle à partir de son nom, ce qui rend le code principal plus simple, modulaire et évolutif. On peut ajouter de nouveaux modèles sans toucher à la logique de l'app.

---

**Ce document explique chaque bloc, boucle et fonction clé du code Streamlit. Pour toute question sur une ligne précise, indique-la et je détaillerai encore plus !**
