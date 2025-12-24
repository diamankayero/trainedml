# Explication ligne par ligne du code app.py

---

## 1. Imports et configuration

```python
import matplotlib.pyplot as plt  # Pour les graphiques matplotlib
# -*- coding: utf-8 -*-  # Encodage du fichier (UTF-8)
import streamlit as st  # Framework web interactif
import pandas as pd  # Manipulation de données tabulaires
from trainedml.data.loader import DataLoader  # Chargement de jeux de données intégrés
from trainedml.models.knn import KNNModel  # Modèle K plus proches voisins
from trainedml.models.logistic import LogisticModel  # Modèle de régression logistique
from trainedml.models.random_forest import RandomForestModel  # Modèle forêt aléatoire
from trainedml.visualization import Visualizer  # Outils de visualisation (heatmap, histogramme)
from trainedml.evaluation import Evaluator  # Outils d'évaluation (accuracy, f1, etc.)
from sklearn.model_selection import train_test_split  # Split train/test
from sklearn.preprocessing import LabelEncoder  # Encodage des labels catégoriels
import time  # Mesure du temps d'exécution
import numpy as np  # Calcul scientifique
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
```
- Ces lignes importent toutes les librairies nécessaires pour l'app, y compris les modules internes du package trainedml.
- `sys.path.insert...` permet d'importer le package trainedml en mode développement (src/).

---

## 2. Mise en forme et logo

```python
st.markdown("""<style>...</style>""", unsafe_allow_html=True)
```
- Ajoute du CSS personnalisé pour le style de l'app.

```python
st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
```
- Affiche un logo en haut de la page.

```python
st.title("Démo trainedml : Comparaison de modèles ML")
st.markdown("""<span style='font-size:1.2rem;'>Comparez facilement plusieurs modèles de machine learning sur des jeux de données classiques.</span>""", unsafe_allow_html=True)
```
- Affiche le titre et un sous-titre explicatif.

---

## 3. Sidebar : configuration utilisateur

```python
st.sidebar.header("Configuration")
st.sidebar.markdown("""<div style='background:#e0e7ff;...'>...</div>""", unsafe_allow_html=True)
```
- Affiche un encadré d'identité dans la sidebar.

```python
dataset_name = st.sidebar.selectbox("Choisir un dataset", ("iris", "wine"))
```
- Permet de choisir un dataset intégré (iris ou wine).

```python
st.sidebar.markdown("<b>Ou charger un CSV par URL :</b>", unsafe_allow_html=True)
user_url = st.sidebar.text_input("URL d'un CSV (optionnel)")
sep = st.sidebar.selectbox("Séparateur CSV", [",", ";", "\t"], ...)
user_file = st.sidebar.file_uploader("Uploader un CSV (optionnel)", type=["csv"])
```
- Permet de charger un CSV distant (URL) ou local (upload), et de choisir le séparateur.

---

## 4. Chargement des données

```python
df_user = None
if user_url:
    try:
        df_user = pd.read_csv(user_url, sep=sep)
        st.sidebar.success(f"Données chargées depuis l'URL !")
    except Exception as e:
        st.sidebar.error(f"Erreur de chargement : {e}...)
elif user_file is not None:
    df_user = pd.read_csv(user_file, sep=sep)
    st.sidebar.success(f"Fichier chargé : {user_file.name}")
if df_user is not None:
    X = df_user.iloc[:, :-1]
    y = df_user.iloc[:, -1]
    dataset_name = f"Upload: {user_file.name if user_file else user_url}"
else:
    loader = DataLoader()
    X, y = loader.load_dataset(name=dataset_name)
```
- Si un CSV est fourni, on le charge (la dernière colonne est la cible y, le reste X).
- Sinon, on charge un dataset intégré via DataLoader.

---

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
- Si la cible est catégorielle, on encode les labels (LabelEncoder) pour scikit-learn.
- On garde la correspondance pour l'affichage ultérieur.

---

## 6. Sélection des modèles et split

```python
from trainedml.utils.factory import get_model
models_selected = st.sidebar.multiselect("Choisir les modèles", ["KNN", "Logistic Regression", "Random Forest"], default=["KNN"])
seed = st.sidebar.number_input("Seed", value=42)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
```
- Permet de choisir un ou plusieurs modèles à comparer.
- Paramètres de split train/test.
- Split effectif des données.

---

## 7. Layout principal (2 colonnes)

```python
col1, col2 = st.columns([1, 2])
```
- Crée deux colonnes pour l'affichage principal.

### Colonne 1 : infos dataset

```python
with col1:
    st.markdown("<b>Infos dataset :</b>", unsafe_allow_html=True)
    st.write(f"Nombre d'échantillons : {X.shape[0]}")
    st.write(f"Nombre de variables : {X.shape[1]}")
    st.write(f"Classes : {', '.join(class_names)}")
    st.dataframe(X.head())
```
- Affiche les infos principales du dataset (shape, classes, preview).

### Colonne 2 : benchmark, prédiction, visualisations

#### a) Entraînement et benchmark

```python
with col2:
    if 'trained_models' not in st.session_state:
        st.session_state['trained_models'] = {}
    trained_models = st.session_state['trained_models']

    if st.sidebar.button("Entraîner et comparer"):
        results = []
        new_trained_models = {}
        with st.spinner("Entraînement des modèles..."):
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
        st.success("Comparaison terminée !")
        df_results = pd.DataFrame(results)
        st.markdown("<div class='score-box'><b>Comparaison des modèles :</b></div>", unsafe_allow_html=True)
        st.dataframe(df_results)
        # Barplot Accuracy avec matplotlib
        fig_acc, ax_acc = plt.subplots()
        df_results.plot(x="Modèle", y="Accuracy", kind="bar", legend=False, ax=ax_acc, color="#2563eb")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Modèle")
        ax_acc.set_title("Comparaison des Accuracy")
        st.pyplot(fig_acc)
        # Barplot Train time avec matplotlib
        fig_time, ax_time = plt.subplots()
        df_results.plot(x="Modèle", y="Train time (s)", kind="bar", legend=False, ax=ax_time, color="#16a34a")
        ax_time.set_ylabel("Temps d'entraînement (s)")
        ax_time.set_xlabel("Modèle")
        ax_time.set_title("Temps d'entraînement par modèle")
        st.pyplot(fig_time)
```
- Initialise le stockage des modèles entraînés dans la session.
- Si l'utilisateur clique sur "Entraîner et comparer" :
  - Boucle sur chaque modèle sélectionné, entraîne, prédit, mesure le temps, calcule les scores, stocke les résultats.
  - Affiche les résultats dans un tableau et sous forme de barplots.

#### b) Prédiction manuelle

```python
    st.subheader("Prédiction manuelle")
    st.markdown("<i>Testez une prédiction sur un échantillon personnalisé :</i>", unsafe_allow_html=True)
    input_features = list(X.columns)
    user_inputs = {}
    for feat in input_features:
        val = st.number_input(f"{feat}", value=float(X[feat].mean()))
        user_inputs[feat] = val
    model_pred_name = st.selectbox("Modèle pour la prédiction", models_selected)
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
- Affiche un formulaire dynamique pour saisir les features d'un échantillon.
- L'utilisateur choisit le modèle pour la prédiction.
- Si "Prédire" est cliqué, on effectue la prédiction et on affiche le label original.

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
- Affiche la heatmap de corrélation et les histogrammes des variables numériques.

---

## 8. Footer et aide

```python
st.sidebar.markdown("---")
st.sidebar.info("Contact : https://github.com/diamankayero/trainedml")
st.sidebar.markdown("""<div style='background:#fef9c3;...'>...</div>""", unsafe_allow_html=True)
```
- Affiche un encadré d'aide pour l'utilisateur (ex : comment récupérer un lien GitHub Raw).

---

**Chaque ligne et bloc du code principal est expliqué ci-dessus. Si tu veux le détail d'une ligne précise, indique-la et je t'expliquerai encore plus en profondeur !**
