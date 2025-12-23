# -*- coding: utf-8 -*-
"""
App Streamlit principale pour la comparaison de modèles ML avec trainedml.
Permet de charger un dataset (exemple, upload, ou URL), de choisir un modèle, d'entraîner, d'évaluer et de visualiser les résultats.
"""

import streamlit as st  # Framework web interactif pour data science
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
import sys
## Import direct du package trainedml (après installation en mode editable)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
# ...existing code...
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fa;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #e0e7ff 0%, #f0fdfa 100%);
    }
    .block-container {
        padding-top: 2rem;
    }
    .score-box {
        background: #e0e7ff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #c7d2fe;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Logo fictif ---
# Affiche un logo en haut de page pour donner une identité visuelle à l'app
st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)

# --- Titre et sous-titre ---
# Présente l'objectif de l'application
st.title("Démo trainedml : Comparaison de modèles ML")
st.markdown(
    """
    <span style='font-size:1.2rem;'>Comparez facilement plusieurs modèles de machine learning sur des jeux de données classiques.</span>
    """,
    unsafe_allow_html=True
)

# --- Sidebar : configuration et aide ---
# Zone de configuration à gauche (choix dataset, modèle, params, upload...)
st.sidebar.header("Configuration")
st.sidebar.markdown(
    """
    <div style='background:#e0e7ff;padding:0.7em 1em;border-radius:0.5em;margin-bottom:1em;'>
    <b>trainedml webapp</b><br>
    <span style='font-size:0.95em;'>Démonstrateur interactif ML<br>by <a href='https://github.com/' target='_blank'>VotreNom</a></span>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sélection du dataset de démonstration ---
# Propose des datasets intégrés (iris, wine)
dataset_name = st.sidebar.selectbox("Choisir un dataset", ("iris", "wine"))

# --- Upload ou chargement par URL ---
# Permet à l'utilisateur de charger un CSV local ou distant (GitHub, UCI, etc.)
st.sidebar.markdown("<b>Ou charger un CSV par URL :</b>", unsafe_allow_html=True)
user_url = st.sidebar.text_input("URL d'un CSV (optionnel)")
sep = st.sidebar.selectbox("Séparateur CSV", [",", ";", "\t"], format_func=lambda x: {',': 'Virgule (,)', ';': 'Point-virgule (;)', '\t': 'Tabulation (\\t)'}[x])
user_file = st.sidebar.file_uploader("Uploader un CSV (optionnel)", type=["csv"])
df_user = None
if user_url:
    try:
        # Lecture du CSV distant avec le séparateur choisi
        df_user = pd.read_csv(user_url, sep=sep)
        st.sidebar.success(f"Données chargées depuis l'URL !")
    except Exception as e:
        st.sidebar.error(f"Erreur de chargement : {e}\nVérifiez que l'URL pointe bien vers un fichier CSV brut (ex : bouton 'Raw' sur GitHub).")
elif user_file is not None:
    # Lecture du CSV uploadé
    df_user = pd.read_csv(user_file, sep=sep)
    st.sidebar.success(f"Fichier chargé : {user_file.name}")
if df_user is not None:
    # On suppose que la dernière colonne est la cible (y)
    X = df_user.iloc[:, :-1]  # Toutes les colonnes sauf la dernière = features
    y = df_user.iloc[:, -1]   # Dernière colonne = cible
    dataset_name = f"Upload: {user_file.name if user_file else user_url}"
else:
    # Chargement d'un dataset intégré (iris, wine)
    loader = DataLoader()
    X, y = loader.load_dataset(name=dataset_name)

# --- Encodage automatique des labels cibles (y) si besoin ---
# Si la cible est catégorielle ou float (mais discrète), on encode pour la classification
if y.dtype == object or y.dtype.name == "category" or (hasattr(y, 'dtype') and str(y.dtype).startswith('float')):
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Pour affichage lisible des classes (ex : ['setosa', 'versicolor', ...])
    class_names = [str(c) for c in le.classes_]
else:
    class_names = [str(c) for c in np.unique(y)]

# --- Sélection du modèle ML ---
# Propose 3 modèles classiques de classification
model_name = st.sidebar.selectbox("Modèle", ("KNN", "Régression Logistique", "Random Forest"))
MODEL_MAP = {
    "KNN": KNNModel,  # k plus proches voisins
    "Régression Logistique": LogisticModel,  # régression logistique
    "Random Forest": RandomForestModel  # forêt aléatoire
}
model = MODEL_MAP[model_name]()  # Instancie le modèle choisi

# --- Paramètres de split train/test ---
seed = st.sidebar.number_input("Seed", value=42)  # Seed pour reproductibilité
# Taille du jeu de test (slider)
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)
# Split des données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# --- Layout principal : deux colonnes ---
col1, col2 = st.columns([1, 2])

# --- Colonne 1 : infos dataset ---
with col1:
    st.markdown("<b>Infos dataset :</b>", unsafe_allow_html=True)
    st.write(f"Nombre d'échantillons : {X.shape[0]}")
    st.write(f"Nombre de variables : {X.shape[1]}")
    st.write(f"Classes : {', '.join(class_names)}")
    st.dataframe(X.head())  # Affiche les premières lignes du dataset

# --- Colonne 2 : entraînement, scores, visualisations ---
with col2:
    if st.sidebar.button("Entraîner et évaluer"):
        with st.spinner("Entraînement du modèle..."):
            start = time.time()
            model.fit(X_train, y_train)  # Entraînement du modèle
            fit_time = time.time() - start
            y_pred = model.predict(X_test)  # Prédiction sur le test set
            predict_time = time.time() - start - fit_time
            scores = Evaluator.evaluate_all(y_test, y_pred)  # Calcul des scores
        st.success("Modèle entraîné !")
        st.markdown(f"<div class='score-box'><b>Scores :</b> {scores}</div>", unsafe_allow_html=True)
        st.write(f"Temps d'entraînement : {fit_time:.3f} s")
        st.write(f"Temps de prédiction : {predict_time:.3f} s")

        # --- Visualisation : heatmap de corrélation ---
        # Affiche la matrice de corrélation des variables/features
        viz = Visualizer(pd.concat([X, pd.Series(y, name='target')], axis=1))
        st.subheader("Heatmap de corrélation")
        fig = viz.heatmap(features=list(X.columns))
        if hasattr(fig, 'figure'):
            st.pyplot(fig.figure)
        else:
            st.pyplot(fig)

        # --- Visualisation : histogramme des variables numériques ---
        st.subheader("Histogramme des variables numériques")
        fig2 = viz.histogram(columns=list(X.columns), legend=True)
        if hasattr(fig2, 'figure'):
            st.pyplot(fig2.figure)
        else:
            st.pyplot(fig2)

# --- Footer sidebar : contact et aide GitHub Raw ---
st.sidebar.markdown("---")
st.sidebar.info("Contact : https://github.com/diamankayero/trainedml")

# Encadré d'aide pour bien utiliser les liens GitHub Raw
st.sidebar.markdown("""
<div style='background:#fef9c3;padding:0.7em 1em;border-radius:0.5em;margin-bottom:1em;font-size:0.95em;'>
<b>Astuce :</b> Pour charger un CSV depuis GitHub, <br>
1. Copiez l'URL de la page du fichier (exemple visible dans le navigateur) :<br>
<code>https://github.com/diamankayero/projets/blob/main/trees.csv</code><br>
2. Cliquez sur le bouton <b>Raw</b> du fichier sur GitHub, puis copiez l'URL affichée dans la barre d'adresse (c'est ce lien qu'il faut coller ici) :<br>
<code>https://raw.githubusercontent.com/diamankayero/projets/main/trees.csv</code>
</div>
""", unsafe_allow_html=True)
