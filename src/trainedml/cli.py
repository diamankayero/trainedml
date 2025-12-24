
"""
Script CLI pour exécuter des pipelines de machine learning avec trainedml.
Permet de charger des jeux de données, d'entraîner/évaluer des modèles, de comparer plusieurs modèles,
et de visualiser les résultats (heatmap, histogramme).
"""

import argparse  # Pour parser les arguments de la ligne de commande
from trainedml.data.loader import DataLoader  # Chargement des données
from trainedml.models.knn import KNNModel  # Modèle KNN
from trainedml.models.logistic import LogisticModel  # Modèle régression logistique
from trainedml.models.random_forest import RandomForestModel  # Modèle forêt aléatoire
from trainedml.evaluation import Evaluator  # Outils d'évaluation
from trainedml.visualization import Visualizer  # Outils de visualisation
from sklearn.model_selection import train_test_split  # Split train/test
import pandas as pd  # Manipulation de données


# Dictionnaire pour faire le lien entre le nom du modèle et sa classe
MODEL_MAP = {
    'knn': KNNModel,
    'logistic': LogisticModel,
    'random_forest': RandomForestModel
}



    """
    Fonction principale du script CLI.
    Parse les arguments, charge les données, entraîne/évalue les modèles,
    affiche les résultats et les visualisations selon les options choisies.
    """

    # Définition des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="trainedml: pipeline ML simple")
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys(), default='random_forest', help='Type de modèle à utiliser')
    parser.add_argument('--dataset', type=str, default='iris', help='Nom du dataset (iris, wine)')
    parser.add_argument('--url', type=str, default=None, help="URL d'un CSV distant")
    parser.add_argument('--target', type=str, default=None, help='Nom de la colonne cible (si url)')
    parser.add_argument('--seed', type=int, default=42, help='Seed pour le split train/test')
    parser.add_argument('--test-size', type=float, default=0.3, help='Proportion de test (0-1)')
    parser.add_argument('--show', action='store_true', help='Afficher la heatmap ou l\'histogramme après entraînement')
    parser.add_argument('--histogram', action='store_true', help='Afficher un histogramme des colonnes numériques')
    parser.add_argument('--benchmark', action='store_true', help='Comparer tous les modèles et afficher scores et temps')
    args = parser.parse_args()

    # Chargement des données
    print(f"Chargement du dataset {args.dataset if args.url is None else args.url} ...")
    loader = DataLoader()
    X, y = loader.load_dataset(name=args.dataset if args.url is None else None, url=args.url, target=args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    print(f"Taille X_train : {X_train.shape}, X_test : {X_test.shape} (seed={args.seed})")

    # Construction du DataFrame complet pour la visualisation
    if args.url is not None:
        data = pd.concat([X, y], axis=1)
    else:
        # Pour iris, on recharge le CSV complet pour avoir les noms de colonnes d'origine
        data = loader.load_csv_from_url("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv") if args.dataset == "iris" else pd.concat([X, y], axis=1)

    viz = Visualizer(data)  # Outil de visualisation
    numeric_cols = [col for col in data.columns if data[col].dtype != 'O']  # Colonnes numériques

    # Mode benchmark : compare tous les modèles
    if args.benchmark:
        print("\n--- BENCHMARK ---")
        from trainedml.benchmark import Benchmark
        models = {name: cls() for name, cls in MODEL_MAP.items()}
        bench = Benchmark(models)
        results = bench.run(X_train, y_train, X_test, y_test)
        for name, res in results.items():
            print(f"\nModèle : {name}")
            for metric, value in res['scores'].items():
                print(f"  {metric}: {value:.3f}")
            print(f"  fit_time: {res['fit_time']:.4f} s")
            print(f"  predict_time: {res['predict_time']:.4f} s")
    else:
        # Entraînement et évaluation d'un seul modèle
        print(f"Entraînement du modèle {args.model}...")
        model = MODEL_MAP[args.model]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Évaluation :")
        scores = Evaluator.evaluate_all(y_test, y_pred)
        for metric, value in scores.items():
            print(f"{metric}: {value:.3f}")

    # Visualisation : histogramme ou heatmap
    if args.histogram:
        print("Génération de l'histogramme des colonnes numériques...")
        fig = viz.histogram(columns=numeric_cols, legend=True)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Utilisez --show pour afficher l'histogramme.")
    else:
        print("Génération de la heatmap de corrélation...")
        fig = viz.heatmap(features=numeric_cols)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Utilisez --show pour afficher la heatmap.")

# Point d'entrée du script
if __name__ == "__main__":
    main()
