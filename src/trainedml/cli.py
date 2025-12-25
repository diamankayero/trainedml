import argparse
from trainedml.data.loader import DataLoader
from trainedml.models.knn import KNNModel
from trainedml.models.logistic import LogisticModel
from trainedml.models.random_forest import RandomForestModel
from trainedml.evaluation import Evaluator
from trainedml.visualization import Visualizer
from sklearn.model_selection import train_test_split

MODEL_MAP = {
    'knn': KNNModel,
    'logistic': LogisticModel,
    'random_forest': RandomForestModel
}

def main():

    parser = argparse.ArgumentParser(description="trainedml: pipeline ML simple")
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys(), default='random_forest', help='Type de modèle à utiliser')
    parser.add_argument('--dataset', type=str, default='iris', help='Nom du dataset (iris, wine)')
    parser.add_argument('--url', type=str, default=None, help='URL d\'un CSV distant')
    parser.add_argument('--target', type=str, default=None, help='Nom de la colonne cible (si url)')
    parser.add_argument('--seed', type=int, default=42, help='Seed pour le split train/test')
    parser.add_argument('--test-size', type=float, default=0.3, help='Proportion de test (0-1)')
    parser.add_argument('--show', action='store_true', help='Afficher la heatmap après entraînement')
    parser.add_argument('--histogram', action='store_true', help='Afficher un histogramme des colonnes numériques')
    parser.add_argument('--benchmark', action='store_true', help='Comparer tous les modèles et afficher scores et temps')
    parser.add_argument('--line', nargs=2, metavar=('X', 'Y'), help='Tracer une courbe (line plot) entre deux colonnes')
    args = parser.parse_args()

    print(f"Chargement du dataset {args.dataset if args.url is None else args.url} ...")
    loader = DataLoader()
    X, y = loader.load_dataset(name=args.dataset if args.url is None else None, url=args.url, target=args.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    print(f"Taille X_train : {X_train.shape}, X_test : {X_test.shape} (seed={args.seed})")

    import pandas as pd
    if args.url is not None:
        data = pd.concat([X, y], axis=1)
    else:
        data = loader.load_csv_from_url("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv") if args.dataset == "iris" else pd.concat([X, y], axis=1)

    viz = Visualizer(data)
    numeric_cols = [col for col in data.columns if data[col].dtype != 'O']

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
        print(f"Entraînement du modèle {args.model}...")
        model = MODEL_MAP[args.model]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("Évaluation :")
        scores = Evaluator.evaluate_all(y_test, y_pred)
        for metric, value in scores.items():
            print(f"{metric}: {value:.3f}")

    if args.line:
        x_col, y_col = args.line
        print(f"Génération de la courbe {y_col} en fonction de {x_col}...")
        fig = viz.line(x_column=x_col, y_column=y_col)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Utilisez --show pour afficher la courbe.")
    elif args.histogram:
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

if __name__ == "__main__":
    main()
