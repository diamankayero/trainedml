"""
Command-line interface (CLI) for trainedml.

This script provides a simple and flexible CLI for running machine learning pipelines
with trainedml: data loading, model training, evaluation, benchmarking, visualization,
model persistence and batch prediction.

Features
--------
- Load built-in datasets (offline) or remote CSVs
- Train/test split with configurable seed and test size
- Automatic task type detection (classification vs regression)
- Model selection (KNN, Logistic Regression, Random Forest, regressors...)
- Benchmarking of all models for the task, with optional cross-validation (--cv)
- Visualization: heatmap, histogram, line plot
- Save a trained model (--save) and predict later on a CSV (--load/--input)

Examples (to run in terminal)
----------------------------
Entraîner un modèle Random Forest sur Iris et afficher la heatmap :
    python -m trainedml --model random_forest --dataset iris --show

Comparer tous les modèles sur Wine par validation croisée 5 plis :
    python -m trainedml --dataset wine --benchmark --cv 5

Entraîner puis sauvegarder un modèle :
    python -m trainedml --dataset iris --model knn --save model.joblib

Prédire sur un nouveau CSV avec un modèle sauvegardé :
    python -m trainedml --load model.joblib --input nouvelles_donnees.csv --output preds.csv

Charger un CSV distant et tracer une courbe :
    python -m trainedml --url https://.../data.csv --target classe --line feature1 feature2 --show

Notes
-----
- Utilisez --show pour afficher les figures matplotlib à la fin du script.
- Le CLI détecte automatiquement le type de tâche (classification/régression).
"""

import argparse

from trainedml.data.loader import DataLoader
from trainedml.models import MODEL_MAP, CLASSIFIER_MAP, REGRESSOR_MAP
from trainedml.tasks import is_classification_target as _is_classification_target
from trainedml.visualization import Visualizer


def _predict_mode(args):
    """Mode prédiction : charge un modèle sauvegardé et prédit sur un CSV."""
    import pandas as pd
    from trainedml import Trainer

    if not args.input:
        raise SystemExit("--load nécessite --input <fichier.csv>")
    trainer = Trainer.load(args.load)
    X = pd.read_csv(args.input)
    preds = trainer.predict(X)
    out = pd.DataFrame({"prediction": preds})
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"{len(out)} prédictions écrites dans {args.output}")
    else:
        print(out.to_string(index=False))


def main():

    # --- Argument parsing ---
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
    parser.add_argument('--cv', type=int, default=0, help='Validation croisée à N plis pour le benchmark (0 = simple split)')
    parser.add_argument('--line', nargs=2, metavar=('X', 'Y'), help='Tracer une courbe (line plot) entre deux colonnes')
    parser.add_argument('--save', type=str, default=None, help='Sauvegarder le modèle entraîné (fichier .joblib)')
    parser.add_argument('--load', type=str, default=None, help='Charger un modèle sauvegardé pour prédire (avec --input)')
    parser.add_argument('--input', type=str, default=None, help='CSV d\'entrée pour la prédiction (avec --load)')
    parser.add_argument('--output', type=str, default=None, help='CSV de sortie des prédictions (avec --load)')
    args = parser.parse_args()

    # --- Predict mode (no training) ---
    if args.load:
        _predict_mode(args)
        return

    # --- Data loading ---
    print(f"Chargement du dataset {args.dataset if args.url is None else args.url} ...")
    loader = DataLoader()
    X, y = loader.load_dataset(name=args.dataset if args.url is None else None, url=args.url, target=args.target)
    X_train, X_test, y_train, y_test = loader.split(X, y, test_size=args.test_size, random_state=args.seed)
    print(f"Taille X_train : {X_train.shape}, X_test : {X_test.shape} (seed={args.seed})")

    # --- Task type detection ---
    is_classification = _is_classification_target(y)
    task_type = "classification" if is_classification else "régression"
    print(f"Type de tâche détecté : {task_type}")

    # --- DataFrame for visualization ---
    import pandas as pd
    data = pd.concat([X, y], axis=1)

    viz = Visualizer(data)
    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

    # --- Benchmark mode ---
    if args.benchmark:
        print("\n--- BENCHMARK ---")
        from trainedml.benchmark import Benchmark
        from trainedml.preprocessing import PreprocessedModel
        # Utiliser uniquement les modèles adaptés au type de tâche
        models_to_use = CLASSIFIER_MAP if is_classification else REGRESSOR_MAP
        print(f"Modèles comparés : {list(models_to_use.keys())}")

        models = {name: PreprocessedModel(cls()) for name, cls in models_to_use.items()}
        bench = Benchmark(models)
        if args.cv and args.cv > 1:
            bench.run_cv(X, y, cv=args.cv, random_state=args.seed)
        else:
            bench.run(X_train, y_train, X_test, y_test)
        print(bench.to_dataframe().to_string())

    # --- Single model mode ---
    else:
        from trainedml import Trainer
        print(f"Entraînement du modèle {args.model}...")
        trainer = Trainer(
            dataset=args.dataset if args.url is None else None,
            url=args.url, target=args.target,
            model=args.model, test_size=args.test_size, seed=args.seed,
        )
        trainer.fit()

        print("Évaluation :")
        for metric, value in trainer.evaluate().items():
            print(f"{metric}: {value:.3f}")

        if args.save:
            trainer.save(args.save)
            print(f"Modèle sauvegardé dans {args.save}")

    # --- Visualization options ---
    if args.line:
        x_col, y_col = args.line
        print(f"Génération de la courbe {y_col} en fonction de {x_col}...")
        viz.line(x_column=x_col, y_column=y_col)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Utilisez --show pour afficher la courbe.")
    elif args.histogram:
        print("Génération de l'histogramme des colonnes numériques...")
        viz.histogram(columns=numeric_cols, legend=True)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Utilisez --show pour afficher l'histogramme.")
    else:
        print("Génération de la heatmap de corrélation...")
        viz.heatmap(features=numeric_cols)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Utilisez --show pour afficher la heatmap.")

if __name__ == "__main__":
    main()
