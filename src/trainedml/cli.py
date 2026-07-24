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
Train a Random Forest model on Iris and show the heatmap:
    python -m trainedml --model random_forest --dataset iris --show

Compare every model on Wine with 5-fold cross-validation:
    python -m trainedml --dataset wine --benchmark --cv 5

Train then save a model:
    python -m trainedml --dataset iris --model knn --save model.joblib

Predict on a new CSV with a saved model:
    python -m trainedml --load model.joblib --input new_data.csv --output preds.csv

Load a remote CSV and plot a line chart:
    python -m trainedml --url https://.../data.csv --target class --line feature1 feature2 --show

Notes
-----
- Use --show to display the matplotlib figures at the end of the script.
- The CLI automatically detects the task type (classification/regression).
"""

import argparse

from trainedml.data.loader import DataLoader
from trainedml.models import MODEL_MAP, CLASSIFIER_MAP, REGRESSOR_MAP
from trainedml.tasks import is_classification_target as _is_classification_target
from trainedml.visualization import Visualizer


def _predict_mode(args):
    """Predict mode: load a saved model and predict on a CSV."""
    import pandas as pd
    from trainedml import Trainer

    if not args.input:
        raise SystemExit("--load requires --input <file.csv>")
    trainer = Trainer.load(args.load)
    X = pd.read_csv(args.input)
    preds = trainer.predict(X)
    out = pd.DataFrame({"prediction": preds})
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"{len(out)} predictions written to {args.output}")
    else:
        print(out.to_string(index=False))


def main():

    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="trainedml: simple ML pipeline")
    parser.add_argument('--model', type=str, choices=MODEL_MAP.keys(), default='random_forest', help='Model type to use')
    parser.add_argument('--dataset', type=str, default='iris', help='Dataset name (iris, wine, diabetes)')
    parser.add_argument('--url', type=str, default=None, help='URL of a remote CSV')
    parser.add_argument('--target', type=str, default=None, help='Target column name (if url)')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the train/test split')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test proportion (0-1)')
    parser.add_argument('--show', action='store_true', help='Show the heatmap after training')
    parser.add_argument('--histogram', action='store_true', help='Show a histogram of the numeric columns')
    parser.add_argument('--benchmark', action='store_true', help='Compare every model and show scores and timing')
    parser.add_argument('--cv', type=int, default=0, help='N-fold cross-validation for the benchmark (0 = simple split)')
    parser.add_argument('--line', nargs=2, metavar=('X', 'Y'), help='Plot a line chart between two columns')
    parser.add_argument('--save', type=str, default=None, help='Save the trained model (.joblib file)')
    parser.add_argument('--load', type=str, default=None, help='Load a saved model to predict (with --input)')
    parser.add_argument('--input', type=str, default=None, help='Input CSV for prediction (with --load)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV for predictions (with --load)')
    args = parser.parse_args()

    # --- Predict mode (no training) ---
    if args.load:
        _predict_mode(args)
        return

    # --- Data loading ---
    print(f"Loading dataset {args.dataset if args.url is None else args.url} ...")
    loader = DataLoader()
    X, y = loader.load_dataset(name=args.dataset if args.url is None else None, url=args.url, target=args.target)
    X_train, X_test, y_train, y_test = loader.split(X, y, test_size=args.test_size, random_state=args.seed)
    print(f"X_train shape: {X_train.shape}, X_test: {X_test.shape} (seed={args.seed})")

    # --- Task type detection ---
    is_classification = _is_classification_target(y)
    task_type = "classification" if is_classification else "regression"
    print(f"Detected task type: {task_type}")

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
        # Only use the models suited to the task type
        models_to_use = CLASSIFIER_MAP if is_classification else REGRESSOR_MAP
        print(f"Models compared: {list(models_to_use.keys())}")

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
        print(f"Training model {args.model}...")
        trainer = Trainer(
            dataset=args.dataset if args.url is None else None,
            url=args.url, target=args.target,
            model=args.model, test_size=args.test_size, seed=args.seed,
        )
        trainer.fit()

        print("Evaluation:")
        for metric, value in trainer.evaluate().items():
            print(f"{metric}: {value:.3f}")

        if args.save:
            trainer.save(args.save)
            print(f"Model saved to {args.save}")

    # --- Visualization options ---
    if args.line:
        x_col, y_col = args.line
        print(f"Generating line chart {y_col} vs {x_col}...")
        viz.line(x_column=x_col, y_column=y_col)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Use --show to display the line chart.")
    elif args.histogram:
        print("Generating histogram of numeric columns...")
        viz.histogram(columns=numeric_cols, legend=True)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Use --show to display the histogram.")
    else:
        print("Generating correlation heatmap...")
        viz.heatmap(features=numeric_cols)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            print("Use --show to display the heatmap.")

if __name__ == "__main__":
    main()
