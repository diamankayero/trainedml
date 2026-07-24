<p align="center">
  <a href="https://diamankayero.github.io/trainedml/"><img src="https://raw.githubusercontent.com/diamankayero/trainedml/main/doc/source/_static/banner.svg" alt="trainedml" height="72"></a>
</p>

> Modular machine learning framework for Python - train, benchmark, and visualize ML models with a unified API, CLI, and web interface.

<p align="left">
  <a href="https://pypi.org/project/trainedml/"><img src="https://img.shields.io/pypi/v/trainedml" alt="PyPI version"></a>
  <a href="https://pypi.org/project/trainedml/"><img src="https://img.shields.io/pypi/pyversions/trainedml" alt="Python versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://diamankayero.github.io/trainedml/"><img src="https://img.shields.io/badge/Documentation-GitHub%20Pages-blue?logo=github" alt="Documentation"></a>
  <a href="https://trainedml.onrender.com"><img src="https://img.shields.io/badge/Demo-trainedml.onrender.com-46b3e6?logo=render&logoColor=white" alt="Live demo"></a>
  <a href="https://github.com/diamankayero/trainedml/actions/workflows/workflow.yml"><img src="https://github.com/diamankayero/trainedml/actions/workflows/workflow.yml/badge.svg" alt="CI"></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Features](#features)
- [API Reference](#api-reference)
- [CLI Usage](#cli-usage)
- [Architecture](#architecture)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

<p align="center">
   <img src="https://raw.githubusercontent.com/diamankayero/trainedml/main/public/matric_corre.png" alt="Matrice de corrélation" width="260"/>
   <img src="https://raw.githubusercontent.com/diamankayero/trainedml/main/public/histogram.png" alt="Histogramme" width="260"/>
   <img src="https://raw.githubusercontent.com/diamankayero/trainedml/main/public/line.png" alt="Courbe" width="260"/>
   <img src="https://raw.githubusercontent.com/diamankayero/trainedml/main/public/comparaison_de_model.jpeg" alt="Comparaison de modèles" width="260"/>
</p>

---

## Overview

**trainedml** is a modular Python package to train, compare, and visualize machine learning models on classic or custom datasets: from a CSV to a model comparison and an exploratory report, in one line. It offers a unified API (`Trainer`, `compare()`), a command-line interface, and a FastAPI web API with a demo page.

### Where trainedml fits

trainedml sits between raw scikit-learn (full control, more boilerplate) and
full AutoML frameworks (more automated, more opinionated, heavier). It
targets teaching, prototyping, and small-to-medium projects where you want a
one-line comparison and a real EDA report without giving up direct access to
the underlying scikit-learn estimators.

| | trainedml | scikit-learn | PyCaret | lazypredict |
|---|---|---|---|---|
| Model comparison in one line | ✅ `compare()` | ❌ manual loop | ✅ `compare_models()` | ✅ `LazyClassifier` |
| Cross-validated comparison with std dev | ✅ | ❌ manual | ➖ (available per-model, not always shown in the comparison grid) | ❌ (single split) |
| Self-contained HTML EDA report | ✅ `report()` | ❌ | ➖ (`setup(profile=True)`, via `ydata-profiling`) | ❌ |
| Confusion matrix / ROC / feature importance | ✅ built in | ➖ (assemble yourself) | ✅ | ❌ |
| Hyperparameter search (grid/random) | ✅ `grid_search()`/`random_search()` | ✅ (`GridSearchCV` directly) | ✅ (`tune_model()`) | ❌ |
| Accepts any scikit-learn estimator directly | ✅ `Trainer(model=SVC())` | ✅ (it's the source) | ✅ `create_model(SVC())` | ❌ |
| CLI included | ✅ | ❌ | ❌ | ❌ |
| Dependency footprint | Light (scikit-learn, pandas, matplotlib, seaborn, statsmodels) | Lightest (itself) | Heavy (many optional backends) | Moderate (adds xgboost, lightgbm, pmdarima) |
| Typical use case | Teaching, prototyping, small/medium projects | Full control, production pipelines | Fast end-to-end AutoML | Quick first-pass model ranking |

*Comparison as of mid-2026, based on each project's public documentation;
verify against their latest release before relying on it.*

trainedml does not try to replace scikit-learn (it wraps and re-exposes it
directly) or to out-automate PyCaret (no automatic feature engineering, no
stacking/blending, no deployment tooling). The trade-off is intentional: a
smaller surface, in plain scikit-learn terms, that stays easy to read,
extend, and teach from.

---

## Installation

```bash
pip install trainedml
```

Or install from source:

```bash
git clone https://github.com/diamankayero/trainedml.git
cd trainedml
pip install -e .
```

**Requirements:** Python 3.9+

---

## Quickstart

```python
from trainedml import Trainer, compare

# Train on a built-in dataset (loaded locally, no network needed)
trainer = Trainer(dataset="iris", model="random_forest")
trainer.fit()

# Evaluate - metrics match the task (classification or regression)
print(trainer.evaluate())

# Predict
print(trainer.predict([[5.1, 3.5, 1.4, 0.2]]))

# Compare every suitable model with 5-fold cross-validation, in one line
print(compare(dataset="wine", cv=5))
```

---

## Features

- **Unified API** - train, evaluate, predict, save/load with a single `Trainer` class
- **One-line model comparison** - `compare(dataset="wine", cv=5)` returns a sorted DataFrame with cross-validated scores and timings
- **Automatic preprocessing** - imputation, scaling, one-hot encoding, refit per CV fold (no data leakage)
- **Any scikit-learn estimator** - `Trainer(model=SVC())` works out of the box, plus built-in KNN, Logistic Regression, Random Forest, Linear/Ridge/Lasso
- **Built-in datasets offline** - Iris and Wine load locally via scikit-learn; any remote CSV via URL (cached)
- **Task auto-detection** - classification vs regression, with matching metrics (accuracy/precision/recall/F1 or R²/MSE/RMSE/MAE)
- **Benchmarking** - single split or K-fold cross-validation, with timing and parallelization
- **Model persistence** - `trainer.save("model.joblib")` / `Trainer.load(...)`, batch prediction from the CLI
- **EDA report** - one self-contained HTML report: correlations, distributions, missing values, outliers, normality, VIF
- **Visualization** - heatmaps, histograms, line plots, boxplots, bivariate charts
- **CLI** - automate ML pipelines from the terminal
- **Web demo** - [trainedml-webapp](https://github.com/diamankayero/trainedml-webapp): FastAPI API + HTML/JS front consuming this package, live at [trainedml.onrender.com](https://trainedml.onrender.com)

---

## API Reference

### `Trainer`

The main entry point for the framework.

```python
from trainedml import Trainer

trainer = Trainer(dataset="iris", model="knn")
trainer.fit()
scores = trainer.evaluate()
predictions = trainer.predict([[5.1, 3.5, 1.4, 0.2]])
```

The train/test split is handled internally by the package (`DataLoader.split`) - no need to call scikit-learn yourself. The splits are available as attributes, and you can re-split with a new seed without recreating the trainer:

```python
trainer.X_train, trainer.X_test, trainer.y_train, trainer.y_test  # after fit()

# Vary the seed to check model stability across splits
for s in range(5):
    print(trainer.fit(seed=s).evaluate())
```

Train on a custom remote dataset, in-memory data, hyperparameters, or any scikit-learn estimator:

```python
# Remote CSV
trainer = Trainer(
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    target="quality",
    model="logistic"
)

# In-memory data (X, y)
trainer = Trainer(X=my_dataframe, y=my_target, model="ridge")

# Hyperparameters
trainer = Trainer(dataset="wine", model="knn", model_params={"n_neighbors": 7})

# Any scikit-learn estimator
from sklearn.svm import SVC
trainer = Trainer(dataset="iris", model=SVC(kernel="rbf"))
```

Save and reload a trained model:

```python
trainer.fit().save("model.joblib")
restored = Trainer.load("model.joblib")
restored.predict([[5.1, 3.5, 1.4, 0.2]])
```

### `compare`

Compare every suitable model (auto-detected task) with cross-validation, in one line:

```python
from trainedml import compare

df = compare(dataset="wine", cv=5)      # built-in dataset
df = compare(X=X, y=y, cv=5)            # your own data
print(df)  # sorted DataFrame: metrics ± std, fit/predict times
```

### `DataLoader`

```python
from trainedml.data.loader import DataLoader

X, y = DataLoader().load_dataset(name="wine")
```

### `KNNModel` (and other models)

```python
from trainedml.models.knn import KNNModel

model = KNNModel(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### `Benchmark`

```python
from trainedml.benchmark import Benchmark
from trainedml.models.knn import KNNModel
from trainedml.models.random_forest import RandomForestModel

models = {"knn": KNNModel(), "rf": RandomForestModel()}
bench = Benchmark(models)
results = bench.run(X_train, y_train, X_test, y_test)
print(results)
```

### `Visualizer`

```python
from trainedml.visualization import Visualizer

viz = Visualizer(X)
fig = viz.heatmap()
fig.show()
```

---

## CLI Usage

```bash
# Show help
python -m trainedml --help

# Benchmark all models on Iris (5-fold cross-validation)
python -m trainedml --benchmark --dataset iris --cv 5

# Train KNN on Wine and save the model
python -m trainedml --model knn --dataset wine --save model.joblib

# Predict on a new CSV with a saved model
python -m trainedml --load model.joblib --input new_data.csv --output predictions.csv

# Visualize correlation heatmap
python -m trainedml --dataset iris --show
```

### Web demo

**Live demo: [trainedml.onrender.com](https://trainedml.onrender.com)** (free tier: first visit after idle takes ~30 s to wake up).

The web demo (FastAPI API + HTML/JS front) lives in its own repository,
[trainedml-webapp](https://github.com/diamankayero/trainedml-webapp), which
consumes this package from PyPI like any user would. A full Next.js product
version of the same idea, [ModeLmL](https://github.com/diamankayero/ModeLmL),
is live at [modelml.vercel.app](https://modelml.vercel.app).

---

## Architecture

```
trainedml/
├── src/trainedml/
│   ├── __init__.py        # Trainer API (train/evaluate/predict/save/load)
│   ├── compare.py         # One-line model comparison (cross-validated DataFrame)
│   ├── preprocessing.py   # Automatic preprocessing (impute, scale, one-hot)
│   ├── tasks.py           # Task detection (classification vs regression)
│   ├── benchmark.py       # Model benchmarking (single split or K-fold CV)
│   ├── evaluation.py      # Evaluation metrics
│   ├── report.py          # Self-contained HTML EDA report
│   ├── analyzer.py        # Exploratory data analysis
│   ├── cli.py             # CLI interface
│   ├── visualization.py   # Visualization facade (Visualizer)
│   ├── data/              # Data loading (offline built-ins + remote CSV)
│   ├── models/            # ML models (KNN, LR, RF, regressors...)
│   └── viz/               # Advanced visualizations
├── doc/                   # Sphinx documentation
├── examples/              # Example gallery source (also runnable as plain scripts)
├── tests/                 # Unit tests
└── CHANGELOG.md

Web demo: separate repo, github.com/diamankayero/trainedml-webapp
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

CI runs the suite on Python 3.9 to 3.13 (Ubuntu) and Windows, plus ruff and mypy.

---

## Documentation

- [Online docs (GitHub Pages)](https://diamankayero.github.io/trainedml/)
- [Example gallery](https://diamankayero.github.io/trainedml/auto_examples/01_bases/index.html): eleven runnable, narrated examples from first model to production, with real output and plots
- [Usage guide and project journal](DOC_UTILISATION.md)
- [Changelog](CHANGELOG.md)
- [Example gallery source](examples/README.md)
- Every folder has its own README (architecture, conventions, how to contribute)

---

## Contributing

Contributions are welcome!

1. Fork the project
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

For bugs or suggestions, open an [issue](https://github.com/diamankayero/trainedml/issues).

---

## License

MIT - see [LICENSE](LICENCE) for details.
