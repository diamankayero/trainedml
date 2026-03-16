# trainedml

> Modular machine learning framework for Python — train, benchmark, and visualize ML models with a unified API, CLI, and web interface.

<p align="left">
  <a href="https://pypi.org/project/trainedml/"><img src="https://img.shields.io/pypi/v/trainedml" alt="PyPI version"></a>
  <a href="https://pypi.org/project/trainedml/"><img src="https://img.shields.io/pypi/pyversions/trainedml" alt="Python versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://diamankayero.github.io/trainedml/"><img src="https://img.shields.io/badge/Documentation-GitHub%20Pages-blue?logo=github" alt="Documentation"></a>
  <a href="https://trainedml.streamlit.app"><img src="https://img.shields.io/badge/Webapp-Streamlit-ff4b4b?logo=streamlit" alt="Webapp"></a>
  <img src="https://img.shields.io/badge/tests-passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/coverage-100%25-success" alt="Coverage">
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

---

## Overview

**trainedml** is a modular Python package for training, benchmarking, and visualizing machine learning models on classic or custom datasets. Inspired by the clarity of numpy, pandas, and matplotlib, it provides a unified API, a command-line interface, and an interactive Streamlit web app for end-to-end ML workflows.

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
from trainedml import Trainer

# Train on a built-in dataset
trainer = Trainer(dataset="iris", model="random_forest")
trainer.fit()

# Evaluate
scores = trainer.evaluate()
print(scores)

# Predict
predictions = trainer.predict([[5.1, 3.5, 1.4, 0.2]])
print(predictions)
```

---

## Features

- **Unified API** — train, evaluate, and predict with a single `Trainer` class
- **Built-in datasets** — Iris, Wine, and any remote CSV via URL
- **Modular models** — KNN, Logistic Regression, Random Forest, and more
- **Standard metrics** — accuracy, precision, recall, F1, R², MSE, RMSE, MAE
- **Benchmarking** — compare models side-by-side with timing and parallelization
- **Exploratory analysis** — correlations, distributions, missing values, outliers, multicollinearity
- **Visualization** — heatmaps, histograms, line plots, boxplots, bivariate charts
- **CLI** — automate ML pipelines from the terminal
- **Streamlit webapp** — interactive web interface for exploration and prediction
- **Full test coverage** — 100% coverage with Sphinx documentation

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

Train on a custom remote dataset:

```python
trainer = Trainer(
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    target="quality",
    model="logistic"
)
trainer.fit()
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

# Benchmark all models on Iris
python -m trainedml --benchmark --dataset iris

# Train KNN on Wine
python -m trainedml --model knn --dataset wine

# Visualize correlation heatmap
python -m trainedml --dataset iris --show
```

### Interactive Webapp

```bash
streamlit run trainedml_webapp/src/app.py
```

Or visit the hosted version: [trainedml.streamlit.app](https://trainedml.streamlit.app)

---

## Architecture

```
trainedml/
├── src/trainedml/
│   ├── __init__.py        # Trainer API
│   ├── analyzer.py        # Exploratory data analysis
│   ├── benchmark.py       # Model benchmarking
│   ├── cli.py             # CLI interface
│   ├── evaluation.py      # Evaluation metrics
│   ├── figure.py          # Figure generation
│   ├── visualization.py   # Visualization tools
│   ├── data/              # Data loading
│   ├── models/            # ML models (KNN, LR, RF, ...)
│   ├── utils/             # Utility functions
│   └── viz/               # Advanced visualizations
├── doc/                   # Sphinx documentation
├── tests/                 # Unit tests
├── trainedml_webapp/      # Streamlit webapp
└── requirements.txt
```

---

## Testing

```bash
pytest tests/
```

---

## Documentation

- [Online docs (GitHub Pages)](https://diamankayero.github.io/trainedml/)
- [Usage guide](DOC_UTILISATION.md)
- [Webapp docs](trainedml_webapp/doc/streamlit_app.md)

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

MIT — see [LICENSE](LICENSE) for details.
