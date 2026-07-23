"""
Makes it easy to import models from the models sub-package.
"""
from typing import Callable, Dict

from .base import BaseModel, BaseRegressor
from .knn import KNNModel
from .logistic import LogisticModel
from .random_forest import RandomForestModel
from .regressors import (
    RandomForestRegressorModel,
    KNNRegressorModel,
    LinearRegressorModel,
    RidgeRegressorModel,
    LassoRegressorModel
)

# Central registry of classification models
CLASSIFIER_MAP: Dict[str, Callable[..., BaseModel]] = {
    'knn': KNNModel,
    'logistic': LogisticModel,
    'random_forest': RandomForestModel
}

# Central registry of regression models
REGRESSOR_MAP: Dict[str, Callable[..., BaseModel]] = {
    'knn_regressor': KNNRegressorModel,
    'linear': LinearRegressorModel,
    'ridge': RidgeRegressorModel,
    'lasso': LassoRegressorModel,
    'random_forest_regressor': RandomForestRegressorModel
}

# Full registry (classification + regression)
MODEL_MAP: Dict[str, Callable[..., BaseModel]] = {**CLASSIFIER_MAP, **REGRESSOR_MAP}

__all__ = [
    "BaseModel", "BaseRegressor",
    "KNNModel", "LogisticModel", "RandomForestModel",
    "RandomForestRegressorModel", "KNNRegressorModel",
    "LinearRegressorModel", "RidgeRegressorModel", "LassoRegressorModel",
    "CLASSIFIER_MAP", "REGRESSOR_MAP", "MODEL_MAP",
    "get_model", "get_classifier", "get_regressor",
]


def get_model(name: str, **kwargs):
    """
    Factory to get a model instance by name.

    Args:
        name (str): model name (e.g. 'knn', 'random_forest', 'linear', 'ridge')
        **kwargs: hyperparameters to pass to the model

    Returns:
        BaseModel: model instance

    Raises:
        ValueError: if the model name is not recognized
    """
    if name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[name](**kwargs)


def get_classifier(name: str, **kwargs):
    """
    Factory to get a classifier instance by name.

    Args:
        name (str): classifier name ('knn', 'logistic', 'random_forest')
        **kwargs: hyperparameters to pass to the model

    Returns:
        BaseModel: classifier instance
    """
    if name not in CLASSIFIER_MAP:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(CLASSIFIER_MAP.keys())}")
    return CLASSIFIER_MAP[name](**kwargs)


def get_regressor(name: str, **kwargs):
    """
    Factory to get a regressor instance by name.

    Args:
        name (str): regressor name ('linear', 'ridge', 'lasso', 'knn_regressor', 'random_forest_regressor')
        **kwargs: hyperparameters to pass to the model

    Returns:
        BaseRegressor: regressor instance
    """
    if name not in REGRESSOR_MAP:
        raise ValueError(f"Unknown regressor: {name}. Available: {list(REGRESSOR_MAP.keys())}")
    return REGRESSOR_MAP[name](**kwargs)
