"""
Permet d'importer facilement les modèles depuis le sous-package models.
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

# Registre centralisé des modèles de classification
CLASSIFIER_MAP: Dict[str, Callable[..., BaseModel]] = {
    'knn': KNNModel,
    'logistic': LogisticModel,
    'random_forest': RandomForestModel
}

# Registre centralisé des modèles de régression
REGRESSOR_MAP: Dict[str, Callable[..., BaseModel]] = {
    'knn_regressor': KNNRegressorModel,
    'linear': LinearRegressorModel,
    'ridge': RidgeRegressorModel,
    'lasso': LassoRegressorModel,
    'random_forest_regressor': RandomForestRegressorModel
}

# Registre complet (classification + régression)
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
    Factory pour obtenir une instance de modèle par son nom.
    
    Args:
        name (str): nom du modèle (ex: 'knn', 'random_forest', 'linear', 'ridge')
        **kwargs: hyperparamètres à passer au modèle
    
    Returns:
        BaseModel: instance du modèle
    
    Raises:
        ValueError: si le nom du modèle n'est pas reconnu
    """
    if name not in MODEL_MAP:
        raise ValueError(f"Modèle inconnu: {name}. Disponibles: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[name](**kwargs)


def get_classifier(name: str, **kwargs):
    """
    Factory pour obtenir une instance de classificateur par son nom.
    
    Args:
        name (str): nom du classificateur ('knn', 'logistic', 'random_forest')
        **kwargs: hyperparamètres à passer au modèle
    
    Returns:
        BaseModel: instance du classificateur
    """
    if name not in CLASSIFIER_MAP:
        raise ValueError(f"Classificateur inconnu: {name}. Disponibles: {list(CLASSIFIER_MAP.keys())}")
    return CLASSIFIER_MAP[name](**kwargs)


def get_regressor(name: str, **kwargs):
    """
    Factory pour obtenir une instance de régresseur par son nom.
    
    Args:
        name (str): nom du régresseur ('linear', 'ridge', 'lasso', 'knn_regressor', 'random_forest_regressor')
        **kwargs: hyperparamètres à passer au modèle
    
    Returns:
        BaseRegressor: instance du régresseur
    """
    if name not in REGRESSOR_MAP:
        raise ValueError(f"Régresseur inconnu: {name}. Disponibles: {list(REGRESSOR_MAP.keys())}")
    return REGRESSOR_MAP[name](**kwargs)
