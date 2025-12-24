from trainedml.models.knn import KNNModel
from trainedml.models.logistic import LogisticModel
from trainedml.models.random_forest import RandomForestModel

def get_model(name, **params):
    if name == "KNN":
        return KNNModel(**params)
    if name == "Logistic Regression":
        return LogisticModel(**params)
    if name == "Random Forest":
        return RandomForestModel(**params)
    raise ValueError(f"Modèle inconnu: {name}")
