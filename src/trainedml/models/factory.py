"""
Factory pour instancier les modèles ML de trainedml.
Permet de récupérer un modèle par son nom (KNN, Logistic Regression, Random Forest).
"""

from trainedml.models.knn import KNNModel
from trainedml.models.logistic import LogisticModel
from trainedml.models.random_forest import RandomForestModel

def get_model(model_name):
    """
    Retourne une instance du modèle ML correspondant au nom donné.
    Args:
        model_name (str): Nom du modèle ('KNN', 'Logistic Regression', 'Random Forest')
    Returns:
        Instance du modèle ML
    """
    name = model_name.lower()
    if name in ['knn', 'k-nn', 'k nearest neighbors', 'k plus proches voisins']:
        return KNNModel()
    elif name in ['logistic', 'logistic regression', 'régression logistique']:
        return LogisticModel()
    elif name in ['random forest', 'forêt aléatoire', 'rf']:
        return RandomForestModel()
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
