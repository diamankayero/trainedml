"""
Module d'évaluation des modèles de classification pour trainedml.
Calcule les métriques standards de classification.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    """
    Classe utilitaire pour évaluer les performances d'un modèle de classification.
    """
    @staticmethod
    def evaluate_all(y_true, y_pred):
        """
        Calcule plusieurs métriques de classification.
        Args:
            y_true: vraies classes
            y_pred: classes prédites
        Returns:
            dict: dictionnaire des métriques
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
