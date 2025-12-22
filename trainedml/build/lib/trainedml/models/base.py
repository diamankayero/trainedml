"""
Classe de base abstraite pour tous les modèles de machine learning du projet trainedml.
Définit l'interface commune à tous les modèles supervisés.
"""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Classe abstraite pour les modèles de classification.
    Toutes les classes de modèles doivent hériter de cette classe et implémenter ses méthodes.
    """
    def __init__(self):
        self.model = None  # L'objet du modèle sous-jacent (scikit-learn, etc.)

    @abstractmethod
    def fit(self, X, y):
        """Entraîne le modèle sur les données X et la cible y."""
        pass

    @abstractmethod
    def predict(self, X):
        """Prédit la cible pour de nouvelles données X."""
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Évalue le modèle sur des données de test et retourne une métrique de performance."""
        pass
