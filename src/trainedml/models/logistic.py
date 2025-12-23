"""
Implémentation du modèle de régression logistique pour trainedml.
"""
from .base import BaseModel
from sklearn.linear_model import LogisticRegression

class LogisticModel(BaseModel):
    """
    Modèle de classification par régression logistique.
    """
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(max_iter=200)

    def fit(self, X, y):
        """Entraîne le modèle de régression logistique."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la classe pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne la précision du modèle sur les données de test."""
        return self.model.score(X, y)
