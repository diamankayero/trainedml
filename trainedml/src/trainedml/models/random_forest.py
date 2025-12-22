"""
Implémentation du modèle Random Forest pour trainedml.
"""
from .base import BaseModel
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(BaseModel):
    """
    Modèle de classification Random Forest.
    """
    def __init__(self, n_estimators=100):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def fit(self, X, y):
        """Entraîne le modèle Random Forest."""
        self.model.fit(X, y)

    def predict(self, X):
        """Prédit la classe pour de nouvelles données."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Retourne la précision du modèle sur les données de test."""
        return self.model.score(X, y)
