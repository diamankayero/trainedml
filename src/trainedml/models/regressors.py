"""
Regression model implementations for trainedml.
Contains the regressors: RandomForestRegressor, KNNRegressor, LinearRegressor.
"""
from .base import BaseRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso


class RandomForestRegressorModel(BaseRegressor):
    """
    Random Forest regression model.

    Args:
        n_estimators (int): Number of trees in the forest (default: 100)
        **kwargs: Other hyperparameters passed to RandomForestRegressor
    """
    def __init__(self, n_estimators=100, **kwargs):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)

    def fit(self, X, y):
        """Fit the Random Forest model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict the target value for new data."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Return the model's R² score on the test data."""
        return self.model.score(X, y)


class KNNRegressorModel(BaseRegressor):
    """
    K-Nearest Neighbors regression model.

    Args:
        n_neighbors (int): Number of neighbors to consider (default: 5)
        **kwargs: Other hyperparameters passed to KNeighborsRegressor
    """
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__()
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)

    def fit(self, X, y):
        """Fit the KNN model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict the target value for new data."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Return the model's R² score on the test data."""
        return self.model.score(X, y)


class LinearRegressorModel(BaseRegressor):
    """
    Linear regression model.

    Args:
        **kwargs: Hyperparameters passed to LinearRegression
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LinearRegression(**kwargs)

    def fit(self, X, y):
        """Fit the linear regression model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict the target value for new data."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Return the model's R² score on the test data."""
        return self.model.score(X, y)


class RidgeRegressorModel(BaseRegressor):
    """
    Ridge (L2) regression model.

    Args:
        alpha (float): Regularization parameter (default: 1.0)
        **kwargs: Other hyperparameters passed to Ridge
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.model = Ridge(alpha=alpha, **kwargs)

    def fit(self, X, y):
        """Fit the Ridge model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict the target value for new data."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Return the model's R² score on the test data."""
        return self.model.score(X, y)


class LassoRegressorModel(BaseRegressor):
    """
    Lasso (L1) regression model.

    Args:
        alpha (float): Regularization parameter (default: 1.0)
        **kwargs: Other hyperparameters passed to Lasso
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__()
        self.model = Lasso(alpha=alpha, **kwargs)

    def fit(self, X, y):
        """Fit the Lasso model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict the target value for new data."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Return the model's R² score on the test data."""
        return self.model.score(X, y)
