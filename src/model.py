"""Linear Regression model implemented from scratch using Gradient Descent."""

from typing import List, Optional, Union
import numpy as np
import pandas as pd


class LinearRegressionGD:
    """Multiple Linear Regression using Batch Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size taken at each iteration during gradient descent.
    n_iterations : int, default=1000
        Maximum number of iterations through the dataset.
    fit_intercept : bool, default=True
        Whether to calculate the intercept (bias) for this model.
    verbose : bool, default=False
        If True, prints loss at regular intervals during training.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        verbose: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.cost_history: List[float] = []

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the Mean Squared Error cost function.
        
        Cost = (1 / (2 * m)) * sum((y_pred - y) ** 2)
        """
        m = len(y)
        predictions = np.dot(X, self.weights) + self.bias
        cost = (1.0 / (2.0 * m)) * np.sum((predictions - y) ** 2)
        return float(cost)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "LinearRegressionGD":
        """Fit linear model using Gradient Descent.
        
        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Training features of shape (n_samples, n_features).
        y : Union[np.ndarray, pd.Series]
            Target values of shape (n_samples,).
            
        Returns
        -------
        self : LinearRegressionGD
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy().reshape(-1)
        else:
            y = y.reshape(-1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for iteration in range(self.n_iterations):
            # Compute predictions: y_pred = Xw + b
            y_pred = np.dot(X, self.weights) + self.bias
            errors = y_pred - y

            # Compute gradients
            dw = (1.0 / n_samples) * np.dot(X.T, errors)
            db = (1.0 / n_samples) * np.sum(errors) if self.fit_intercept else 0.0

            # Update parameters
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db

            # Record cost
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

            if self.verbose and (iteration % (self.n_iterations // 10 or 1) == 0):
                print(f"Iteration {iteration:5d} | Cost: {cost:.6f}")

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict using the linear model.
        
        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Samples of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
            
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        return np.dot(X, self.weights) + self.bias


# Alias for flexible importing as indicated in structure
linear_model = LinearRegressionGD
