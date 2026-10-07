"""Unit tests for the scratch Linear Regression model, preprocessing, and metrics."""

import numpy as np
import pytest

from src.preprocessing import StandardScalerCustom, train_test_split_custom
from src.model import LinearRegressionGD
from src.metrics import evaluate_model, mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


def test_train_test_split():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    X_train, X_test, y_train, y_test = train_test_split_custom(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    assert len(X_train) == 40
    assert len(X_test) == 10
    assert len(y_train) == 40
    assert len(y_test) == 10


def test_standard_scaler():
    X = np.array([[10.0, 200.0], [20.0, 400.0], [30.0, 600.0]])
    scaler = StandardScalerCustom()
    X_scaled = scaler.fit_transform(X)
    
    # Scaled data should have mean ~ 0 and std ~ 1
    np.testing.assert_almost_equal(np.mean(X_scaled, axis=0), np.zeros(2))
    np.testing.assert_almost_equal(np.std(X_scaled, axis=0), np.ones(2))
    
    # Test inverse transform
    X_inv = scaler.inverse_transform(X_scaled)
    np.testing.assert_almost_equal(X_inv, X)


def test_linear_regression_gd_convergence():
    # Synthetic linear dataset: y = 3*x1 + 2*x2 + 5
    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_weights = np.array([3.0, 2.0])
    true_bias = 5.0
    y = np.dot(X, true_weights) + true_bias
    
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=300)
    model.fit(X, y)
    
    # Cost should strictly decrease or converge
    assert model.cost_history[0] > model.cost_history[-1]
    
    # Predictions should be close to true values
    y_pred = model.predict(X)
    assert np.allclose(y_pred, y, atol=1e-1)
    
    # Learned parameters should be close to true parameters
    np.testing.assert_almost_equal(model.weights, true_weights, decimal=1)
    np.testing.assert_almost_equal(model.bias, true_bias, decimal=1)


def test_metrics():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    
    # MAE = (0.5 + 0.5 + 0.0 + 1.0) / 4 = 0.5
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.5)
    
    # MSE = (0.25 + 0.25 + 0.0 + 1.0) / 4 = 0.375
    assert mean_squared_error(y_true, y_pred) == pytest.approx(0.375)
    
    # RMSE = sqrt(0.375)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(0.375))
    
    metrics = evaluate_model(y_true, y_pred)
    assert "MAE" in metrics
    assert "MSE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["R2"] > 0.9
