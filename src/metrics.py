"""Evaluation metrics implemented from scratch."""

from typing import Dict, Union
import numpy as np
import pandas as pd


def _to_numpy_1d(y: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Helper to convert array-like input to a 1D numpy array."""
    if isinstance(y, (pd.Series, pd.DataFrame)):
        return y.to_numpy().reshape(-1)
    return np.asarray(y).reshape(-1)


def mean_absolute_error(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Mean Absolute Error (MAE).
    
    MAE = (1 / n) * sum(|y_true - y_pred|)
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(np.abs(yt - yp)))


def mean_squared_error(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Mean Squared Error (MSE).
    
    MSE = (1 / n) * sum((y_true - y_pred) ** 2)
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean((yt - yp) ** 2))


def root_mean_squared_error(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Root Mean Squared Error (RMSE).
    
    RMSE = sqrt(MSE)
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Coefficient of Determination (R^2 Score).
    
    R^2 = 1 - (SS_res / SS_tot)
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length.")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def evaluate_model(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    weights: np.ndarray | None = None,
    bias: float | None = None,
) -> Dict[str, float]:
    """Calculate common regression metrics.

    With ``weights`` and ``bias``, ``y_true`` is treated as ``X_test`` and
    predictions are generated using the notebook's scratch-model convention.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, MSE, RMSE, and R2.
    """
    if (weights is None) != (bias is None):
        raise ValueError("weights and bias must be provided together.")
    if weights is not None and bias is not None:
        X_test = np.asarray(y_true)
        y_true = y_pred
        y_pred = np.dot(X_test, weights) + bias

    values = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    # Keep the original package's uppercase keys as backwards-compatible aliases.
    return {
        **values,
        "MAE": values["mae"],
        "MSE": values["mse"],
        "RMSE": values["rmse"],
        "R2": values["r2"],
    }
