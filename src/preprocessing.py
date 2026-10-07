"""Data preprocessing utilities implemented from scratch."""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform initial cleaning on the raw mobile phone dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe.
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    
    # Strip whitespace from string columns if present
    string_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        
    return df_clean


def train_test_split_custom(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays or matrices into random train and test subsets from scratch.
    
    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        Features array or dataframe of shape (n_samples, n_features).
    y : Union[np.ndarray, pd.Series]
        Target array or series of shape (n_samples,) or (n_samples, 1).
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : Optional[int], default=None
        Seed used by the random number generator.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.
        
    Returns
    -------
    X_train, X_test, y_train, y_test : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
        
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError("X and y must have the same number of samples.")
        
    if random_state is not None:
        np.random.seed(random_state)
        
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
        
    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class StandardScalerCustom:
    """Standardize features by removing the mean and scaling to unit variance."""
    
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "StandardScalerCustom":
        """Compute the mean and std to be used for later scaling."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Prevent division by zero for constant features
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self
        
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Perform standardization by centering and scaling."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScalerCustom instance is not fitted yet.")
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return (X - self.mean_) / self.std_
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Scale back the data to the original representation."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScalerCustom instance is not fitted yet.")
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return (X * self.std_) + self.mean_


class MinMaxScalerCustom:
    """Transform features by scaling each feature to a given range [0, 1]."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "MinMaxScalerCustom":
        """Compute the minimum and maximum to be used for later scaling."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        return self
        
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Scale features of X according to feature_range."""
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("MinMaxScalerCustom instance is not fitted yet.")
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        range_diff = self.data_max_ - self.data_min_
        range_diff = np.where(range_diff == 0, 1.0, range_diff)
        
        X_scaled = (X - self.data_min_) / range_diff
        res = X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return res
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
