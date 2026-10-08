"""Data preprocessing utilities implemented from scratch."""

import random
import re
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from the raw mobile phone dataset.
    
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
    df_clean.columns = df_clean.columns.str.strip()

    required = {
        "Storage", "RAM", "Price ($)", "Screen Size (inches)",
        "Camera (MP)", "Battery Capacity (mAh)",
    }
    missing = required.difference(df_clean.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df_clean["Storage"] = _extract_numeric(df_clean["Storage"], integer=True)
    df_clean["RAM"] = _extract_numeric(df_clean["RAM"], integer=True)
    df_clean["Price ($)"] = (
        df_clean["Price ($)"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).astype(int)
    )
    df_clean["Screen Size (inches)"] = _extract_numeric(
        df_clean["Screen Size (inches)"], integer=False
    )

    lens_list = df_clean["Camera (MP)"].apply(parse_lenses)
    if not lens_list.apply(bool).all():
        raise ValueError("Some rows have no parsable camera values.")
    camera_features = lens_list.apply(name_lenses)
    df_clean = pd.concat([df_clean, camera_features], axis=1)
    df_clean = df_clean.drop(columns=["Camera (MP)", "ambiguous"])
    df_clean["Num_camera"] = df_clean["Num_camera"].astype(int)
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.rename(columns={
        "Screen Size (inches)": "Screen_size",
        "Battery Capacity (mAh)": "Battery_cap",
        "Price ($)": "Price",
    })

    ordered_columns = [
        "Brand", "Model", "Storage", "RAM", "Screen_size", "Num_camera",
        "Cam_main", "Cam_ultra", "Cam_telephoto", "Cam_macro",
        "Battery_cap", "Price",
    ]
    return df_clean[ordered_columns]


def _extract_numeric(series: pd.Series, *, integer: bool) -> pd.Series:
    pattern = r"(\d+)" if integer else r"(\d+\.?\d*)"
    values = series.astype(str).str.extract(pattern, expand=False)
    return values.astype(int if integer else float)


def parse_lenses(value: object) -> list[float]:
    """Extract camera megapixels, ignoring non-MP tokens such as ``ToF``."""
    lenses: list[float] = []
    for part in str(value).split("+"):
        match = re.fullmatch(r"\s*(\d+\.?\d*)\s*(?:MP)?\s*", part, flags=re.I)
        if match:
            lenses.append(float(match.group(1)))
    return lenses


MACRO_MAX_MP = 5.0


def name_lenses(lenses: list[float]) -> pd.Series:
    """Map ordered camera resolutions to the engineered camera features."""
    if not lenses:
        raise ValueError("At least one camera resolution is required.")
    secondary = lenses[1:]
    macro = [value for value in secondary if value <= MACRO_MAX_MP]
    wide_or_tele = [value for value in secondary if value > MACRO_MAX_MP]
    return pd.Series({
        "Num_camera": len(lenses),
        "Cam_main": lenses[0],
        "Cam_ultra": wide_or_tele[0] if wide_or_tele else 0.0,
        "Cam_telephoto": wide_or_tele[1] if len(wide_or_tele) > 1 else 0.0,
        "Cam_macro": macro[0] if macro else 0.0,
        "ambiguous": len(wide_or_tele) >= 2,
    })


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
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1.")
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()

    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError("X and y must have the same number of samples.")
        
    rng = random.Random(random_state)
    indices = list(range(n_samples))
    if shuffle:
        rng.shuffle(indices)
        
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
