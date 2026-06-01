import numpy as np
from sklearn.preprocessing import MinMaxScaler
from loguru import logger


class FeatureEncoder:
    """
    Normalizes 4-dim sensor features to [0, 2π] for angle encoding.
    
    Feature layout:
      [0] temperature_c      (°C)
      [1] vibration_rms      (m/s²)
      [2] current_a          (A)
      [3] accel_magnitude    (m/s²)
    """
    FEATURE_NAMES = ["temperature_c", "vibration_rms", "current_a", "accel_magnitude"]

    def __init__(self, n_features: int = 4):
        self.n_features = n_features
        self.scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureEncoder":
        X = np.atleast_2d(X)
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        self.scaler.fit(X)
        self._fitted = True
        logger.debug(f"Encoder fitted on {len(X)} samples")
        for i, name in enumerate(self.FEATURE_NAMES):
            logger.debug(f"  {name}: [{self.scaler.data_min_[i]:.2f}, {self.scaler.data_max_[i]:.2f}]")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        x = np.atleast_2d(x)
        encoded = self.scaler.transform(x)
        encoded = np.clip(encoded, 0, 2 * np.pi)
        return encoded

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
