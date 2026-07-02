from typing import Optional, Tuple

import numpy as np
from loguru import logger


class Normalizer:
    """Normalizes satellite image data to specified ranges."""

    def __init__(
        self,
        method: str = "min_max",
        target_range: Tuple[float, float] = (0.0, 1.0),
        clip_percentile: Optional[float] = None,
    ):
        """Initialize the normalizer.

        Args:
            method: Normalization method ('min_max', 'z_score', 'percentile').
            target_range: Target range for min_max normalization.
            clip_percentile: Percentile values for clipping outliers (e.g., 2.0, 98.0).
        """
        self.method = method
        self.target_range = target_range
        self.clip_percentile = clip_percentile

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize image data.

        Args:
            data: Input array with shape (bands, height, width) or (height, width).

        Returns:
            Normalized array.
        """
        if self.clip_percentile is not None:
            data = self._clip_outliers(data)

        if self.method == "min_max":
            return self._min_max_normalize(data)
        elif self.method == "z_score":
            return self._z_score_normalize(data)
        elif self.method == "percentile":
            return self._percentile_normalize(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _clip_outliers(self, data: np.ndarray) -> np.ndarray:
        """Clip outlier values using percentile thresholds.

        Args:
            data: Input array.

        Returns:
            Clipped array.
        """
        if isinstance(self.clip_percentile, (tuple, list)):
            low_pct, high_pct = self.clip_percentile
        else:
            low_pct = self.clip_percentile
            high_pct = 100 - self.clip_percentile

        low, high = np.percentile(data, [low_pct, high_pct])
        return np.clip(data, low, high)

    def _min_max_normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max normalization.

        Args:
            data: Input array.

        Returns:
            Normalized array in target_range.
        """
        data_min = data.min()
        data_max = data.max()

        if data_max - data_min < 1e-10:
            logger.warning("Data has zero range, returning zeros")
            return np.zeros_like(data)

        normalized = (data - data_min) / (data_max - data_min)
        min_val, max_val = self.target_range
        return min_val + normalized * (max_val - min_val)

    def _z_score_normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Args:
            data: Input array.

        Returns:
            Standardized array with mean=0, std=1.
        """
        mean = data.mean()
        std = data.std()

        if std < 1e-10:
            logger.warning("Data has zero std, returning zeros")
            return np.zeros_like(data)

        return (data - mean) / std

    def _percentile_normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply percentile-based normalization.

        Uses 2nd and 98th percentiles as bounds.

        Args:
            data: Input array.

        Returns:
            Normalized array.
        """
        p2, p98 = np.percentile(data, [2, 98])

        if p98 - p2 < 1e-10:
            logger.warning("Data percentile range is zero, returning zeros")
            return np.zeros_like(data)

        normalized = (data - p2) / (p98 - p2)
        min_val, max_val = self.target_range
        return np.clip(min_val + normalized * (max_val - min_val), min_val, max_val)

    def fit(self, data: np.ndarray) -> "Normalizer":
        """Fit normalizer statistics on training data.

        Args:
            data: Training data to compute statistics from.

        Returns:
            Self for method chaining.
        """
        self._mean = data.mean()
        self._std = data.std()
        self._min = data.min()
        self._max = data.max()
        self._percentiles = np.percentile(data, [2, 98])
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted statistics.

        Args:
            data: Data to transform.

        Returns:
            Transformed array.
        """
        if not hasattr(self, "_mean"):
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        if self.method == "min_max":
            if self._max - self._min < 1e-10:
                return np.zeros_like(data)
            normalized = (data - self._min) / (self._max - self._min)
            min_val, max_val = self.target_range
            return min_val + normalized * (max_val - min_val)
        elif self.method == "z_score":
            if self._std < 1e-10:
                return np.zeros_like(data)
            return (data - self._mean) / self._std
        elif self.method == "percentile":
            p2, p98 = self._percentiles
            if p98 - p2 < 1e-10:
                return np.zeros_like(data)
            normalized = (data - p2) / (p98 - p2)
            min_val, max_val = self.target_range
            return np.clip(min_val + normalized * (max_val - min_val), min_val, max_val)
        else:
            raise ValueError(f"Unknown method: {self.method}")
