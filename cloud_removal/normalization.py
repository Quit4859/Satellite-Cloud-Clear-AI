"""
normalization.py
=================
Reflectance normalization helpers (raw DN <-> float32 [0,1]).
"""
from .common import np


def normalize_image(array: np.ndarray, src_max: float = 10000.0) -> np.ndarray:
    """Convert a raw Sentinel-2 array into float32 reflectance in [0, 1]."""
    arr = array.astype(np.float32) / np.float32(src_max)
    return np.clip(arr, 0.0, 1.0)


def denormalize_image(array: np.ndarray, dst_dtype: np.dtype, dst_max: float = 10000.0) -> np.ndarray:
    """Inverse of `normalize_image`."""
    arr = np.clip(array, 0.0, 1.0) * dst_max
    return arr.astype(dst_dtype)
