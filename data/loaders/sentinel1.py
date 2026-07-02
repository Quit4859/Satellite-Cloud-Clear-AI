from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from data.loaders.base import BaseLoader


class Sentinel1Loader(BaseLoader):
    """Loader for Sentinel-1 SAR satellite images.

    Sentinel-1 provides C-band Synthetic Aperture Radar (SAR) data.
    It offers dual-polarization (VV, VH) imagery with 10m resolution.
    """

    BANDS = ["VV", "VH"]
    POLARIZATION_DESCRIPTIONS = {
        "VV": "Vertical-Vertical (co-polarization)",
        "VH": "Vertical-Horizontal (cross-polarization)",
    }
    RESOLUTION = 10  # meters
    SENSOR_NAME = "Sentinel-1"
    WAVELENGTH = 5.405  # GHz (C-band)

    def __init__(self, file_path: str | Path):
        """Initialize Sentinel-1 loader.

        Args:
            file_path: Path to the Sentinel-1 GeoTIFF file.
        """
        super().__init__(file_path)
        self.sensor_name = self.SENSOR_NAME

    def get_bands(self) -> List[str]:
        """Return all available Sentinel-1 band names.

        Returns:
            List of band names: [VV, VH].
        """
        return self.BANDS.copy()

    def get_default_bands(self) -> List[str]:
        """Return default bands for analysis.

        Returns:
            List of default band names: [VV, VH].
        """
        return ["VV", "VH"]

    def get_vv_band(self) -> str:
        """Return VV polarization band name.

        Returns:
            VV band name: 'VV'.
        """
        return "VV"

    def get_vh_band(self) -> str:
        """Return VH polarization band name.

        Returns:
            VH band name: 'VH'.
        """
        return "VH"

    def get_polarization_info(self) -> Dict[str, str]:
        """Return polarization descriptions.

        Returns:
            Dictionary mapping polarization names to descriptions.
        """
        return self.POLARIZATION_DESCRIPTIONS.copy()

    def read_sar(self, polarization: Optional[str] = None) -> np.ndarray:
        """Read SAR data for specified polarization.

        Args:
            polarization: Specific polarization to read ('VV' or 'VH').
                         If None, reads both.

        Returns:
            numpy array with SAR backscatter values.
        """
        if polarization is None:
            return self.read(bands=self.BANDS)
        elif polarization in self.BANDS:
            return self.read(bands=[polarization])
        else:
            raise ValueError(f"Invalid polarization: {polarization}. Use VV or VH.")

    def to_db(self, data: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """Convert linear power to decibels (dB).

        dB = 10 * log10(power)

        Args:
            data: Linear power values.
            epsilon: Small value to avoid log(0).

        Returns:
            numpy array with values in dB scale.
        """
        return 10 * np.log10(np.maximum(data, epsilon))

    def to_linear(self, data_db: np.ndarray) -> np.ndarray:
        """Convert decibels to linear power.

        power = 10^(dB/10)

        Args:
            data_db: Values in dB scale.

        Returns:
            numpy array with linear power values.
        """
        return 10 ** (data_db / 10)

    def compute_rvi(self) -> np.ndarray:
        """Compute Radar Vegetation Index (RVI).

        RVI = 4 * VH / (VV + VH)

        Returns:
            numpy array with RVI values in range [0, 4].
        """
        vv = self.read(bands=["VV"])[0].astype(np.float32)
        vh = self.read(bands=["VH"])[0].astype(np.float32)

        denominator = vv + vh
        rvi = np.where(denominator > 0, (4 * vh) / denominator, 0)

        return rvi

    def compute_backscatter_ratio(self) -> np.ndarray:
        """Compute VH/VV backscatter ratio.

        Returns:
            numpy array with VH/VV ratio.
        """
        vv = self.read(bands=["VV"])[0].astype(np.float32)
        vh = self.read(bands=["VH"])[0].astype(np.float32)

        ratio = np.where(vv > 0, vh / vv, 0)

        return ratio

    def filter_speckle(
        self,
        data: np.ndarray,
        kernel_size: int = 5,
        method: str = "mean",
    ) -> np.ndarray:
        """Apply speckle filter to SAR data.

        Args:
            data: Input SAR image.
            kernel_size: Size of the filtering kernel.
            method: Filtering method ('mean' or 'median').

        Returns:
            Filtered SAR image.
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available, returning unfiltered data")
            return data

        if method == "mean":
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
            filtered = cv2.filter2D(data, -1, kernel)
        elif method == "median":
            filtered = cv2.medianBlur(data, kernel_size)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mean' or 'median'.")

        return filtered

    def validate_bands(self) -> bool:
        """Validate that all expected bands are present.

        Returns:
            True if all bands are valid, False otherwise.
        """
        available = self.get_bands()
        expected = self.BANDS

        missing = [band for band in expected if band not in available]
        if missing:
            logger.warning(f"Missing bands: {missing}")
            return False

        logger.info(f"All {len(expected)} Sentinel-1 bands validated")
        return True
