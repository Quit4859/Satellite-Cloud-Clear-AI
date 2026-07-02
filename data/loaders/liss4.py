from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from data.loaders.base import BaseLoader


class LISS4Loader(BaseLoader):
    """Loader for LISS-IV (Linear Imaging Self-Scanning Sensor IV) satellite images.

    LISS-IV is a multispectral sensor aboard Resourcesat-2/2A satellites.
    It provides imagery in 4 spectral bands with 5.8m spatial resolution.
    """

    BANDS = ["B2", "B3", "B4", "B5"]
    WAVELENGTHS = {
        "B2": "0.52-0.59 μm (Green)",
        "B3": "0.62-0.68 μm (Red)",
        "B4": "0.77-0.86 μm (NIR)",
        "B5": "1.55-1.70 μm (SWIR)",
    }
    RESOLUTION = 5.8  # meters
    SENSOR_NAME = "LISS-IV"

    def __init__(self, file_path: str | Path):
        """Initialize LISS-IV loader.

        Args:
            file_path: Path to the LISS-IV GeoTIFF file.
        """
        super().__init__(file_path)
        self.sensor_name = self.SENSOR_NAME

    def get_bands(self) -> List[str]:
        """Return all available LISS-IV band names.

        Returns:
            List of band names: [B2, B3, B4, B5].
        """
        return self.BANDS.copy()

    def get_default_bands(self) -> List[str]:
        """Return default bands for analysis (RGB + NIR).

        Returns:
            List of default band names: [B4, B3, B2, B5].
        """
        return ["B4", "B3", "B2", "B5"]

    def get_rgb_bands(self) -> List[str]:
        """Return bands for true color composite.

        Returns:
            List of RGB band names: [B4, B3, B2].
        """
        return ["B4", "B3", "B2"]

    def get_nir_band(self) -> str:
        """Return near-infrared band name.

        Returns:
            NIR band name: 'B4'.
        """
        return "B4"

    def get_swir_band(self) -> str:
        """Return shortwave infrared band name.

        Returns:
            SWIR band name: 'B5'.
        """
        return "B5"

    def get_wavelength_info(self) -> Dict[str, str]:
        """Return wavelength information for each band.

        Returns:
            Dictionary mapping band names to wavelength ranges.
        """
        return self.WAVELENGTHS.copy()

    def read_true_color(self) -> np.ndarray:
        """Read RGB bands for true color visualization.

        Returns:
            numpy array with shape (3, height, width) for RGB.
        """
        return self.read(bands=self.get_rgb_bands())

    def read_false_color(self) -> np.ndarray:
        """Read bands for false color composite (NIR, Red, Green).

        Returns:
            numpy array with shape (3, height, width).
        """
        return self.read(bands=["B4", "B3", "B2"])

    def compute_ndvi(self) -> np.ndarray:
        """Compute Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)

        Returns:
            numpy array with NDVI values in range [-1, 1].
        """
        nir = self.read(bands=["B4"])[0].astype(np.float32)
        red = self.read(bands=["B3"])[0].astype(np.float32)

        denominator = nir + red
        ndvi = np.where(denominator > 0, (nir - red) / denominator, 0)

        return ndvi

    def compute_ndwi(self) -> np.ndarray:
        """Compute Normalized Difference Water Index.

        NDWI = (Green - NIR) / (Green + NIR)

        Returns:
            numpy array with NDWI values in range [-1, 1].
        """
        green = self.read(bands=["B2"])[0].astype(np.float32)
        nir = self.read(bands=["B4"])[0].astype(np.float32)

        denominator = green + nir
        ndwi = np.where(denominator > 0, (green - nir) / denominator, 0)

        return ndwi

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

        logger.info(f"All {len(expected)} LISS-IV bands validated")
        return True
