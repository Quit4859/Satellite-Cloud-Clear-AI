from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from data.loaders.base import BaseLoader


class Sentinel2Loader(BaseLoader):
    """Loader for Sentinel-2 multispectral satellite images.

    Sentinel-2 provides 13 spectral bands with resolutions of 10m, 20m, and 60m.
    It is widely used for land cover monitoring and vegetation analysis.
    """

    BANDS = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B08A", "B09", "B10", "B11", "B12"
    ]
    WAVELENGTHS = {
        "B01": "0.443 μm (Coastal aerosol)",
        "B02": "0.490 μm (Blue)",
        "B03": "0.560 μm (Green)",
        "B04": "0.665 μm (Red)",
        "B05": "0.705 μm (Red Edge 1)",
        "B06": "0.740 μm (Red Edge 2)",
        "B07": "0.783 μm (Red Edge 3)",
        "B08": "0.842 μm (NIR)",
        "B08A": "0.865 μm (NIR narrow)",
        "B09": "0.945 μm (Water vapour)",
        "B10": "1.375 μm (Cirrus)",
        "B11": "1.610 μm (SWIR 1)",
        "B12": "2.190 μm (SWIR 2)",
    }
    RESOLUTIONS = {
        "10m": ["B02", "B03", "B04", "B08"],
        "20m": ["B05", "B06", "B07", "B08A", "B11", "B12"],
        "60m": ["B01", "B09", "B10"],
    }
    SENSOR_NAME = "Sentinel-2"

    def __init__(self, file_path: str | Path):
        """Initialize Sentinel-2 loader.

        Args:
            file_path: Path to the Sentinel-2 GeoTIFF file.
        """
        super().__init__(file_path)
        self.sensor_name = self.SENSOR_NAME

    def get_bands(self) -> List[str]:
        """Return all available Sentinel-2 band names.

        Returns:
            List of all 13 band names.
        """
        return self.BANDS.copy()

    def get_default_bands(self) -> List[str]:
        """Return default bands for analysis (RGB + NIR).

        Returns:
            List of default band names: [B04, B03, B02, B08].
        """
        return ["B04", "B03", "B02", "B08"]

    def get_rgb_bands(self) -> List[str]:
        """Return bands for true color composite.

        Returns:
            List of RGB band names: [B04, B03, B02].
        """
        return ["B04", "B03", "B02"]

    def get_nir_band(self) -> str:
        """Return near-infrared band name.

        Returns:
            NIR band name: 'B08'.
        """
        return "B08"

    def get_swir_bands(self) -> List[str]:
        """Return shortwave infrared band names.

        Returns:
            List of SWIR band names: [B11, B12].
        """
        return ["B11", "B12"]

    def get_red_edge_bands(self) -> List[str]:
        """Return red edge band names.

        Returns:
            List of red edge band names: [B05, B06, B07].
        """
        return ["B05", "B06", "B07"]

    def get_10m_bands(self) -> List[str]:
        """Return bands with 10m resolution.

        Returns:
            List of 10m band names.
        """
        return self.RESOLUTIONS["10m"].copy()

    def get_20m_bands(self) -> List[str]:
        """Return bands with 20m resolution.

        Returns:
            List of 20m band names.
        """
        return self.RESOLUTIONS["20m"].copy()

    def get_60m_bands(self) -> List[str]:
        """Return bands with 60m resolution.

        Returns:
            List of 60m band names.
        """
        return self.RESOLUTIONS["60m"].copy()

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
        return self.read(bands=["B08", "B04", "B03"])

    def read_swir_composite(self) -> np.ndarray:
        """Read bands for SWIR composite.

        Returns:
            numpy array with shape (3, height, width).
        """
        return self.read(bands=["B12", "B08", "B04"])

    def compute_ndvi(self) -> np.ndarray:
        """Compute Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)

        Returns:
            numpy array with NDVI values in range [-1, 1].
        """
        nir = self.read(bands=["B08"])[0].astype(np.float32)
        red = self.read(bands=["B04"])[0].astype(np.float32)

        denominator = nir + red
        ndvi = np.where(denominator > 0, (nir - red) / denominator, 0)

        return ndvi

    def compute_ndwi(self) -> np.ndarray:
        """Compute Normalized Difference Water Index.

        NDWI = (Green - NIR) / (Green + NIR)

        Returns:
            numpy array with NDWI values in range [-1, 1].
        """
        green = self.read(bands=["B03"])[0].astype(np.float32)
        nir = self.read(bands=["B08"])[0].astype(np.float32)

        denominator = green + nir
        ndwi = np.where(denominator > 0, (green - nir) / denominator, 0)

        return ndwi

    def compute_ndmi(self) -> np.ndarray:
        """Compute Normalized Difference Moisture Index.

        NDMI = (NIR - SWIR1) / (NIR + SWIR1)

        Returns:
            numpy array with NDMI values in range [-1, 1].
        """
        nir = self.read(bands=["B08"])[0].astype(np.float32)
        swir1 = self.read(bands=["B11"])[0].astype(np.float32)

        denominator = nir + swir1
        ndmi = np.where(denominator > 0, (nir - swir1) / denominator, 0)

        return ndmi

    def compute_nbr(self) -> np.ndarray:
        """Compute Normalized Burn Ratio.

        NBR = (NIR - SWIR2) / (NIR + SWIR2)

        Returns:
            numpy array with NBR values in range [-1, 1].
        """
        nir = self.read(bands=["B08"])[0].astype(np.float32)
        swir2 = self.read(bands=["B12"])[0].astype(np.float32)

        denominator = nir + swir2
        nbr = np.where(denominator > 0, (nir - swir2) / denominator, 0)

        return nbr

    def compute_evi(self) -> np.ndarray:
        """Compute Enhanced Vegetation Index.

        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)

        Returns:
            numpy array with EVI values.
        """
        nir = self.read(bands=["B08"])[0].astype(np.float32)
        red = self.read(bands=["B04"])[0].astype(np.float32)
        blue = self.read(bands=["B02"])[0].astype(np.float32)

        denominator = nir + 6 * red - 7.5 * blue + 1
        evi = np.where(denominator > 0, 2.5 * (nir - red) / denominator, 0)

        return evi

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

        logger.info(f"All {len(expected)} Sentinel-2 bands validated")
        return True
