from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from loguru import logger


class BaseLoader(ABC):
    """Base class for satellite image loaders.

    Provides common functionality for loading GeoTIFF satellite images
    with different sensor-specific implementations.
    """

    def __init__(self, file_path: str | Path):
        """Initialize the loader.

        Args:
            file_path: Path to the GeoTIFF file.
        """
        self.file_path = Path(file_path)
        self._dataset: Optional[rasterio.DatasetReader] = None
        self._metadata: Dict = {}

    def __enter__(self) -> "BaseLoader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        """Open the GeoTIFF file for reading."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self._dataset = rasterio.open(self.file_path, "r")
        self._extract_metadata()
        logger.info(f"Opened {self.file_path.name} with shape {self.shape}")

    def close(self) -> None:
        """Close the dataset."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def _extract_metadata(self) -> None:
        """Extract metadata from the dataset."""
        if self._dataset is None:
            raise RuntimeError("Dataset not opened")

        self._metadata = {
            "width": self._dataset.width,
            "height": self._dataset.height,
            "count": self._dataset.count,
            "dtype": self._dataset.dtypes[0],
            "crs": self._dataset.crs,
            "transform": self._dataset.transform,
            "bounds": self._dataset.bounds,
            "nodata": self._dataset.nodata,
        }

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return image shape as (bands, height, width)."""
        if self._dataset is None:
            raise RuntimeError("Dataset not opened")
        return (self._dataset.count, self._dataset.height, self._dataset.width)

    @property
    def metadata(self) -> Dict:
        """Return dataset metadata."""
        return self._metadata.copy()

    @property
    def crs(self) -> Optional[str]:
        """Return coordinate reference system."""
        return self._metadata.get("crs")

    @property
    def transform(self) -> Optional[Affine]:
        """Return affine transform."""
        return self._metadata.get("transform")

    @abstractmethod
    def get_bands(self) -> List[str]:
        """Return list of band names for this sensor.

        Returns:
            List of band names.
        """
        pass

    @abstractmethod
    def get_default_bands(self) -> List[str]:
        """Return default bands to load for this sensor.

        Returns:
            List of default band names.
        """
        pass

    def read(
        self,
        bands: Optional[List[str]] = None,
        window: Optional[rasterio.windows.Window] = None,
    ) -> np.ndarray:
        """Read image data from the dataset.

        Args:
            bands: List of band names to read. If None, reads all bands.
            window: Optional window to read a subset of the image.

        Returns:
            numpy array with shape (bands, height, width).
        """
        if self._dataset is None:
            raise RuntimeError("Dataset not opened. Call open() first.")

        if bands is None:
            data = self._dataset.read(window=window)
        else:
            band_indices = self._get_band_indices(bands)
            data = self._dataset.read(band_indices, window=window)

        return data.astype(np.float32)

    def _get_band_indices(self, bands: List[str]) -> List[int]:
        """Convert band names to 1-based indices.

        Args:
            bands: List of band names.

        Returns:
            List of 1-based band indices.
        """
        available_bands = self.get_bands()
        indices = []
        for band in bands:
            if band in available_bands:
                indices.append(available_bands.index(band) + 1)
            else:
                logger.warning(f"Band {band} not found in available bands")
        return indices

    def read_as_dict(self, bands: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Read image data as a dictionary of bands.

        Args:
            bands: List of band names to read. If None, reads all bands.

        Returns:
            Dictionary mapping band names to numpy arrays.
        """
        data = self.read(bands)
        if bands is None:
            bands = self.get_bands()

        return {band: data[i] for i, band in enumerate(bands)}
