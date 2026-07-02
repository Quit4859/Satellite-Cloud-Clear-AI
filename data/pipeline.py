from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import Affine
from loguru import logger

from data.loaders import LISS4Loader, Sentinel1Loader, Sentinel2Loader, BaseLoader
from data.processors import Normalizer, Resizer


class SatelliteDataPipeline:
    """Complete data pipeline for satellite image processing.

    Loads GeoTIFF satellite images, normalizes values, resizes images,
    and saves processed outputs.
    """

    LOADER_MAP = {
        "liss4": LISS4Loader,
        "sentinel1": Sentinel1Loader,
        "sentinel2": Sentinel2Loader,
    }

    def __init__(
        self,
        sensor_type: str,
        output_dir: str | Path = "processed_data",
        normalize_method: str = "min_max",
        target_size: Tuple[int, int] = (256, 256),
        clip_percentile: Optional[float] = 2.0,
    ):
        """Initialize the pipeline.

        Args:
            sensor_type: Sensor type ('liss4', 'sentinel1', 'sentinel2').
            output_dir: Directory for saving processed images.
            normalize_method: Normalization method ('min_max', 'z_score', 'percentile').
            target_size: Target (height, width) for resized images.
            clip_percentile: Percentile for outlier clipping (None to disable).
        """
        if sensor_type not in self.LOADER_MAP:
            raise ValueError(
                f"Unknown sensor: {sensor_type}. Use: {list(self.LOADER_MAP.keys())}"
            )

        self.sensor_type = sensor_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.normalizer = Normalizer(
            method=normalize_method,
            clip_percentile=clip_percentile,
        )
        self.resizer = Resizer(target_size=target_size)

        self._loader_class = self.LOADER_MAP[sensor_type]

    def process_file(
        self,
        input_path: str | Path,
        bands: Optional[List[str]] = None,
        save: bool = True,
        output_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Process a single satellite image file.

        Args:
            input_path: Path to input GeoTIFF file.
            bands: Bands to process (None for defaults).
            save: Whether to save the processed image.
            output_name: Custom output filename (without extension).

        Returns:
            Dictionary with 'data', 'transform', and 'metadata' keys.
        """
        input_path = Path(input_path)
        logger.info(f"Processing {input_path.name}")

        with self._loader_class(input_path) as loader:
            data = loader.read(bands=bands)
            transform = loader.transform
            metadata = loader.metadata

        processed = self._process_data(data)

        if save:
            output_path = self._get_output_path(input_path, output_name)
            self._save_image(processed, output_path, transform, metadata)

        return {
            "data": processed,
            "transform": transform,
            "metadata": metadata,
        }

    def process_batch(
        self,
        input_paths: List[str | Path],
        bands: Optional[List[str]] = None,
        save: bool = True,
    ) -> List[Dict[str, np.ndarray]]:
        """Process multiple satellite image files.

        Args:
            input_paths: List of input GeoTIFF file paths.
            bands: Bands to process (None for defaults).
            save: Whether to save processed images.

        Returns:
            List of result dictionaries.
        """
        results = []
        for path in input_paths:
            try:
                result = self.process_file(path, bands=bands, save=save)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append(None)

        successful = sum(1 for r in results if r is not None)
        logger.info(f"Processed {successful}/{len(input_paths)} files successfully")
        return results

    def _process_data(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization and resizing.

        Args:
            data: Raw image data.

        Returns:
            Processed image data.
        """
        data = self.normalizer.normalize(data)
        data = self.resizer.resize(data)
        return data

    def _get_output_path(
        self,
        input_path: Path,
        output_name: Optional[str] = None,
    ) -> Path:
        """Generate output file path.

        Args:
            input_path: Original input path.
            output_name: Custom output name.

        Returns:
            Full output path.
        """
        if output_name:
            filename = f"{output_name}.tif"
        else:
            filename = f"processed_{input_path.name}"
        return self.output_dir / filename

    def _save_image(
        self,
        data: np.ndarray,
        output_path: Path,
        transform: Affine,
        metadata: Dict,
    ) -> None:
        """Save processed image as GeoTIFF.

        Args:
            data: Image data to save (bands, height, width).
            output_path: Output file path.
            transform: Affine transform to preserve georeference.
            metadata: Dataset metadata.
        """
        bands, height, width = data.shape

        profile = {
            "driver": "GTiff",
            "dtype": data.dtype,
            "width": width,
            "height": height,
            "count": bands,
            "crs": metadata.get("crs"),
            "transform": transform,
            "compress": "lzw",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(bands):
                dst.write(data[i], i + 1)

        logger.info(f"Saved: {output_path}")

    def normalize_only(self, data: np.ndarray) -> np.ndarray:
        """Normalize data without resizing.

        Args:
            data: Input array.

        Returns:
            Normalized array.
        """
        return self.normalizer.normalize(data)

    def resize_only(self, data: np.ndarray) -> np.ndarray:
        """Resize data without normalizing.

        Args:
            data: Input array.

        Returns:
            Resized array.
        """
        return self.resizer.resize(data)

    def get_loader(self, file_path: str | Path) -> BaseLoader:
        """Get appropriate loader for a file.

        Args:
            file_path: Path to satellite image file.

        Returns:
            Loader instance for the configured sensor type.
        """
        return self._loader_class(file_path)
