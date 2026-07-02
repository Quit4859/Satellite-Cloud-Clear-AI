"""LISS-IV data acquisition module for loading local GeoTIFF images."""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from loguru import logger


class LISS4Acquisition:
    """Handles loading and organizing local LISS-IV GeoTIFF images.

    LISS-IV (Linear Imaging Self-Scanning Sensor IV) is a multispectral sensor
    aboard Resourcesat-2/2A satellites providing imagery in 4 spectral bands
    (B2, B3, B4, B5) at 5.8m spatial resolution.
    """

    BANDS = ["B2", "B3", "B4", "B5"]
    WAVELENGTHS = {
        "B2": "0.52-0.59 μm (Green)",
        "B3": "0.62-0.68 μm (Red)",
        "B4": "0.77-0.86 μm (NIR)",
        "B5": "1.55-1.70 μm (SWIR)",
    }
    RESOLUTION = 5.8  # meters

    def __init__(self, data_dir: str | Path = "data/liss4"):
        """Initialize LISS-IV acquisition handler.

        Args:
            data_dir: Directory for storing LISS-IV images.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, file_path: str | Path) -> Dict:
        """Load a single LISS-IV GeoTIFF image.

        Args:
            file_path: Path to the GeoTIFF file.

        Returns:
            Dictionary with 'data', 'metadata', 'transform', 'bounds', 'crs'.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with rasterio.open(file_path, "r") as src:
            data = src.read().astype(np.float32)
            metadata = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "driver": src.driver,
                "file_path": str(file_path),
                "sensor": "LISS-IV",
                "resolution": self.RESOLUTION,
            }

        logger.info(
            f"Loaded LISS-IV image: {file_path.name} "
            f"shape={data.shape} crs={metadata['crs']}"
        )

        return {
            "data": data,
            "metadata": metadata,
            "transform": metadata["transform"],
            "bounds": metadata["bounds"],
            "crs": metadata["crs"],
        }

    def load_images(self, file_paths: List[str | Path]) -> List[Dict]:
        """Load multiple LISS-IV GeoTIFF images.

        Args:
            file_paths: List of paths to GeoTIFF files.

        Returns:
            List of result dictionaries.
        """
        results = []
        for path in file_paths:
            try:
                result = self.load_image(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                results.append(None)

        successful = sum(1 for r in results if r is not None)
        logger.info(f"Loaded {successful}/{len(file_paths)} LISS-IV images")
        return results

    def copy_to_data_dir(
        self,
        source_path: str | Path,
        new_name: Optional[str] = None,
    ) -> Path:
        """Copy a LISS-IV image to the data directory.

        Args:
            source_path: Path to source GeoTIFF file.
            new_name: Optional new filename (without extension).

        Returns:
            Path to the copied file in the data directory.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if new_name:
            dest_name = f"{new_name}.tif"
        else:
            dest_name = source_path.name

        dest_path = self.data_dir / dest_name
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied LISS-IV image to {dest_path}")
        return dest_path

    def list_images(self) -> List[Path]:
        """List all LISS-IV images in the data directory.

        Returns:
            List of paths to GeoTIFF files.
        """
        patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        images = []
        for pattern in patterns:
            images.extend(self.data_dir.glob(pattern))
        return sorted(images)

    def get_metadata(self, file_path: str | Path) -> Dict:
        """Get metadata from a LISS-IV image without loading data.

        Args:
            file_path: Path to the GeoTIFF file.

        Returns:
            Dictionary with image metadata.
        """
        file_path = Path(file_path)
        with rasterio.open(file_path, "r") as src:
            metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "bands": self.BANDS[:src.count],
                "dtype": src.dtypes[0],
                "crs": str(src.crs),
                "transform": list(src.transform),
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "nodata": src.nodata,
                "resolution": self.RESOLUTION,
                "sensor": "LISS-IV",
            }
        return metadata

    def validate_image(self, file_path: str | Path) -> bool:
        """Validate that a file is a valid LISS-IV GeoTIFF.

        Args:
            file_path: Path to the GeoTIFF file.

        Returns:
            True if valid, False otherwise.
        """
        try:
            with rasterio.open(file_path, "r") as src:
                if src.count < 1 or src.count > 4:
                    logger.warning(
                        f"Unexpected band count {src.count} for LISS-IV"
                    )
                    return False
                if src.driver != "GTiff":
                    logger.warning(f"Not a GeoTIFF: {src.driver}")
                    return False
                return True
        except Exception as e:
            logger.error(f"Validation failed for {file_path}: {e}")
            return False
