"""Sentinel-2 data acquisition module using Google Earth Engine."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from loguru import logger
from rasterio.transform import from_bounds

try:
    import ee
except ImportError:
    ee = None
    logger.warning("earthengine-api not installed. GEE features will be unavailable.")


class Sentinel2Acquisition:
    """Handles Sentinel-2 imagery download via Google Earth Engine.

    Sentinel-2 provides 13 spectral bands at 10m, 20m, and 60m resolution.
    This class manages cloud-masked composites and single-scene downloads.
    """

    BANDS_10M = ["B2", "B3", "B4", "B8"]
    BANDS_20M = ["B5", "B6", "B7", "B8A", "B11", "B12"]
    BANDS_60M = ["B01", "B09", "B10"]
    ALL_BANDS = BANDS_10M + BANDS_20M + BANDS_60M

    RESOLUTIONS = {
        "10m": ["B02", "B03", "B04", "B08"],
        "20m": ["B05", "B06", "B07", "B08A", "B11", "B12"],
        "60m": ["B01", "B09", "B10"],
    }

    def __init__(
        self,
        project_id: Optional[str] = None,
        data_dir: str | Path = "data/sentinel2",
    ):
        """Initialize Sentinel-2 acquisition handler.

        Args:
            project_id: Google Earth Engine project ID.
            data_dir: Directory for storing downloaded images.
        """
        self.project_id = project_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def initialize_gee(self) -> None:
        """Initialize Google Earth Engine authentication.

        Raises:
            RuntimeError: If earthengine-api is not installed.
        """
        if ee is None:
            raise RuntimeError(
                "earthengine-api is not installed. "
                "Install it with: pip install earthengine-api"
            )

        if self._initialized:
            return

        try:
            ee.Initialize(project=self.project_id)
            self._initialized = True
            logger.info("Google Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GEE: {e}")
            raise

    def _get_aoi(self, bounds: Tuple[float, float, float, float]) -> "ee.Geometry":
        """Convert bounding box to GEE Geometry.

        Args:
            bounds: (west, south, east, north) coordinates.

        Returns:
            GEE Geometry object.
        """
        west, south, east, north = bounds
        return ee.Geometry.Rectangle([west, south, east, north])

    def _get_collection(
        self,
        aoi: "ee.Geometry",
        start_date: str,
        end_date: str,
        cloud_max: int = 20,
    ) -> "ee.ImageCollection":
        """Get Sentinel-2 collection filtered by date and cloud cover.

        Args:
            aoi: Area of interest geometry.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            cloud_max: Maximum cloud cover percentage.

        Returns:
            Filtered Sentinel-2 image collection.
        """
        return (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
        )

    def _mask_clouds(self, image: "ee.Image") -> "ee.Image":
        """Apply cloud mask using QA60 band.

        Args:
            image: Sentinel-2 image.

        Returns:
            Cloud-masked image.
        """
        qa = image.select("QA60")
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
            qa.bitwiseAnd(cirrus_bit_mask).eq(0)
        )
        return image.updateMask(mask).divide(10000)

    def download_image(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        output_name: Optional[str] = None,
        resolution: int = 10,
        cloud_max: int = 20,
    ) -> Path:
        """Download a cloud-free Sentinel-2 composite.

        Args:
            bounds: (west, south, east, north) coordinates.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            output_name: Custom output filename (without extension).
            resolution: Output resolution in meters (10, 20, or 60).
            cloud_max: Maximum cloud cover percentage.

        Returns:
            Path to the downloaded GeoTIFF file.
        """
        self.initialize_gee()

        aoi = self._get_aoi(bounds)
        collection = self._get_collection(aoi, start_date, end_date, cloud_max)

        count = collection.size().getInfo()
        if count == 0:
            raise ValueError(
                f"No Sentinel-2 images found for bounds={bounds} "
                f"date={start_date}/{end_date} cloud_max={cloud_max}"
            )
        logger.info(f"Found {count} Sentinel-2 images")

        composite = collection.map(self._mask_clouds).median().clip(aoi)

        bands = self.RESOLUTIONS.get(f"{resolution}m", self.BANDS_10M)
        composite = composite.select(bands)

        region = aoi.bounds().getInfo()["coordinates"]
        image_array = ee.Array(
            composite.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=resolution,
                maxPixels=1e9,
            ).getInfo()
        )

        grid = ee.Image.pixelLonLat().clip(aoi)
        coords = ee.Array(grid.sampleRegion(
            region=aoi, scale=resolution, projection="EPSG:4326", geometries=True
        ).aggregate_array(".geo").map(lambda g: ee.List(g).coordinates()))

        return self._save_geotiff(
            composite=composite,
            aoi=aoi,
            resolution=resolution,
            output_name=output_name,
            start_date=start_date,
            end_date=end_date,
        )

    def _save_geotiff(
        self,
        composite: "ee.Image",
        aoi: "ee.Geometry",
        resolution: int,
        output_name: Optional[str],
        start_date: str,
        end_date: str,
    ) -> Path:
        """Export composite as GeoTIFF via GEE export.

        Args:
            composite: Processed Sentinel-2 composite.
            aoi: Area of interest.
            resolution: Output resolution.
            output_name: Custom output name.
            start_date: Start date.
            end_date: End date.

        Returns:
            Path to the exported GeoTIFF.
        """
        if output_name is None:
            output_name = f"S2_{start_date}_to_{end_date}"

        output_path = self.data_dir / f"{output_name}.tif"

        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=output_name,
            folder=str(self.data_dir),
            region=aoi.bounds().getInfo()["coordinates"],
            scale=resolution,
            crs="EPSG:4326",
            maxPixels=1e9,
            fileFormat="GeoTIFF",
        )
        task.start()

        logger.info(f"Started GEE export task: {task.id}")
        logger.info(f"Export will be saved to: {output_path}")

        return output_path

    def download_single_scene(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        output_name: Optional[str] = None,
        resolution: int = 10,
        cloud_max: int = 20,
    ) -> Path:
        """Download a single Sentinel-2 scene (least cloudy).

        Args:
            bounds: (west, south, east, north) coordinates.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            output_name: Custom output filename.
            resolution: Output resolution in meters.
            cloud_max: Maximum cloud cover percentage.

        Returns:
            Path to the downloaded GeoTIFF file.
        """
        self.initialize_gee()

        aoi = self._get_aoi(bounds)
        collection = self._get_aoi(aoi, start_date, end_date, cloud_max)

        scene = ee.Image(collection.sort("CLOUDY_PIXEL_PERCENTAGE").first())

        return self._save_geotiff(
            composite=scene,
            aoi=aoi,
            resolution=resolution,
            output_name=output_name,
            start_date=start_date,
            end_date=end_date,
        )

    def list_images(self) -> List[Path]:
        """List all downloaded Sentinel-2 images.

        Returns:
            List of paths to GeoTIFF files.
        """
        patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        images = []
        for pattern in patterns:
            images.extend(self.data_dir.glob(pattern))
        return sorted(images)

    def get_metadata(self, file_path: str | Path) -> Dict:
        """Get metadata from a Sentinel-2 image.

        Args:
            file_path: Path to the GeoTIFF file.

        Returns:
            Dictionary with image metadata.
        """
        file_path = Path(file_path)
        with rasterio.open(file_path, "r") as src:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "width": src.width,
                "height": src.height,
                "count": src.count,
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
                "sensor": "Sentinel-2",
            }
