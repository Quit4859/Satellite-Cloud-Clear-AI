"""Sentinel-1 SAR data acquisition module using Google Earth Engine."""

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


class Sentinel1Acquisition:
    """Handles Sentinel-1 SAR imagery download via Google Earth Engine.

    Sentinel-1 provides C-band Synthetic Aperture Radar data with dual
    polarization (VV, VH) at 10m resolution. SAR data is useful for
    cloud-penetrating observations.
    """

    POLARIZATIONS = ["VV", "VH"]
    RESOLUTION = 10  # meters
    WAVELENGTH = 5.405  # GHz (C-band)
    COLLECTION_ID = "COPERNICUS/S1_GRD"

    def __init__(
        self,
        project_id: Optional[str] = None,
        data_dir: str | Path = "data/sentinel1",
    ):
        """Initialize Sentinel-1 acquisition handler.

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
        polarization: str = "VV",
    ) -> "ee.ImageCollection":
        """Get Sentinel-1 GRD collection filtered by parameters.

        Args:
            aoi: Area of interest geometry.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            polarization: Polarization mode (VV or VH).

        Returns:
            Filtered Sentinel-1 image collection.
        """
        if polarization not in self.POLARIZATIONS:
            raise ValueError(
                f"Invalid polarization: {polarization}. "
                f"Use: {self.POLARIZATIONS}"
            )

        return (
            ee.ImageCollection(self.COLLECTION_ID)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", polarization))
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        )

    def download_image(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        polarization: str = "VV",
        output_name: Optional[str] = None,
        resolution: int = 10,
    ) -> Path:
        """Download a Sentinel-1 SAR composite.

        Args:
            bounds: (west, south, east, north) coordinates.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            polarization: Polarization to download (VV or VH).
            output_name: Custom output filename (without extension).
            resolution: Output resolution in meters.

        Returns:
            Path to the downloaded GeoTIFF file.
        """
        self.initialize_gee()

        aoi = self._get_aoi(bounds)
        collection = self._get_collection(aoi, start_date, end_date, polarization)

        count = collection.size().getInfo()
        if count == 0:
            raise ValueError(
                f"No Sentinel-1 images found for bounds={bounds} "
                f"date={start_date}/{end_date} pol={polarization}"
            )
        logger.info(f"Found {count} Sentinel-1 images")

        composite = collection.median().clip(aoi)

        return self._save_geotiff(
            composite=composite,
            aoi=aoi,
            resolution=resolution,
            output_name=output_name,
            start_date=start_date,
            end_date=end_date,
            polarization=polarization,
        )

    def download_dual_polarization(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        output_name: Optional[str] = None,
        resolution: int = 10,
    ) -> Path:
        """Download both VV and VH polarizations as a multi-band composite.

        Args:
            bounds: (west, south, east, north) coordinates.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            output_name: Custom output filename.
            resolution: Output resolution in meters.

        Returns:
            Path to the downloaded multi-band GeoTIFF file.
        """
        self.initialize_gee()

        aoi = self._get_aoi(bounds)

        vv_collection = self._get_collection(aoi, start_date, end_date, "VV")
        vh_collection = self._get_collection(aoi, start_date, end_date, "VH")

        vv_composite = vv_collection.select("VV").median().clip(aoi)
        vh_composite = vh_collection.select("VH").median().clip(aoi)

        composite = vv_composite.addBands(vh_composite)

        return self._save_geotiff(
            composite=composite,
            aoi=aoi,
            resolution=resolution,
            output_name=output_name,
            start_date=start_date,
            end_date=end_date,
            polarization="VV_VH",
        )

    def _save_geotiff(
        self,
        composite: "ee.Image",
        aoi: "ee.Geometry",
        resolution: int,
        output_name: Optional[str],
        start_date: str,
        end_date: str,
        polarization: str,
    ) -> Path:
        """Export SAR composite as GeoTIFF via GEE export.

        Args:
            composite: Processed Sentinel-1 composite.
            aoi: Area of interest.
            resolution: Output resolution.
            output_name: Custom output name.
            start_date: Start date.
            end_date: End date.
            polarization: Polarization identifier.

        Returns:
            Path to the exported GeoTIFF.
        """
        if output_name is None:
            output_name = f"S1_{polarization}_{start_date}_to_{end_date}"

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
        polarization: str = "VV",
        output_name: Optional[str] = None,
        resolution: int = 10,
    ) -> Path:
        """Download a single Sentinel-1 scene.

        Args:
            bounds: (west, south, east, north) coordinates.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            polarization: Polarization to download.
            output_name: Custom output filename.
            resolution: Output resolution in meters.

        Returns:
            Path to the downloaded GeoTIFF file.
        """
        self.initialize_gee()

        aoi = self._get_aoi(bounds)
        collection = self._get_collection(aoi, start_date, end_date, polarization)

        scene = ee.Image(collection.first())

        return self._save_geotiff(
            composite=scene,
            aoi=aoi,
            resolution=resolution,
            output_name=output_name,
            start_date=start_date,
            end_date=end_date,
            polarization=polarization,
        )

    def list_images(self) -> List[Path]:
        """List all downloaded Sentinel-1 images.

        Returns:
            List of paths to GeoTIFF files.
        """
        patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        images = []
        for pattern in patterns:
            images.extend(self.data_dir.glob(pattern))
        return sorted(images)

    def get_metadata(self, file_path: str | Path) -> Dict:
        """Get metadata from a Sentinel-1 image.

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
                "sensor": "Sentinel-1",
                "resolution": self.RESOLUTION,
            }
