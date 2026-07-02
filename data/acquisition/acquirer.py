"""Top-level satellite data acquisition functions.

Provides a unified interface for downloading and loading satellite imagery
from multiple sensors (LISS-IV, Sentinel-1, Sentinel-2) with temporal support.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger

from data.acquisition.liss4 import LISS4Acquisition
from data.acquisition.sentinel1 import Sentinel1Acquisition
from data.acquisition.sentinel2 import Sentinel2Acquisition
from data.acquisition.temporal import TemporalSelector


class SatelliteAcquirer:
    """Unified satellite data acquisition interface.

    Provides methods to download and load imagery from multiple sensors
    with temporal support for multi-temporal analysis.
    """

    def __init__(
        self,
        base_dir: str | Path = "data",
        gee_project_id: Optional[str] = None,
    ):
        """Initialize the satellite acquirer.

        Args:
            base_dir: Base directory for all satellite data.
            gee_project_id: Google Earth Engine project ID.
        """
        self.base_dir = Path(base_dir)

        self.liss4 = LISS4Acquisition(data_dir=self.base_dir / "liss4")
        self.sentinel1 = Sentinel1Acquisition(
            project_id=gee_project_id,
            data_dir=self.base_dir / "sentinel1",
        )
        self.sentinel2 = Sentinel2Acquisition(
            project_id=gee_project_id,
            data_dir=self.base_dir / "sentinel2",
        )
        self.temporal = TemporalSelector(temporal_dir=self.base_dir / "temporal")

        # Create processed directory
        processed_dir = self.base_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SatelliteAcquirer initialized with base_dir={self.base_dir}")

    def load_liss4(self, file_path: str | Path) -> Dict:
        """Load a local LISS-IV GeoTIFF image.

        Args:
            file_path: Path to the LISS-IV GeoTIFF file.

        Returns:
            Dictionary with image data and metadata.
        """
        return self.liss4.load_image(file_path)

    def download_sentinel2(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        output_name: Optional[str] = None,
        resolution: int = 10,
        cloud_max: int = 20,
    ) -> Path:
        """Download Sentinel-2 imagery using Google Earth Engine.

        Args:
            bounds: (west, south, east, north) coordinates.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            output_name: Custom output filename.
            resolution: Output resolution (10, 20, or 60 meters).
            cloud_max: Maximum cloud cover percentage.

        Returns:
            Path to the downloaded GeoTIFF.
        """
        return self.sentinel2.download_image(
            bounds=bounds,
            start_date=start_date,
            end_date=end_date,
            output_name=output_name,
            resolution=resolution,
            cloud_max=cloud_max,
        )

    def download_sentinel1(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        polarization: str = "VV",
        output_name: Optional[str] = None,
        resolution: int = 10,
    ) -> Path:
        """Download Sentinel-1 SAR imagery using Google Earth Engine.

        Args:
            bounds: (west, south, east, north) coordinates.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            polarization: Polarization mode (VV or VH).
            output_name: Custom output filename.
            resolution: Output resolution in meters.

        Returns:
            Path to the downloaded GeoTIFF.
        """
        return self.sentinel1.download_image(
            bounds=bounds,
            start_date=start_date,
            end_date=end_date,
            polarization=polarization,
            output_name=output_name,
            resolution=resolution,
        )

    def get_temporal_images(
        self,
        target_date: str | datetime,
        sensor: Optional[str] = None,
        n_previous: int = 1,
        n_next: int = 1,
        max_time_delta_days: int = 365,
    ) -> Dict[str, List[Tuple[Path, datetime]]]:
        """Get previous and next temporal images for a target date.

        Args:
            target_date: Reference date (YYYY-MM-DD string or datetime).
            sensor: Sensor type ('liss4', 'sentinel1', 'sentinel2', or None for all).
            n_previous: Number of previous images to retrieve.
            n_next: Number of next images to retrieve.
            max_time_delta_days: Maximum allowed time difference in days.

        Returns:
            Dictionary with 'target', 'previous', and 'next' keys.
        """
        if sensor:
            return self.temporal.get_temporal_images_for_sensor(
                target_date=target_date,
                sensor=sensor,
                n_previous=n_previous,
                n_next=n_next,
                max_time_delta_days=max_time_delta_days,
            )

        results = {}
        for sensor_name in ["liss4", "sentinel1", "sentinel2"]:
            sensor_dir = self.base_dir / sensor_name
            if sensor_dir.exists():
                sensor_result = self.temporal.get_temporal_images(
                    target_date=target_date,
                    directory=sensor_dir,
                    n_previous=n_previous,
                    n_next=n_next,
                    max_time_delta_days=max_time_delta_days,
                )
                if sensor_result["previous"] or sensor_result["next"]:
                    results[sensor_name] = sensor_result

        return results


def download_liss4(
    file_path: str | Path,
    data_dir: str | Path = "data/liss4",
) -> Dict:
    """Load a local LISS-IV GeoTIFF image.

    Args:
        file_path: Path to the LISS-IV GeoTIFF file.
        data_dir: Directory for LISS-IV data.

    Returns:
        Dictionary with image data and metadata.
    """
    acquirer = LISS4Acquisition(data_dir=data_dir)
    return acquirer.load_image(file_path)


def download_sentinel2(
    bounds: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    output_name: Optional[str] = None,
    resolution: int = 10,
    cloud_max: int = 20,
    project_id: Optional[str] = None,
    data_dir: str | Path = "data/sentinel2",
) -> Path:
    """Download Sentinel-2 imagery using Google Earth Engine.

    Args:
        bounds: (west, south, east, north) coordinates.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        output_name: Custom output filename.
        resolution: Output resolution (10, 20, or 60 meters).
        cloud_max: Maximum cloud cover percentage.
        project_id: Google Earth Engine project ID.
        data_dir: Directory for storing downloaded images.

    Returns:
        Path to the downloaded GeoTIFF.
    """
    acquirer = Sentinel2Acquisition(project_id=project_id, data_dir=data_dir)
    return acquirer.download_image(
        bounds=bounds,
        start_date=start_date,
        end_date=end_date,
        output_name=output_name,
        resolution=resolution,
        cloud_max=cloud_max,
    )


def download_sentinel1(
    bounds: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    polarization: str = "VV",
    output_name: Optional[str] = None,
    resolution: int = 10,
    project_id: Optional[str] = None,
    data_dir: str | Path = "data/sentinel1",
) -> Path:
    """Download Sentinel-1 SAR imagery using Google Earth Engine.

    Args:
        bounds: (west, south, east, north) coordinates.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        polarization: Polarization mode (VV or VH).
        output_name: Custom output filename.
        resolution: Output resolution in meters.
        project_id: Google Earth Engine project ID.
        data_dir: Directory for storing downloaded images.

    Returns:
        Path to the downloaded GeoTIFF.
    """
    acquirer = Sentinel1Acquisition(project_id=project_id, data_dir=data_dir)
    return acquirer.download_image(
        bounds=bounds,
        start_date=start_date,
        end_date=end_date,
        polarization=polarization,
        output_name=output_name,
        resolution=resolution,
    )


def get_temporal_images(
    target_date: str | datetime,
    sensor: str,
    n_previous: int = 1,
    n_next: int = 1,
    max_time_delta_days: int = 365,
    data_dir: str | Path = "data",
) -> Dict[str, List[Tuple[Path, datetime]]]:
    """Get previous and next temporal images for a target date.

    Args:
        target_date: Reference date (YYYY-MM-DD string or datetime).
        sensor: Sensor type ('liss4', 'sentinel1', 'sentinel2').
        n_previous: Number of previous images to retrieve.
        n_next: Number of next images to retrieve.
        max_time_delta_days: Maximum allowed time difference in days.
        data_dir: Base data directory.

    Returns:
        Dictionary with 'target', 'previous', and 'next' keys.
    """
    selector = TemporalSelector(temporal_dir=Path(data_dir) / "temporal")
    return selector.get_temporal_images_for_sensor(
        target_date=target_date,
        sensor=sensor,
        n_previous=n_previous,
        n_next=n_next,
        max_time_delta_days=max_time_delta_days,
    )
