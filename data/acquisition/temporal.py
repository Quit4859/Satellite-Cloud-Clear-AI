"""Temporal image selection module for multi-temporal satellite analysis."""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


class TemporalSelector:
    """Selects previous and next images based on acquisition date.

    Supports finding temporal neighbors from local directories or
    organizing temporal sequences for multi-temporal analysis.
    """

    DATE_PATTERNS = [
        r"(\d{4})-(\d{2})-(\d{2})",         # YYYY-MM-DD
        r"(\d{4})(\d{2})(\d{2})",            # YYYYMMDD
        r"(\d{4})\.(\d{2})\.(\d{2})",        # YYYY.MM.DD
        r"(\d{4})_(\d{2})_(\d{2})",          # YYYY_MM_DD
        r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})",  # YYYYMMDDTHHMM
    ]

    def __init__(self, temporal_dir: str | Path = "data/temporal"):
        """Initialize temporal selector.

        Args:
            temporal_dir: Directory for temporal image storage.
        """
        self.temporal_dir = Path(temporal_dir)
        self.temporal_dir.mkdir(parents=True, exist_ok=True)

    def extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract acquisition date from filename.

        Args:
            filename: Image filename.

        Returns:
            Datetime object if date found, None otherwise.
        """
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                try:
                    if len(groups) >= 5:
                        return datetime(
                            int(groups[0]), int(groups[1]), int(groups[2]),
                            int(groups[3]), int(groups[4]),
                        )
                    elif len(groups) >= 3:
                        return datetime(
                            int(groups[0]), int(groups[1]), int(groups[2])
                        )
                except ValueError:
                    continue
        return None

    def _get_image_date(self, file_path: Path) -> Optional[datetime]:
        """Get acquisition date from image file.

        Tries filename first, then falls back to metadata.

        Args:
            file_path: Path to image file.

        Returns:
            Acquisition date or None.
        """
        date = self.extract_date_from_filename(file_path.name)
        if date:
            return date

        try:
            import rasterio
            with rasterio.open(file_path, "r") as src:
                tags = src.tags()
                for key in ["ACQUISITION_DATE", "DATE_ACQUIRED", "datetime"]:
                    if key in tags:
                        date_str = tags[key]
                        for pattern in self.DATE_PATTERNS:
                            match = re.search(pattern, date_str)
                            if match:
                                groups = match.groups()
                                return datetime(
                                    int(groups[0]), int(groups[1]), int(groups[2])
                                )
        except Exception:
            pass

        return None

    def list_images_with_dates(
        self, directory: str | Path
    ) -> List[Tuple[Path, datetime]]:
        """List images with their acquisition dates.

        Args:
            directory: Directory containing satellite images.

        Returns:
            List of (file_path, datetime) tuples, sorted by date.
        """
        directory = Path(directory)
        patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
        images = []
        for pattern in patterns:
            images.extend(directory.glob(pattern))

        dated_images = []
        for img_path in images:
            date = self._get_image_date(img_path)
            if date:
                dated_images.append((img_path, date))
            else:
                logger.warning(f"Could not extract date from {img_path.name}")

        dated_images.sort(key=lambda x: x[1])
        return dated_images

    def get_temporal_images(
        self,
        target_date: str | datetime,
        directory: str | Path,
        n_previous: int = 1,
        n_next: int = 1,
        max_time_delta_days: int = 365,
    ) -> Dict[str, List[Tuple[Path, datetime]]]:
        """Get previous and next temporal images relative to target date.

        Args:
            target_date: Reference date (string YYYY-MM-DD or datetime).
            directory: Directory containing satellite images.
            n_previous: Number of previous images to retrieve.
            n_next: Number of next images to retrieve.
            max_time_delta_days: Maximum allowed time difference in days.

        Returns:
            Dictionary with 'target', 'previous', and 'next' keys.
        """
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, "%Y-%m-%d")

        dated_images = self.list_images_with_dates(directory)

        if not dated_images:
            logger.warning(f"No dated images found in {directory}")
            return {"target": None, "previous": [], "next": []}

        previous = []
        next_images = []

        for img_path, img_date in dated_images:
            delta = (target_date - img_date).days

            if delta > 0 and delta <= max_time_delta_days:
                previous.append((img_path, img_date, delta))
            elif delta < 0 and abs(delta) <= max_time_delta_days:
                next_images.append((img_path, img_date, abs(delta)))

        previous.sort(key=lambda x: x[2])
        next_images.sort(key=lambda x: x[2])

        previous = [(p[0], p[1]) for p in previous[:n_previous]]
        next_images = [(n[0], n[1]) for n in next_images[:n_next]]

        logger.info(
            f"Found {len(previous)} previous, {len(next_images)} next images "
            f"for target date {target_date.date()}"
        )

        return {
            "target": target_date,
            "previous": previous,
            "next": next_images,
        }

    def get_temporal_images_for_sensor(
        self,
        target_date: str | datetime,
        sensor: str,
        n_previous: int = 1,
        n_next: int = 1,
        max_time_delta_days: int = 365,
    ) -> Dict[str, List[Tuple[Path, datetime]]]:
        """Get temporal images for a specific sensor type.

        Args:
            target_date: Reference date.
            sensor: Sensor type ('liss4', 'sentinel1', 'sentinel2').
            n_previous: Number of previous images to retrieve.
            n_next: Number of next images to retrieve.
            max_time_delta_days: Maximum time difference in days.

        Returns:
            Dictionary with temporal image results.
        """
        sensor_dirs = {
            "liss4": "data/liss4",
            "sentinel1": "data/sentinel1",
            "sentinel2": "data/sentinel2",
        }

        if sensor not in sensor_dirs:
            raise ValueError(
                f"Unknown sensor: {sensor}. Use: {list(sensor_dirs.keys())}"
            )

        directory = Path(sensor_dirs[sensor])
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return {"target": None, "previous": [], "next": []}

        return self.get_temporal_images(
            target_date=target_date,
            directory=directory,
            n_previous=n_previous,
            n_next=n_next,
            max_time_delta_days=max_time_delta_days,
        )

    def organize_temporal(
        self,
        images: List[Tuple[Path, datetime]],
        target_date: str | datetime,
    ) -> Dict[str, Path]:
        """Organize temporal images into directory structure.

        Creates:
            temporal/
                YYYY-MM-DD/
                    previous/
                    next/

        Args:
            images: List of (file_path, datetime) tuples.
            target_date: Target reference date.

        Returns:
            Dictionary mapping image roles to their new paths.
        """
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, "%Y-%m-%d")

        target_dir = self.temporal_dir / target_date.strftime("%Y-%m-%d")
        previous_dir = target_dir / "previous"
        next_dir = target_dir / "next"

        previous_dir.mkdir(parents=True, exist_ok=True)
        next_dir.mkdir(parents=True, exist_ok=True)

        organized = {}
        for img_path, img_date in images:
            import shutil
            delta = (target_date - img_date).days

            if delta > 0:
                dest = previous_dir / img_path.name
            else:
                dest = next_dir / img_path.name

            shutil.copy2(img_path, dest)
            organized[f"{'previous' if delta > 0 else 'next'}_{img_date.date()}"] = dest
            logger.info(f"Copied {img_path.name} to {dest}")

        return organized

    def get_date_range(
        self,
        directory: str | Path,
    ) -> Optional[Tuple[datetime, datetime]]:
        """Get the date range of images in a directory.

        Args:
            directory: Directory containing satellite images.

        Returns:
            Tuple of (earliest_date, latest_date) or None if no dates found.
        """
        dated_images = self.list_images_with_dates(directory)
        if not dated_images:
            return None

        dates = [d[1] for d in dated_images]
        return (min(dates), max(dates))
