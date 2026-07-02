"""Tests for the satellite data acquisition module."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from data.acquisition.liss4 import LISS4Acquisition
from data.acquisition.sentinel1 import Sentinel1Acquisition
from data.acquisition.sentinel2 import Sentinel2Acquisition
from data.acquisition.temporal import TemporalSelector
from data.acquisition.acquirer import (
    SatelliteAcquirer,
    download_liss4,
    get_temporal_images,
)


@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a sample GeoTIFF file for testing."""
    file_path = tmp_path / "test_image.tif"
    data = np.random.rand(4, 64, 64).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, 64, 64)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return file_path


@pytest.fixture
def sample_geotiff_with_date(tmp_path):
    """Create a dated GeoTIFF file for temporal testing."""
    file_path = tmp_path / "S2_2024-01-15.tif"
    data = np.random.rand(4, 64, 64).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, 64, 64)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return file_path


@pytest.fixture
def liss4_dir(tmp_path):
    """Create a directory with multiple LISS-IV images."""
    liss4_dir = tmp_path / "liss4"
    liss4_dir.mkdir()

    dates = ["2024-01-01", "2024-02-15", "2024-03-20", "2024-06-01"]
    for date_str in dates:
        file_path = liss4_dir / f"LISS4_{date_str}.tif"
        data = np.random.rand(4, 32, 32).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, 32, 32)

        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=32,
            width=32,
            count=4,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

    return liss4_dir


class TestLISS4Acquisition:
    """Tests for LISS-IV acquisition."""

    def test_init_creates_directory(self, tmp_path):
        data_dir = tmp_path / "liss4_new"
        acquirer = LISS4Acquisition(data_dir=data_dir)
        assert data_dir.exists()

    def test_load_image(self, sample_geotiff):
        acquirer = LISS4Acquisition()
        result = acquirer.load_image(sample_geotiff)

        assert "data" in result
        assert "metadata" in result
        assert "transform" in result
        assert "bounds" in result
        assert "crs" in result
        assert result["data"].shape == (4, 64, 64)
        assert result["metadata"]["sensor"] == "LISS-IV"

    def test_load_image_not_found(self, tmp_path):
        acquirer = LISS4Acquisition()
        with pytest.raises(FileNotFoundError):
            acquirer.load_image(tmp_path / "nonexistent.tif")

    def test_load_images(self, sample_geotiff):
        acquirer = LISS4Acquisition()
        results = acquirer.load_images([sample_geotiff, sample_geotiff])
        assert len(results) == 2
        assert all(r is not None for r in results)

    def test_copy_to_data_dir(self, sample_geotiff, tmp_path):
        data_dir = tmp_path / "liss4_copy"
        data_dir.mkdir()
        acquirer = LISS4Acquisition(data_dir=data_dir)

        dest = acquirer.copy_to_data_dir(sample_geotiff, new_name="custom_name")
        assert dest.exists()
        assert dest.name == "custom_name.tif"

    def test_copy_to_data_dir_default_name(self, sample_geotiff, tmp_path):
        data_dir = tmp_path / "liss4_copy2"
        data_dir.mkdir()
        acquirer = LISS4Acquisition(data_dir=data_dir)

        dest = acquirer.copy_to_data_dir(sample_geotiff)
        assert dest.exists()
        assert dest.name == sample_geotiff.name

    def test_list_images(self, liss4_dir):
        acquirer = LISS4Acquisition(data_dir=liss4_dir)
        images = acquirer.list_images()
        assert len(images) == 4

    def test_get_metadata(self, sample_geotiff):
        acquirer = LISS4Acquisition()
        metadata = acquirer.get_metadata(sample_geotiff)

        assert metadata["sensor"] == "LISS-IV"
        assert metadata["count"] == 4
        assert "bounds" in metadata
        assert "crs" in metadata

    def test_validate_image_valid(self, sample_geotiff):
        acquirer = LISS4Acquisition()
        assert acquirer.validate_image(sample_geotiff) is True

    def test_validate_image_not_geotiff(self, tmp_path):
        fake_file = tmp_path / "fake.txt"
        fake_file.write_text("not an image")
        acquirer = LISS4Acquisition()
        assert acquirer.validate_image(fake_file) is False


class TestTemporalSelector:
    """Tests for temporal image selection."""

    def test_extract_date_yyyy_mm_dd(self):
        selector = TemporalSelector()
        date = selector.extract_date_from_filename("S2_2024-01-15.tif")
        assert date == datetime(2024, 1, 15)

    def test_extract_date_yyyymmdd(self):
        selector = TemporalSelector()
        date = selector.extract_date_from_filename("image_20240115.tif")
        assert date == datetime(2024, 1, 15)

    def test_extract_date_no_date(self):
        selector = TemporalSelector()
        date = selector.extract_date_from_filename("image.tif")
        assert date is None

    def test_list_images_with_dates(self, liss4_dir):
        selector = TemporalSelector()
        dated_images = selector.list_images_with_dates(liss4_dir)
        assert len(dated_images) == 4
        # Check sorted by date
        dates = [d[1] for d in dated_images]
        assert dates == sorted(dates)

    def test_get_temporal_images(self, liss4_dir):
        selector = TemporalSelector()
        result = selector.get_temporal_images(
            target_date="2024-03-01",
            directory=liss4_dir,
            n_previous=2,
            n_next=2,
        )

        assert "target" in result
        assert "previous" in result
        assert "next" in result
        assert result["target"] == datetime(2024, 3, 1)
        assert len(result["previous"]) == 2  # 2024-01-01, 2024-02-15
        assert len(result["next"]) == 2  # 2024-03-20, 2024-06-01

    def test_get_temporal_images_no_results(self, liss4_dir):
        selector = TemporalSelector()
        result = selector.get_temporal_images(
            target_date="2025-01-01",  # After all images
            directory=liss4_dir,
            n_previous=4,
            max_time_delta_days=400,  # Extend to include 2024-01-01 (366 days)
        )
        assert len(result["previous"]) == 4
        assert len(result["next"]) == 0

    def test_get_temporal_images_for_sensor(self, liss4_dir, tmp_path):
        selector = TemporalSelector(temporal_dir=tmp_path / "temporal")
        result = selector.get_temporal_images_for_sensor(
            target_date="2024-03-01",
            sensor="liss4",
            n_previous=1,
            n_next=1,
        )
        assert "previous" in result
        assert "next" in result

    def test_get_temporal_images_invalid_sensor(self, tmp_path):
        selector = TemporalSelector(temporal_dir=tmp_path / "temporal")
        with pytest.raises(ValueError, match="Unknown sensor"):
            selector.get_temporal_images_for_sensor(
                target_date="2024-01-01",
                sensor="invalid",
            )

    def test_organize_temporal(self, liss4_dir, tmp_path):
        selector = TemporalSelector(temporal_dir=tmp_path / "temporal")
        dated_images = selector.list_images_with_dates(liss4_dir)

        organized = selector.organize_temporal(
            images=dated_images[:2],
            target_date="2024-03-01",
        )
        assert len(organized) == 2

    def test_get_date_range(self, liss4_dir):
        selector = TemporalSelector()
        date_range = selector.get_date_range(liss4_dir)
        assert date_range is not None
        assert date_range[0] < date_range[1]


class TestSentinel2Acquisition:
    """Tests for Sentinel-2 acquisition (mocked GEE)."""

    def test_init_creates_directory(self, tmp_path):
        data_dir = tmp_path / "sentinel2_new"
        acquirer = Sentinel2Acquisition(data_dir=data_dir)
        assert data_dir.exists()

    def test_list_images(self, tmp_path):
        data_dir = tmp_path / "sentinel2_list"
        data_dir.mkdir()

        # Create dummy file
        file_path = data_dir / "S2_test.tif"
        data = np.random.rand(4, 32, 32).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, 32, 32)
        with rasterio.open(
            file_path, "w", driver="GTiff", height=32, width=32,
            count=4, dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)

        acquirer = Sentinel2Acquisition(data_dir=data_dir)
        images = acquirer.list_images()
        assert len(images) == 1

    def test_get_metadata(self, sample_geotiff):
        acquirer = Sentinel2Acquisition()
        metadata = acquirer.get_metadata(sample_geotiff)
        assert metadata["sensor"] == "Sentinel-2"
        assert "bounds" in metadata


class TestSentinel1Acquisition:
    """Tests for Sentinel-1 acquisition (mocked GEE)."""

    def test_init_creates_directory(self, tmp_path):
        data_dir = tmp_path / "sentinel1_new"
        acquirer = Sentinel1Acquisition(data_dir=data_dir)
        assert data_dir.exists()

    def test_list_images(self, tmp_path):
        data_dir = tmp_path / "sentinel1_list"
        data_dir.mkdir()

        # Create dummy file
        file_path = data_dir / "S1_test.tif"
        data = np.random.rand(2, 32, 32).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, 32, 32)
        with rasterio.open(
            file_path, "w", driver="GTiff", height=32, width=32,
            count=2, dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data)

        acquirer = Sentinel1Acquisition(data_dir=data_dir)
        images = acquirer.list_images()
        assert len(images) == 1

    def test_get_metadata(self, sample_geotiff):
        acquirer = Sentinel1Acquisition()
        metadata = acquirer.get_metadata(sample_geotiff)
        assert metadata["sensor"] == "Sentinel-1"
        assert metadata["resolution"] == 10


class TestSatelliteAcquirer:
    """Tests for the unified SatelliteAcquirer class."""

    def test_init_creates_directories(self, tmp_path):
        base_dir = tmp_path / "satellite_data"
        acquirer = SatelliteAcquirer(base_dir=base_dir)
        assert (base_dir / "liss4").exists()
        assert (base_dir / "sentinel1").exists()
        assert (base_dir / "sentinel2").exists()
        assert (base_dir / "temporal").exists()
        assert (base_dir / "processed").exists()

    def test_load_liss4(self, sample_geotiff, tmp_path):
        acquirer = SatelliteAcquirer(base_dir=tmp_path / "data")
        result = acquirer.load_liss4(sample_geotiff)
        assert "data" in result

    def test_get_temporal_images(self, liss4_dir, tmp_path):
        acquirer = SatelliteAcquirer(base_dir=tmp_path.parent)
        result = acquirer.get_temporal_images(
            target_date="2024-03-01",
            sensor="liss4",
            n_previous=1,
            n_next=1,
        )
        assert "previous" in result
        assert "next" in result


class TestTopLevelFunctions:
    """Tests for top-level acquisition functions."""

    def test_download_liss4(self, sample_geotiff, tmp_path):
        result = download_liss4(
            file_path=sample_geotiff,
            data_dir=tmp_path / "liss4_func",
        )
        assert "data" in result
        assert result["metadata"]["sensor"] == "LISS-IV"

    def test_get_temporal_images(self, liss4_dir, tmp_path):
        result = get_temporal_images(
            target_date="2024-03-01",
            sensor="liss4",
            n_previous=1,
            n_next=1,
            data_dir=tmp_path.parent,
        )
        assert "previous" in result
        assert "next" in result
