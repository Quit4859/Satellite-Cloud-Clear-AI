"""Tests for the satellite data pipeline module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from data.loaders.base import BaseLoader
from data.loaders.liss4 import LISS4Loader
from data.loaders.sentinel1 import Sentinel1Loader
from data.loaders.sentinel2 import Sentinel2Loader
from data.processors.normalizer import Normalizer
from data.processors.resizer import Resizer
from data.pipeline import SatelliteDataPipeline


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
def sample_geotiff_single_band(tmp_path):
    """Create a single-band GeoTIFF file for testing."""
    file_path = tmp_path / "test_single.tif"
    data = np.random.rand(1, 32, 32).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, 32, 32)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return file_path


class TestNormalizer:
    """Tests for the Normalizer class."""

    def test_min_max_normalization(self):
        data = np.array([[[0, 50, 100]]], dtype=np.float32)
        normalizer = Normalizer(method="min_max")
        result = normalizer.normalize(data)

        assert result.min() == 0.0
        assert result.max() == 1.0
        assert result.shape == data.shape

    def test_z_score_normalization(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        normalizer = Normalizer(method="z_score")
        result = normalizer.normalize(data)

        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1

    def test_percentile_normalization(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        data[0, 0, 0] = 1000  # Add outlier
        normalizer = Normalizer(method="percentile")
        result = normalizer.normalize(data)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_clip_outliers_single_percentile(self):
        data = np.array([[[1, 2, 3, 4, 100]]], dtype=np.float32)
        normalizer_clip = Normalizer(method="min_max", clip_percentile=10.0)
        result_clipped = normalizer_clip.normalize(data)

        normalizer_no_clip = Normalizer(method="min_max", clip_percentile=None)
        result_no_clip = normalizer_no_clip.normalize(data)

        assert result_clipped.shape == result_no_clip.shape
        # Clipped version should reduce outlier influence
        assert result_clipped.max() <= 1.0

    def test_clip_outliers_tuple_percentile(self):
        data = np.array([[[1, 2, 3, 4, 100]]], dtype=np.float32)
        normalizer = Normalizer(method="min_max", clip_percentile=(5, 95))
        result = normalizer.normalize(data)

        assert result.max() <= 1.0

    def test_custom_target_range(self):
        data = np.array([[[0, 50, 100]]], dtype=np.float32)
        normalizer = Normalizer(method="min_max", target_range=(-1.0, 1.0))
        result = normalizer.normalize(data)

        assert result.min() == -1.0
        assert result.max() == 1.0

    def test_zero_range_data(self):
        data = np.ones((3, 64, 64), dtype=np.float32)
        normalizer = Normalizer(method="min_max")
        result = normalizer.normalize(data)

        assert result.shape == data.shape

    def test_invalid_method(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        normalizer = Normalizer(method="invalid")

        with pytest.raises(ValueError, match="Unknown method"):
            normalizer.normalize(data)

    def test_fit_transform(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        normalizer = Normalizer(method="z_score")
        normalizer.fit(data)
        result = normalizer.transform(data)

        assert result.shape == data.shape

    def test_transform_without_fit(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        normalizer = Normalizer(method="z_score")

        with pytest.raises(RuntimeError, match="not fitted"):
            normalizer.transform(data)


class TestResizer:
    """Tests for the Resizer class."""

    def test_basic_resize(self):
        data = np.random.rand(3, 128, 128).astype(np.float32)
        resizer = Resizer(target_size=(64, 64))
        result = resizer.resize(data)

        assert result.shape == (3, 64, 64)

    def test_resize_with_aspect_ratio(self):
        data = np.random.rand(3, 100, 200).astype(np.float32)
        resizer = Resizer(target_size=(64, 64), keep_aspect_ratio=True)
        result = resizer.resize(data)

        assert result.shape == (3, 64, 64)
        # Non-zero pixels should be within the original aspect ratio
        assert result.sum() > 0

    def test_resize_to_multiple(self):
        data = np.random.rand(3, 100, 100).astype(np.float32)
        resizer = Resizer(target_size=(64, 64))
        result = resizer.resize_to_multiple(data, multiple=16)

        _, h, w = result.shape
        assert h % 16 == 0
        assert w % 16 == 0

    def test_resize_to_multiple_no_change(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        resizer = Resizer(target_size=(64, 64))
        result = resizer.resize_to_multiple(data, multiple=16)

        np.testing.assert_array_equal(result, data)

    def test_resize_preserves_target_size(self):
        """Test that resize_to_multiple doesn't permanently change target_size."""
        data = np.random.rand(3, 100, 100).astype(np.float32)
        resizer = Resizer(target_size=(64, 64))
        _ = resizer.resize_to_multiple(data, multiple=16)

        assert resizer.target_size == (64, 64)

    def test_invalid_interpolation(self):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        resizer = Resizer(target_size=(32, 32), interpolation="invalid")

        # Should fallback to bilinear
        result = resizer.resize(data)
        assert result.shape == (3, 32, 32)


class TestBaseLoader:
    """Tests for BaseLoader functionality via concrete subclass."""

    def test_context_manager(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            assert loader._dataset is not None

        assert loader._dataset is None

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader = LISS4Loader(tmp_path / "nonexistent.tif")
            loader.open()

    def test_metadata_extraction(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            metadata = loader.metadata

            assert "width" in metadata
            assert "height" in metadata
            assert "count" in metadata
            assert "crs" in metadata
            assert "transform" in metadata

    def test_shape_property(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            assert loader.shape == (4, 64, 64)

    def test_read_all_bands(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            data = loader.read()

            assert data.shape == (4, 64, 64)
            assert data.dtype == np.float32

    def test_read_specific_bands(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            data = loader.read(bands=["B2", "B3"])

            assert data.shape[0] == 2

    def test_read_as_dict(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            data_dict = loader.read_as_dict()

            assert isinstance(data_dict, dict)
            assert len(data_dict) == 4


class TestLISS4Loader:
    """Tests for the LISS-IV loader."""

    def test_get_bands(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            bands = loader.get_bands()

            assert len(bands) == 4
            assert "B2" in bands
            assert "B3" in bands
            assert "B4" in bands
            assert "B5" in bands

    def test_get_default_bands(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            default_bands = loader.get_default_bands()

            assert len(default_bands) == 4

    def test_get_rgb_bands(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            rgb_bands = loader.get_rgb_bands()

            assert rgb_bands == ["B4", "B3", "B2"]

    def test_sensor_name(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            assert loader.sensor_name == "LISS-IV"

    def test_wavelength_info(self, sample_geotiff):
        with LISS4Loader(sample_geotiff) as loader:
            info = loader.get_wavelength_info()

            assert "B2" in info
            assert "B5" in info


class TestSentinel1Loader:
    """Tests for the Sentinel-1 loader."""

    def test_get_bands(self, sample_geotiff):
        with Sentinel1Loader(sample_geotiff) as loader:
            bands = loader.get_bands()

            assert len(bands) == 2
            assert "VV" in bands
            assert "VH" in bands

    def test_sensor_name(self, sample_geotiff):
        with Sentinel1Loader(sample_geotiff) as loader:
            assert loader.sensor_name == "Sentinel-1"

    def test_to_db_conversion(self, sample_geotiff):
        with Sentinel1Loader(sample_geotiff) as loader:
            linear_data = np.array([[[1.0, 10.0, 100.0]]], dtype=np.float32)
            db_data = loader.to_db(linear_data)

            assert db_data.shape == linear_data.shape
            assert db_data[0, 0, 0] == pytest.approx(0.0, abs=0.1)

    def test_to_linear_conversion(self, sample_geotiff):
        with Sentinel1Loader(sample_geotiff) as loader:
            db_data = np.array([[[0.0, 10.0, 20.0]]], dtype=np.float32)
            linear_data = loader.to_linear(db_data)

            assert linear_data.shape == db_data.shape
            assert linear_data[0, 0, 0] == pytest.approx(1.0, abs=0.1)

    def test_polarization_info(self, sample_geotiff):
        with Sentinel1Loader(sample_geotiff) as loader:
            info = loader.get_polarization_info()

            assert "VV" in info
            assert "VH" in info


class TestSentinel2Loader:
    """Tests for the Sentinel-2 loader."""

    def test_get_bands(self, sample_geotiff):
        with Sentinel2Loader(sample_geotiff) as loader:
            bands = loader.get_bands()

            assert len(bands) == 13
            assert "B01" in bands
            assert "B12" in bands

    def test_sensor_name(self, sample_geotiff):
        with Sentinel2Loader(sample_geotiff) as loader:
            assert loader.sensor_name == "Sentinel-2"

    def test_get_rgb_bands(self, sample_geotiff):
        with Sentinel2Loader(sample_geotiff) as loader:
            rgb_bands = loader.get_rgb_bands()

            assert rgb_bands == ["B04", "B03", "B02"]

    def test_resolution_groups(self, sample_geotiff):
        with Sentinel2Loader(sample_geotiff) as loader:
            bands_10m = loader.get_10m_bands()
            bands_20m = loader.get_20m_bands()
            bands_60m = loader.get_60m_bands()

            assert len(bands_10m) == 4
            assert len(bands_20m) == 6
            assert len(bands_60m) == 3

    def test_wavelength_info(self, sample_geotiff):
        with Sentinel2Loader(sample_geotiff) as loader:
            info = loader.get_wavelength_info()

            assert len(info) == 13
            assert "B01" in info


class TestSatelliteDataPipeline:
    """Tests for the SatelliteDataPipeline class."""

    def test_init_invalid_sensor(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown sensor"):
            SatelliteDataPipeline(sensor_type="invalid", output_dir=tmp_path)

    def test_init_creates_output_dir(self, tmp_path):
        output_dir = tmp_path / "output"
        pipeline = SatelliteDataPipeline(
            sensor_type="liss4",
            output_dir=output_dir,
        )

        assert output_dir.exists()

    def test_process_file(self, sample_geotiff, tmp_path):
        pipeline = SatelliteDataPipeline(
            sensor_type="liss4",
            output_dir=tmp_path / "output",
            target_size=(32, 32),
        )

        result = pipeline.process_file(sample_geotiff, save=False)

        assert "data" in result
        assert "transform" in result
        assert "metadata" in result
        assert result["data"].shape[0] == 4  # bands
        assert result["data"].shape[1] == 32  # height
        assert result["data"].shape[2] == 32  # width

    def test_process_file_with_save(self, sample_geotiff, tmp_path):
        output_dir = tmp_path / "output"
        pipeline = SatelliteDataPipeline(
            sensor_type="liss4",
            output_dir=output_dir,
            target_size=(32, 32),
        )

        result = pipeline.process_file(sample_geotiff, save=True)

        saved_files = list(output_dir.glob("*.tif"))
        assert len(saved_files) == 1

    def test_process_batch(self, sample_geotiff, tmp_path):
        pipeline = SatelliteDataPipeline(
            sensor_type="liss4",
            output_dir=tmp_path / "output",
            target_size=(32, 32),
        )

        results = pipeline.process_batch(
            [sample_geotiff, sample_geotiff],
            save=False,
        )

        assert len(results) == 2
        assert all(r is not None for r in results)

    def test_normalize_only(self, sample_geotiff, tmp_path):
        pipeline = SatelliteDataPipeline(
            sensor_type="liss4",
            output_dir=tmp_path / "output",
        )

        with LISS4Loader(sample_geotiff) as loader:
            data = loader.read()

        normalized = pipeline.normalize_only(data)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_resize_only(self, sample_geotiff, tmp_path):
        pipeline = SatelliteDataPipeline(
            sensor_type="liss4",
            output_dir=tmp_path / "output",
            target_size=(32, 32),
        )

        with LISS4Loader(sample_geotiff) as loader:
            data = loader.read()

        resized = pipeline.resize_only(data)

        assert resized.shape == (4, 32, 32)

    def test_get_loader(self, sample_geotiff, tmp_path):
        pipeline = SatelliteDataPipeline(
            sensor_type="liss4",
            output_dir=tmp_path / "output",
        )

        loader = pipeline.get_loader(sample_geotiff)

        assert isinstance(loader, LISS4Loader)

    def test_custom_output_name(self, sample_geotiff, tmp_path):
        output_dir = tmp_path / "output"
        pipeline = SatelliteDataPipeline(
            sensor_type="liss4",
            output_dir=output_dir,
            target_size=(32, 32),
        )

        pipeline.process_file(
            sample_geotiff,
            save=True,
            output_name="custom_name",
        )

        assert (output_dir / "custom_name.tif").exists()
