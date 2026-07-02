"""Tests for multi-sensor data fusion (Phase 10)."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds

from utils.fusion import (
    FusionConfig,
    FusionResult,
    MultiSensorFusion,
    SensorLayer,
    SensorType,
)


def _make_geotiff(path, data, crs="EPSG:4326", bounds=(0, 0, 1, 1)):
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    n, h, w = data.shape
    transform = from_bounds(*bounds, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=n, dtype=data.dtype, crs=crs, transform=transform) as dst:
        dst.write(data)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return FusionConfig()


@pytest.fixture
def fusion(config):
    return MultiSensorFusion(config)


@pytest.fixture
def sample_layers():
    shape = (3, 32, 32)
    return [
        SensorLayer(SensorType.LISS_IV, np.random.rand(*shape).astype(np.float32),
                    Affine.identity(), CRS.from_epsg(4326), bands=3),
        SensorLayer(SensorType.SENTINEL2, np.random.rand(*shape).astype(np.float32),
                    Affine.identity(), CRS.from_epsg(4326), bands=3),
    ]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestFusionConfig:
    def test_defaults(self):
        cfg = FusionConfig()
        assert cfg.target_crs == "EPSG:4326"
        assert cfg.resampling == "bilinear"

    def test_invalid_resampling(self):
        with pytest.raises(ValueError, match="resampling"):
            FusionConfig(resampling="invalid")


# ---------------------------------------------------------------------------
# SensorLayer
# ---------------------------------------------------------------------------

class TestSensorLayer:
    def test_auto_bands(self):
        layer = SensorLayer(SensorType.DEM, np.random.rand(1, 16, 16).astype(np.float32),
                            Affine.identity(), CRS.from_epsg(4326))
        assert layer.bands == 1


# ---------------------------------------------------------------------------
# Read / Save
# ---------------------------------------------------------------------------

class TestReadSave:
    def test_read_geotiff(self, fusion, tmp_path):
        data = np.random.rand(3, 32, 32).astype(np.float32)
        path = _make_geotiff(tmp_path / "in.tif", data)
        layer = fusion.read_geotiff(path, SensorType.LISS_IV)
        assert layer.data.shape == (3, 32, 32)
        assert layer.sensor_type == SensorType.LISS_IV

    def test_read_not_found(self, fusion, tmp_path):
        with pytest.raises(FileNotFoundError):
            fusion.read_geotiff(tmp_path / "missing.tif")

    def test_save_and_reopen(self, fusion, tmp_path):
        data = np.random.rand(3, 32, 32).astype(np.float32)
        out = fusion.save_geotiff(data, tmp_path / "out.tif",
                                  Affine.identity(), CRS.from_epsg(4326))
        assert out.exists()
        with rasterio.open(out) as src:
            assert src.count == 3
            assert src.crs == CRS.from_epsg(4326)


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------

class TestReprojectLayer:
    def test_same_crs(self, fusion):
        layer = SensorLayer(SensorType.LISS_IV, np.random.rand(3, 16, 16).astype(np.float32),
                            Affine.identity(), CRS.from_epsg(4326), bands=3)
        out = fusion.reproject_layer(layer)
        assert out.data.shape[0] == 3

    def test_resize_to_grid(self, fusion):
        data = np.random.rand(3, 32, 32).astype(np.float32)
        resized = fusion.resize_to_grid(data, 64, 64)
        assert resized.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# Band matching
# ---------------------------------------------------------------------------

class TestBandMatching:
    def test_same_bands(self, fusion):
        data = np.random.rand(3, 16, 16).astype(np.float32)
        out = fusion.match_bands(data, 3)
        np.testing.assert_array_equal(out, data)

    def test_reduce_bands(self, fusion):
        data = np.random.rand(6, 16, 16).astype(np.float32)
        out = fusion.match_bands(data, 3)
        assert out.shape[0] == 3

    def test_pad_bands(self, fusion):
        data = np.random.rand(2, 16, 16).astype(np.float32)
        out = fusion.match_bands(data, 4)
        assert out.shape[0] == 4
        np.testing.assert_array_equal(out[2], 0)


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------

class TestTemporalConsistency:
    def test_same_size(self, fusion, sample_layers):
        aligned = fusion.enforce_temporal_consistency(sample_layers)
        ref_shape = sample_layers[0].data.shape
        for layer in aligned:
            assert layer.data.shape == ref_shape


# ---------------------------------------------------------------------------
# Fuse
# ---------------------------------------------------------------------------

class TestFuse:
    def test_fuse_two_layers(self, fusion, sample_layers):
        result = fusion.fuse(sample_layers)
        assert isinstance(result, FusionResult)
        assert result.fused_data.shape[0] == 6  # 3 + 3
        assert result.fused_data.shape[1:] == (32, 32)

    def test_fuse_empty_raises(self, fusion):
        with pytest.raises(ValueError, match="At least one"):
            fusion.fuse([])

    def test_single_layer(self, fusion):
        layer = SensorLayer(SensorType.DEM, np.random.rand(1, 16, 16).astype(np.float32),
                            Affine.identity(), CRS.from_epsg(4326), bands=1)
        result = fusion.fuse([layer])
        assert result.fused_data.shape[0] == 1


# ---------------------------------------------------------------------------
# Fuse files
# ---------------------------------------------------------------------------

class TestFuseFiles:
    def test_fuse_files(self, fusion, tmp_path):
        s1 = np.random.rand(2, 32, 32).astype(np.float32)
        s2 = np.random.rand(3, 32, 32).astype(np.float32)
        p1 = _make_geotiff(tmp_path / "s1.tif", s1)
        p2 = _make_geotiff(tmp_path / "s2.tif", s2)

        out = tmp_path / "fused.tif"
        result = fusion.fuse_files(
            [p1, p2],
            sensor_types=[SensorType.SENTINEL1, SensorType.SENTINEL2],
            output_path=out,
        )
        assert out.exists()
        assert result.fused_data.ndim == 3

    def test_fuse_files_no_output(self, fusion, tmp_path):
        p = _make_geotiff(tmp_path / "a.tif", np.random.rand(3, 16, 16).astype(np.float32))
        result = fusion.fuse_files([p])
        assert result.fused_data.shape[0] > 0
