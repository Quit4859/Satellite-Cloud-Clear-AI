"""Comprehensive tests for the satellite preprocessing module."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds

from utils.preprocessing import (
    PreprocessingConfig,
    PreprocessingResult,
    SatellitePreprocessor,
    preprocess_folder,
    preprocess_single,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rgb_geotiff(tmp_path: Path) -> Path:
    """3-band RGB GeoTIFF with values in 0-255."""
    path = tmp_path / "rgb.tif"
    data = np.random.randint(0, 256, (3, 128, 128)).astype(np.uint8)
    transform = from_bounds(73.0, 20.0, 74.0, 21.0, 128, 128)
    with rasterio.open(path, "w", driver="GTiff", height=128, width=128,
                       count=3, dtype="uint8", crs="EPSG:4326",
                       transform=transform) as dst:
        dst.write(data)
    return path


@pytest.fixture
def multispectral_geotiff(tmp_path: Path) -> Path:
    """4-band multispectral float32 GeoTIFF."""
    path = tmp_path / "multi.tif"
    data = np.random.rand(4, 64, 64).astype(np.float32) * 10000
    transform = from_bounds(73.0, 20.0, 74.0, 21.0, 64, 64)
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64,
                       count=4, dtype="float32", crs="EPSG:32643",
                       transform=transform) as dst:
        dst.write(data)
    return path


@pytest.fixture
def nodata_geotiff(tmp_path: Path) -> Path:
    """GeoTIFF with explicit NoData = 0."""
    path = tmp_path / "nodata.tif"
    data = np.random.rand(3, 64, 64).astype(np.float32)
    data[0, 10:20, 10:20] = 0.0  # NoData block
    transform = from_bounds(0, 0, 1, 1, 64, 64)
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64,
                       count=3, dtype="float32", crs="EPSG:4326",
                       transform=transform, nodata=0.0) as dst:
        dst.write(data)
    return path


@pytest.fixture
def single_band_geotiff(tmp_path: Path) -> Path:
    """Single-band GeoTIFF."""
    path = tmp_path / "single.tif"
    data = np.random.rand(1, 32, 32).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, 32, 32)
    with rasterio.open(path, "w", driver="GTiff", height=32, width=32,
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=transform) as dst:
        dst.write(data)
    return path


@pytest.fixture
def batch_dir(tmp_path: Path) -> Path:
    """Directory with multiple GeoTIFFs for batch testing."""
    d = tmp_path / "batch_input"
    d.mkdir()
    transform = from_bounds(0, 0, 1, 1, 32, 32)
    for i in range(3):
        path = d / f"image_{i:03d}.tif"
        data = np.random.rand(3, 32, 32).astype(np.float32)
        with rasterio.open(path, "w", driver="GTiff", height=32, width=32,
                           count=3, dtype="float32", crs="EPSG:4326",
                           transform=transform) as dst:
            dst.write(data)
    return d


@pytest.fixture
def preprocessor() -> SatellitePreprocessor:
    return SatellitePreprocessor(PreprocessingConfig(target_size=(64, 64)))


# ---------------------------------------------------------------------------
# PreprocessingConfig
# ---------------------------------------------------------------------------

class TestPreprocessingConfig:
    def test_defaults(self):
        cfg = PreprocessingConfig()
        assert cfg.target_size == (512, 512)
        assert cfg.normalize is True
        assert cfg.clip_percentile == 2.0
        assert cfg.dtype == "float32"
        assert cfg.resampling == Resampling.bilinear
        assert cfg.compress == "lzw"

    def test_custom_values(self):
        cfg = PreprocessingConfig(target_size=(256, 256), normalize=False)
        assert cfg.target_size == (256, 256)
        assert cfg.normalize is False

    def test_invalid_target_size(self):
        with pytest.raises(ValueError, match="must be positive"):
            PreprocessingConfig(target_size=(0, 100))

    def test_invalid_target_size_negative(self):
        with pytest.raises(ValueError, match="must be positive"):
            PreprocessingConfig(target_size=(-1, -1))


# ---------------------------------------------------------------------------
# PreprocessingResult
# ---------------------------------------------------------------------------

class TestPreprocessingResult:
    def test_properties_3d(self):
        r = PreprocessingResult(
            data=np.zeros((3, 64, 128)),
            transform=Affine.identity(),
            crs=CRS.from_epsg(4326),
            metadata={},
            original_shape=(3, 256, 256),
        )
        assert r.bands == 3
        assert r.height == 64
        assert r.width == 128

    def test_properties_2d(self):
        r = PreprocessingResult(
            data=np.zeros((64, 128)),
            transform=Affine.identity(),
            crs=None,
            metadata={},
            original_shape=(64, 256),
        )
        assert r.bands == 1
        assert r.height == 64
        assert r.width == 128


# ---------------------------------------------------------------------------
# SatellitePreprocessor — read
# ---------------------------------------------------------------------------

class TestReadImage:
    def test_read_rgb(self, preprocessor, rgb_geotiff):
        data, meta = preprocessor.read_image(rgb_geotiff)
        assert data.shape == (3, 128, 128)
        assert data.dtype == np.float32
        assert meta["count"] == 3

    def test_read_multispectral(self, preprocessor, multispectral_geotiff):
        data, meta = preprocessor.read_image(multispectral_geotiff)
        assert data.shape == (4, 64, 64)
        assert meta["crs"] == CRS.from_epsg(32643)

    def test_read_single_band(self, preprocessor, single_band_geotiff):
        data, meta = preprocessor.read_image(single_band_geotiff)
        assert data.shape == (1, 32, 32)

    def test_read_preserves_crs(self, preprocessor, rgb_geotiff):
        _, meta = preprocessor.read_image(rgb_geotiff)
        assert meta["crs"] == CRS.from_epsg(4326)

    def test_read_preserves_transform(self, preprocessor, rgb_geotiff):
        _, meta = preprocessor.read_image(rgb_geotiff)
        assert isinstance(meta["transform"], Affine)

    def test_file_not_found(self, preprocessor, tmp_path):
        with pytest.raises(FileNotFoundError):
            preprocessor.read_image(tmp_path / "missing.tif")

    def test_not_geotiff(self, preprocessor, tmp_path):
        bad = tmp_path / "image.png"
        bad.write_bytes(b"not a tiff")
        with pytest.raises(ValueError, match="Not a GeoTIFF"):
            preprocessor.read_image(bad)


# ---------------------------------------------------------------------------
# SatellitePreprocessor — NoData
# ---------------------------------------------------------------------------

class TestHandleNoData:
    def test_nodata_replaced_with_nan(self, preprocessor):
        data = np.array([[[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]]])
        cleaned, mask = preprocessor.handle_nodata(data, nodata_value=0.0)
        assert np.isnan(cleaned[0, 0, 1])
        assert np.isnan(cleaned[0, 1, 0])
        assert not np.isnan(cleaned[0, 0, 0])
        assert mask[0, 0, 0] is True or mask[0, 0, 0] == True
        assert mask[0, 0, 1] is False or mask[0, 0, 1] == False

    def test_no_nodata(self, preprocessor):
        data = np.array([[[1.0, 2.0, 3.0]]])
        cleaned, mask = preprocessor.handle_nodata(data, nodata_value=None)
        assert cleaned.shape == data.shape
        assert mask.all()

    def test_existing_nan(self, preprocessor):
        data = np.array([[[1.0, np.nan, 3.0]]])
        cleaned, mask = preprocessor.handle_nodata(data)
        assert np.isnan(cleaned[0, 0, 1])
        assert not mask[0, 0, 1]

    def test_integer_nodata(self, preprocessor):
        data = np.array([[[255, 100, 200]]], dtype=np.uint8)
        cleaned, mask = preprocessor.handle_nodata(data, nodata_value=255)
        assert mask[0, 0, 0] is False or mask[0, 0, 0] == False


# ---------------------------------------------------------------------------
# SatellitePreprocessor — normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_minmax_range(self, preprocessor):
        data = np.random.rand(3, 64, 64).astype(np.float32) * 1000
        result = preprocessor.normalize_minmax(data)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_minmax_with_nan(self, preprocessor):
        data = np.random.rand(3, 32, 32).astype(np.float32)
        data[0, 0, 0] = np.nan
        result = preprocessor.normalize_minmax(data)
        assert np.isnan(result[0, 0, 0])
        valid = result[~np.isnan(result)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_constant_data(self, preprocessor):
        data = np.ones((2, 16, 16), dtype=np.float32) * 5.0
        result = preprocessor.normalize_minmax(data)
        assert result.shape == data.shape


# ---------------------------------------------------------------------------
# SatellitePreprocessor — clip outliers
# ---------------------------------------------------------------------------

class TestClipOutliers:
    def test_removes_extreme_values(self, preprocessor):
        data = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
        data[0, 0, 0] = 9999
        clipped = preprocessor.clip_outliers(data, percentile=5.0)
        assert clipped[0, 0, 0] < 9999

    def test_preserves_shape(self, preprocessor):
        data = np.random.rand(3, 32, 32).astype(np.float32)
        result = preprocessor.clip_outliers(data)
        assert result.shape == data.shape

    def test_clip_with_nan(self, preprocessor):
        data = np.random.rand(2, 16, 16).astype(np.float32)
        data[0, 5, 5] = np.nan
        result = preprocessor.clip_outliers(data, percentile=5.0)
        assert np.isnan(result[0, 5, 5])


# ---------------------------------------------------------------------------
# SatellitePreprocessor — resize
# ---------------------------------------------------------------------------

class TestResize:
    def test_basic_resize(self, preprocessor):
        data = np.random.rand(3, 128, 128).astype(np.float32)
        result = preprocessor.resize_image(data, target_size=(64, 64))
        assert result.shape == (3, 64, 64)

    def test_upscale(self, preprocessor):
        data = np.random.rand(3, 32, 32).astype(np.float32)
        result = preprocessor.resize_image(data, target_size=(128, 128))
        assert result.shape == (3, 128, 128)

    def test_single_band(self, preprocessor):
        data = np.random.rand(1, 64, 64).astype(np.float32)
        result = preprocessor.resize_image(data, target_size=(32, 32))
        assert result.shape == (1, 32, 32)

    def test_resize_preserves_nan(self, preprocessor):
        data = np.random.rand(2, 32, 32).astype(np.float32)
        data[0, 10:20, 10:20] = np.nan
        result = preprocessor.resize_image(data, target_size=(64, 64))
        nan_count = np.isnan(result[0]).sum()
        assert nan_count > 0  # NaN region should still exist

    def test_resize_uses_config_default(self):
        cfg = PreprocessingConfig(target_size=(48, 48))
        proc = SatellitePreprocessor(cfg)
        data = np.random.rand(3, 96, 96).astype(np.float32)
        result = proc.resize_image(data)
        assert result.shape == (3, 48, 48)


# ---------------------------------------------------------------------------
# SatellitePreprocessor — save
# ---------------------------------------------------------------------------

class TestSaveImage:
    def test_save_and_reopen(self, preprocessor, tmp_path):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        out = tmp_path / "out.tif"

        saved = preprocessor.save_image(data, out, transform, CRS.from_epsg(4326))
        assert saved.exists()

        with rasterio.open(saved) as src:
            assert src.count == 3
            assert src.width == 64
            assert src.height == 64
            assert src.crs == CRS.from_epsg(4326)

    def test_save_single_band(self, preprocessor, tmp_path):
        data = np.random.rand(64, 64).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        out = tmp_path / "single_out.tif"

        preprocessor.save_image(data, out, transform, CRS.from_epsg(4326))
        with rasterio.open(out) as src:
            assert src.count == 1

    def test_save_creates_parent_dirs(self, preprocessor, tmp_path):
        data = np.random.rand(2, 16, 16).astype(np.float32)
        transform = Affine.identity()
        out = tmp_path / "sub" / "dir" / "out.tif"

        preprocessor.save_image(data, out, transform, CRS.from_epsg(4326))
        assert out.exists()

    def test_save_metadata_tags(self, preprocessor, tmp_path):
        data = np.random.rand(2, 16, 16).astype(np.float32)
        transform = Affine.identity()
        out = tmp_path / "tagged.tif"
        meta = {"band_descriptions": ["Red", "NIR"]}

        preprocessor.save_image(data, out, transform, CRS.from_epsg(4326), metadata=meta)
        with rasterio.open(out) as src:
            assert src.descriptions[0] == "Red"
            assert src.descriptions[1] == "NIR"


# ---------------------------------------------------------------------------
# SatellitePreprocessor — full pipeline
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_full_pipeline_rgb(self, rgb_geotiff, tmp_path):
        cfg = PreprocessingConfig(target_size=(64, 64))
        proc = SatellitePreprocessor(cfg)
        out = tmp_path / "processed.tif"

        result = proc.preprocess(rgb_geotiff, output_path=out)
        assert result.data.shape == (3, 64, 64)
        assert result.bands == 3
        assert out.exists()

        with rasterio.open(out) as src:
            assert src.count == 3
            assert src.width == 64
            assert src.height == 64

    def test_full_pipeline_multispectral(self, multispectral_geotiff, tmp_path):
        cfg = PreprocessingConfig(target_size=(32, 32))
        proc = SatellitePreprocessor(cfg)
        out = tmp_path / "processed_multi.tif"

        result = proc.preprocess(multispectral_geotiff, output_path=out)
        assert result.data.shape == (4, 32, 32)
        assert result.crs == CRS.from_epsg(32643)

    def test_full_pipeline_no_save(self, rgb_geotiff):
        cfg = PreprocessingConfig(target_size=(48, 48))
        proc = SatellitePreprocessor(cfg)
        result = proc.preprocess(rgb_geotiff)
        assert result.data.shape == (3, 48, 48)

    def test_full_pipeline_select_bands(self, multispectral_geotiff, tmp_path):
        cfg = PreprocessingConfig(target_size=(32, 32))
        proc = SatellitePreprocessor(cfg)
        out = tmp_path / "subset.tif"

        result = proc.preprocess(multispectral_geotiff, output_path=out, bands=[1, 3])
        assert result.data.shape == (2, 32, 32)

    def test_full_pipeline_nodata(self, nodata_geotiff, tmp_path):
        cfg = PreprocessingConfig(target_size=(32, 32))
        proc = SatellitePreprocessor(cfg)
        out = tmp_path / "nodata_processed.tif"

        result = proc.preprocess(nodata_geotiff, output_path=out)
        assert result.data.shape == (3, 32, 32)
        assert out.exists()

    def test_output_normalized(self, rgb_geotiff, tmp_path):
        cfg = PreprocessingConfig(target_size=(32, 32), normalize=True)
        proc = SatellitePreprocessor(cfg)
        result = proc.preprocess(rgb_geotiff)
        valid = result.data[~np.isnan(result.data)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_output_not_normalized(self, rgb_geotiff, tmp_path):
        cfg = PreprocessingConfig(target_size=(32, 32), normalize=False)
        proc = SatellitePreprocessor(cfg)
        result = proc.preprocess(rgb_geotiff)
        # Values should not be in [0,1] range after no normalization
        assert result.data.max() > 1.0 or result.data.min() < 0.0

    def test_crs_preserved(self, multispectral_geotiff, tmp_path):
        cfg = PreprocessingConfig(target_size=(32, 32))
        proc = SatellitePreprocessor(cfg)
        out = tmp_path / "crs_test.tif"
        proc.preprocess(multispectral_geotiff, output_path=out)

        with rasterio.open(out) as src:
            assert src.crs == CRS.from_epsg(32643)

    def test_transform_adjusted_for_resize(self, rgb_geotiff):
        cfg = PreprocessingConfig(target_size=(64, 64))
        proc = SatellitePreprocessor(cfg)
        result = proc.preprocess(rgb_geotiff)
        # Original was 128x128, target is 64x64, so pixel size should double
        assert abs(result.transform.a) == pytest.approx(
            abs(from_bounds(73.0, 20.0, 74.0, 21.0, 128, 128).a) * 2, rel=1e-6
        )


# ---------------------------------------------------------------------------
# SatellitePreprocessor — batch
# ---------------------------------------------------------------------------

class TestPreprocessBatch:
    def test_batch_processes_all(self, batch_dir, tmp_path):
        out = tmp_path / "batch_out"
        cfg = PreprocessingConfig(target_size=(16, 16))
        proc = SatellitePreprocessor(cfg)

        saved = proc.preprocess_batch(batch_dir, out)
        assert len(saved) == 3
        for p in saved:
            assert p.exists()
            assert p.name.startswith("processed_")

    def test_batch_with_no_files(self, tmp_path):
        empty_in = tmp_path / "empty_in"
        empty_in.mkdir()
        out = tmp_path / "empty_out"
        cfg = PreprocessingConfig(target_size=(16, 16))
        proc = SatellitePreprocessor(cfg)

        saved = proc.preprocess_batch(empty_in, out)
        assert saved == []

    def test_batch_select_bands(self, batch_dir, tmp_path):
        out = tmp_path / "batch_bands"
        cfg = PreprocessingConfig(target_size=(16, 16))
        proc = SatellitePreprocessor(cfg)

        saved = proc.preprocess_batch(batch_dir, out, bands=[1, 2])
        assert len(saved) == 3
        with rasterio.open(saved[0]) as src:
            assert src.count == 2


# ---------------------------------------------------------------------------
# get_image_info
# ---------------------------------------------------------------------------

class TestGetImageInfo:
    def test_info_rgb(self, preprocessor, rgb_geotiff):
        info = preprocessor.get_image_info(rgb_geotiff)
        assert info["bands"] == 3
        assert info["width"] == 128
        assert info["height"] == 128
        assert info["crs"] == "EPSG:4326"

    def test_info_multispectral(self, preprocessor, multispectral_geotiff):
        info = preprocessor.get_image_info(multispectral_geotiff)
        assert info["bands"] == 4
        assert "pixel_size" in info
        assert "bounds" in info


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_preprocess_single(self, rgb_geotiff, tmp_path):
        out = tmp_path / "convenience.tif"
        result = preprocess_single(rgb_geotiff, out, target_size=(48, 48))
        assert result.data.shape == (3, 48, 48)
        assert out.exists()

    def test_preprocess_folder(self, batch_dir, tmp_path):
        out = tmp_path / "convenience_batch"
        saved = preprocess_folder(batch_dir, out, target_size=(16, 16))
        assert len(saved) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_nan_band(self, tmp_path):
        path = tmp_path / "all_nan.tif"
        data = np.full((2, 16, 16), np.nan, dtype=np.float32)
        transform = Affine.identity()
        with rasterio.open(path, "w", driver="GTiff", height=16, width=16,
                           count=2, dtype="float32", crs="EPSG:4326",
                           transform=transform) as dst:
            dst.write(data)

        cfg = PreprocessingConfig(target_size=(8, 8))
        proc = SatellitePreprocessor(cfg)
        result = proc.preprocess(path)
        assert result.data.shape == (2, 8, 8)

    def test_very_small_image(self, tmp_path):
        path = tmp_path / "tiny.tif"
        data = np.random.rand(3, 2, 2).astype(np.float32)
        transform = Affine.identity()
        with rasterio.open(path, "w", driver="GTiff", height=2, width=2,
                           count=3, dtype="float32", crs="EPSG:4326",
                           transform=transform) as dst:
            dst.write(data)

        cfg = PreprocessingConfig(target_size=(64, 64))
        proc = SatellitePreprocessor(cfg)
        result = proc.preprocess(path)
        assert result.data.shape == (3, 64, 64)

    def test_non_square_target(self, rgb_geotiff, tmp_path):
        cfg = PreprocessingConfig(target_size=(32, 128))
        proc = SatellitePreprocessor(cfg)
        out = tmp_path / "nonsquare.tif"
        result = proc.preprocess(rgb_geotiff, output_path=out)
        assert result.data.shape == (3, 32, 128)

    def test_different_resampling(self, rgb_geotiff):
        cfg = PreprocessingConfig(target_size=(64, 64), resampling=Resampling.nearest)
        proc = SatellitePreprocessor(cfg)
        result = proc.preprocess(rgb_geotiff)
        assert result.data.shape == (3, 64, 64)

    def test_no_compression(self, rgb_geotiff, tmp_path):
        cfg = PreprocessingConfig(target_size=(64, 64), compress=None)
        proc = SatellitePreprocessor(cfg)
        out = tmp_path / "nocompress.tif"
        proc.preprocess(rgb_geotiff, output_path=out)
        assert out.exists()
