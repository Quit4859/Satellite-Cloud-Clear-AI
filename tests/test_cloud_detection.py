"""Comprehensive tests for the cloud detection module."""

from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_bounds

from utils.cloud_detection import (
    CloudDetectionConfig,
    CloudDetectionResult,
    CloudDetector,
    ThresholdCloudDetector,
    detect_clouds_batch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_cloud_image() -> np.ndarray:
    """3-band float32 image with a synthetic bright cloud region."""
    img = np.random.rand(3, 128, 128).astype(np.float32) * 0.3  # dark ground
    # Bright cloud patch top-left
    img[:, 10:40, 10:40] = 0.95
    return img


@pytest.fixture
def clear_sky_image() -> np.ndarray:
    """3-band float32 image with no clouds — dark with slight variation."""
    return np.random.rand(3, 64, 64).astype(np.float32) * 0.4


@pytest.fixture
def fully_clouded_image() -> np.ndarray:
    """3-band float32 image: bright cloud region with slight dark fringe."""
    img = np.ones((3, 64, 64), dtype=np.float32) * 0.9
    img[:, :2, :] = 0.3  # thin dark strip so Otsu has variance to work with
    return img


@pytest.fixture
def geotiff_with_cloud(tmp_path: Path) -> Path:
    """GeoTIFF with a synthetic cloud patch."""
    path = tmp_path / "scene.tif"
    img = np.random.rand(3, 128, 128).astype(np.float32) * 0.3
    img[:, 10:40, 10:40] = 0.95
    transform = from_bounds(73.0, 20.0, 74.0, 21.0, 128, 128)
    with rasterio.open(
        path, "w", driver="GTiff", height=128, width=128,
        count=3, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(img)
    return path


@pytest.fixture
def geotiff_clear(tmp_path: Path) -> Path:
    """GeoTIFF with no clouds."""
    path = tmp_path / "clear.tif"
    img = np.random.rand(3, 64, 64).astype(np.float32) * 0.2
    transform = from_bounds(0, 0, 1, 1, 64, 64)
    with rasterio.open(
        path, "w", driver="GTiff", height=64, width=64,
        count=3, dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(img)
    return path


@pytest.fixture
def geotiff_batch(tmp_path: Path) -> Path:
    """Directory with multiple GeoTIFFs for batch testing."""
    d = tmp_path / "batch_in"
    d.mkdir()
    transform = from_bounds(0, 0, 1, 1, 32, 32)
    for i in range(4):
        path = d / f"scene_{i:02d}.tif"
        img = np.random.rand(3, 32, 32).astype(np.float32) * 0.3
        # Add a cloud to every other image
        if i % 2 == 0:
            img[:, 5:15, 5:15] = 0.9
        with rasterio.open(
            path, "w", driver="GTiff", height=32, width=32,
            count=3, dtype="float32", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(img)
    return d


@pytest.fixture
def detector() -> ThresholdCloudDetector:
    return ThresholdCloudDetector(CloudDetectionConfig(use_otsu=True))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestCloudDetectionConfig:
    def test_defaults(self):
        cfg = CloudDetectionConfig()
        assert cfg.use_otsu is True
        assert cfg.brightness_threshold is None
        assert cfg.morph_open_kernel == 3
        assert cfg.morph_close_kernel == 7
        assert cfg.min_component_area == 50
        assert cfg.composite_band == "mean"

    def test_custom(self):
        cfg = CloudDetectionConfig(
            use_otsu=False,
            brightness_threshold=200,
            min_component_area=100,
        )
        assert cfg.brightness_threshold == 200
        assert cfg.min_component_area == 100


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class TestCloudDetectionResult:
    def test_shape(self):
        r = CloudDetectionResult(
            cloud_mask=np.zeros((64, 128), dtype=bool),
            coverage=0.0, threshold_used=180.0, component_count=0,
        )
        assert r.shape == (64, 128)


# ---------------------------------------------------------------------------
# _to_brightness
# ---------------------------------------------------------------------------

class TestToBrightness:
    def test_single_band(self):
        det = ThresholdCloudDetector()
        img = np.random.rand(64, 64).astype(np.float32)
        result = det._to_brightness(img)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)

    def test_multiband_mean(self):
        det = ThresholdCloudDetector(CloudDetectionConfig(composite_band="mean"))
        img = np.random.rand(4, 64, 64).astype(np.float32)
        result = det._to_brightness(img)
        assert result.shape == (64, 64)

    def test_multiband_max(self):
        det = ThresholdCloudDetector(CloudDetectionConfig(composite_band="max"))
        img = np.random.rand(3, 64, 64).astype(np.float32)
        result = det._to_brightness(img)
        assert result.shape == (64, 64)

    def test_rgb_max(self):
        det = ThresholdCloudDetector(CloudDetectionConfig(composite_band="rgb_max"))
        img = np.random.rand(3, 32, 32).astype(np.float32)
        result = det._to_brightness(img)
        assert result.shape == (32, 32)

    def test_constant_image_returns_zeros(self):
        det = ThresholdCloudDetector()
        img = np.ones((3, 16, 16), dtype=np.float32) * 0.5
        result = det._to_brightness(img)
        assert result.sum() == 0

    def test_nan_handling(self):
        det = ThresholdCloudDetector()
        img = np.random.rand(3, 16, 16).astype(np.float32)
        img[0, 5, 5] = np.nan
        result = det._to_brightness(img)
        assert not np.any(np.isnan(result.astype(float)))


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_otsu_selects_threshold(self, synthetic_cloud_image):
        det = ThresholdCloudDetector(CloudDetectionConfig(use_otsu=True))
        brightness = det._to_brightness(synthetic_cloud_image)
        mask, thresh = det._threshold(brightness)
        assert mask.dtype == bool
        assert mask.shape == brightness.shape
        assert 0 <= thresh <= 255

    def test_fixed_threshold(self, synthetic_cloud_image):
        det = ThresholdCloudDetector(
            CloudDetectionConfig(use_otsu=False, brightness_threshold=200)
        )
        brightness = det._to_brightness(synthetic_cloud_image)
        mask, thresh = det._threshold(brightness)
        assert thresh == 200

    def test_default_fixed_threshold(self):
        det = ThresholdCloudDetector(CloudDetectionConfig(use_otsu=False))
        brightness = np.zeros((16, 16), dtype=np.uint8)
        _, thresh = det._threshold(brightness)
        assert thresh == 180


# ---------------------------------------------------------------------------
# Morphological operations
# ---------------------------------------------------------------------------

class TestMorphology:
    def test_open_reduces_noise(self):
        det = ThresholdCloudDetector()
        # Mask with single-pixel noise
        mask = np.zeros((64, 64), dtype=bool)
        mask[30, 30] = True  # single pixel
        mask[10:20, 10:20] = True  # solid region
        opened = det._morphological_open(mask)
        # Single pixel should be removed
        assert not opened[30, 30]
        # Solid region should survive
        assert opened[15, 15]

    def test_close_fills_gaps(self):
        det = ThresholdCloudDetector()
        mask = np.ones((64, 64), dtype=bool)
        mask[30:32, 30:32] = False  # small hole
        closed = det._morphological_close(mask)
        # Hole should be filled
        assert closed[30, 30]

    def test_open_kernel_zero_skips(self):
        det = ThresholdCloudDetector(CloudDetectionConfig(morph_open_kernel=0))
        mask = np.zeros((16, 16), dtype=bool)
        mask[8, 8] = True
        result = det._morphological_open(mask)
        np.testing.assert_array_equal(result, mask)

    def test_close_kernel_zero_skips(self):
        det = ThresholdCloudDetector(CloudDetectionConfig(morph_close_kernel=0))
        mask = np.ones((16, 16), dtype=bool)
        result = det._morphological_close(mask)
        np.testing.assert_array_equal(result, mask)


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

class TestConnectedComponents:
    def test_filters_small_components(self):
        det = ThresholdCloudDetector(CloudDetectionConfig(min_component_area=10))
        mask = np.zeros((100, 100), dtype=bool)
        # Large component (should survive)
        mask[10:30, 10:30] = True
        # Tiny component (should be removed)
        mask[60, 60] = True
        mask[70:72, 70:72] = True  # area=4

        filtered, count = det._filter_components(mask)
        assert count == 1  # only the large one
        assert not filtered[60, 60]
        assert filtered[15, 15]

    def test_keeps_all_large(self):
        det = ThresholdCloudDetector(CloudDetectionConfig(min_component_area=5))
        mask = np.zeros((64, 64), dtype=bool)
        mask[5:20, 5:20] = True
        mask[40:55, 40:55] = True
        filtered, count = det._filter_components(mask)
        assert count == 2

    def test_empty_mask(self):
        det = ThresholdCloudDetector()
        mask = np.zeros((32, 32), dtype=bool)
        filtered, count = det._filter_components(mask)
        assert count == 0
        assert not filtered.any()


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_full_coverage(self):
        det = ThresholdCloudDetector()
        mask = np.ones((64, 64), dtype=bool)
        image = np.ones((3, 64, 64), dtype=np.float32)
        cov = det._compute_coverage(mask, image)
        assert cov == pytest.approx(1.0)

    def test_zero_coverage(self):
        det = ThresholdCloudDetector()
        mask = np.zeros((64, 64), dtype=bool)
        image = np.ones((3, 64, 64), dtype=np.float32)
        cov = det._compute_coverage(mask, image)
        assert cov == pytest.approx(0.0)

    def test_nan_pixels_excluded(self):
        det = ThresholdCloudDetector()
        mask = np.ones((4, 4), dtype=bool)
        image = np.ones((3, 4, 4), dtype=np.float32)
        image[:, 0, 0] = np.nan  # 1 valid pixel lost
        cov = det._compute_coverage(mask, image)
        # 15 valid pixels, all cloud
        assert cov == pytest.approx(15.0 / 15.0)

    def test_single_band_nan(self):
        det = ThresholdCloudDetector()
        mask = np.ones((4, 4), dtype=bool)
        image = np.ones((4, 4), dtype=np.float32)
        image[0, 0] = np.nan
        cov = det._compute_coverage(mask, image)
        assert cov == pytest.approx(15.0 / 15.0)

    def test_all_nan_returns_zero(self):
        det = ThresholdCloudDetector()
        mask = np.ones((4, 4), dtype=bool)
        image = np.full((3, 4, 4), np.nan, dtype=np.float32)
        cov = det._compute_coverage(mask, image)
        assert cov == 0.0


# ---------------------------------------------------------------------------
# Full detect pipeline
# ---------------------------------------------------------------------------

class TestDetect:
    def test_cloud_detected(self, synthetic_cloud_image):
        det = ThresholdCloudDetector(CloudDetectionConfig(
            use_otsu=True, min_component_area=10,
        ))
        result = det.detect(synthetic_cloud_image)
        assert result.cloud_mask.shape == (128, 128)
        assert result.coverage > 0.0
        assert result.component_count >= 1
        assert isinstance(result.threshold_used, float)

    def test_clear_sky_low_coverage(self):
        """Fixed-threshold detector on a dark image → near-zero coverage."""
        img = np.random.rand(3, 64, 64).astype(np.float32) * 0.3
        det = ThresholdCloudDetector(CloudDetectionConfig(
            use_otsu=False, brightness_threshold=180, min_component_area=10,
        ))
        result = det.detect(img)
        assert result.coverage < 0.05

    def test_fully_clouded_high_coverage(self, fully_clouded_image):
        det = ThresholdCloudDetector(CloudDetectionConfig(
            use_otsu=True, min_component_area=10,
        ))
        result = det.detect(fully_clouded_image)
        assert result.coverage > 0.5

    def test_metadata_passthrough(self, synthetic_cloud_image):
        det = ThresholdCloudDetector()
        meta = {"file": "test.tif", "crs": "EPSG:4326"}
        result = det.detect(synthetic_cloud_image, metadata=meta)
        assert result.metadata["file"] == "test.tif"

    def test_single_band_input(self):
        img = np.random.rand(64, 64).astype(np.float32)
        img[10:30, 10:30] = 0.95
        det = ThresholdCloudDetector(CloudDetectionConfig(min_component_area=10))
        result = det.detect(img)
        assert result.cloud_mask.shape == (64, 64)


# ---------------------------------------------------------------------------
# detect_file
# ---------------------------------------------------------------------------

class TestDetectFile:
    def test_detect_file_mask_and_png(self, geotiff_with_cloud, tmp_path):
        det = ThresholdCloudDetector(CloudDetectionConfig(min_component_area=10))
        mask_path = tmp_path / "mask.tif"
        png_path = tmp_path / "vis.png"
        result = det.detect_file(geotiff_with_cloud, output_mask=mask_path, output_png=png_path)

        assert result.coverage > 0.0
        assert mask_path.exists()
        assert png_path.exists()

        # Verify mask is valid GeoTIFF
        with rasterio.open(mask_path) as src:
            assert src.count == 1
            assert src.dtypes[0] == "uint8"
            mask_data = src.read(1)
            assert set(np.unique(mask_data)).issubset({0, 1})

        # Verify PNG
        png = cv2.imread(str(png_path))
        assert png is not None
        assert png.shape[2] == 3

    def test_detect_file_no_save(self, geotiff_with_cloud):
        det = ThresholdCloudDetector()
        result = det.detect_file(geotiff_with_cloud)
        assert result.coverage > 0.0

    def test_detect_file_clear(self, geotiff_clear, tmp_path):
        det = ThresholdCloudDetector(CloudDetectionConfig(
            use_otsu=False, brightness_threshold=180, min_component_area=10,
        ))
        result = det.detect_file(geotiff_clear)
        assert result.coverage < 0.1

    def test_file_not_found(self):
        det = ThresholdCloudDetector()
        with pytest.raises(FileNotFoundError):
            det.detect_file("/nonexistent/path.tif")

    def test_preserves_crs_in_mask(self, geotiff_with_cloud, tmp_path):
        det = ThresholdCloudDetector()
        mask_path = tmp_path / "crs_mask.tif"
        det.detect_file(geotiff_with_cloud, output_mask=mask_path)
        with rasterio.open(mask_path) as src:
            assert src.crs is not None


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class TestCloudDetectorInterface:
    def test_is_abstract(self):
        """CloudDetector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CloudDetector()

    def test_custom_detector(self, synthetic_cloud_image):
        """A custom detector subclass works through the same pipeline."""

        class DummyDetector(CloudDetector):
            def detect(self, image, metadata=None):
                h, w = image.shape[-2], image.shape[-1]
                return CloudDetectionResult(
                    cloud_mask=np.zeros((h, w), dtype=bool),
                    coverage=0.0,
                    threshold_used=0.0,
                    component_count=0,
                    metadata=metadata or {},
                )

            def detect_file(self, file_path, output_mask=None, output_png=None):
                with rasterio.open(file_path) as src:
                    img = src.read().astype(np.float32)
                return self.detect(img)

        det = DummyDetector()
        result = det.detect(synthetic_cloud_image)
        assert result.coverage == 0.0
        assert result.cloud_mask.sum() == 0


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestBatch:
    def test_batch_all_processed(self, geotiff_batch, tmp_path):
        out = tmp_path / "batch_out"
        results = detect_clouds_batch(geotiff_batch, out)
        assert len(results) == 4

        # Check outputs exist
        masks = list(out.glob("cloud_mask_*.tif"))
        pngs = list(out.glob("cloud_vis_*.png"))
        assert len(masks) == 4
        assert len(pngs) == 4

    def test_batch_coverage_values(self, geotiff_batch, tmp_path):
        out = tmp_path / "batch_out2"
        results = detect_clouds_batch(geotiff_batch, out)
        for r in results:
            assert 0.0 <= r.coverage <= 1.0

    def test_batch_clouded_images_higher_coverage(self, tmp_path):
        """A bright image should show higher cloud coverage than a dark one."""
        d = tmp_path / "batch_cmp"
        d.mkdir()
        transform = from_bounds(0, 0, 1, 1, 32, 32)

        # Dark "clear" image — small variation so brightness map is non-uniform
        clear_path = d / "clear.tif"
        clear_img = np.ones((3, 32, 32), dtype=np.float32) * 0.1
        clear_img[:, 0:4, 0:4] = 0.15  # tiny bright corner
        with rasterio.open(clear_path, "w", driver="GTiff", height=32, width=32,
                           count=3, dtype="float32", crs="EPSG:4326",
                           transform=transform) as dst:
            dst.write(clear_img)

        # Bright "cloudy" image — also non-uniform
        cloud_path = d / "cloud.tif"
        cloud_img = np.ones((3, 32, 32), dtype=np.float32) * 0.9
        cloud_img[:, 0:2, 0:2] = 0.3  # small dark corner
        with rasterio.open(cloud_path, "w", driver="GTiff", height=32, width=32,
                           count=3, dtype="float32", crs="EPSG:4326",
                           transform=transform) as dst:
            dst.write(cloud_img)

        out = tmp_path / "batch_cmp_out"
        det = ThresholdCloudDetector(CloudDetectionConfig(
            use_otsu=False, brightness_threshold=180, min_component_area=1,
        ))
        results = detect_clouds_batch(d, out, detector=det)
        assert len(results) == 2
        # Cloud image: ~96% bright → higher coverage
        # Clear image: ~4% bright → lower coverage
        assert results[1].coverage > results[0].coverage

    def test_batch_empty_directory(self, tmp_path):
        empty_in = tmp_path / "empty_in"
        empty_in.mkdir()
        out = tmp_path / "empty_out"
        results = detect_clouds_batch(empty_in, out)
        assert results == []

    def test_batch_custom_detector(self, geotiff_batch, tmp_path):
        out = tmp_path / "batch_custom"
        det = ThresholdCloudDetector(CloudDetectionConfig(
            use_otsu=False, brightness_threshold=220, min_component_area=5,
        ))
        results = detect_clouds_batch(geotiff_batch, out, detector=det)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_zero_image(self):
        img = np.zeros((3, 32, 32), dtype=np.float32)
        det = ThresholdCloudDetector()
        result = det.detect(img)
        assert result.cloud_mask.shape == (32, 32)

    def test_all_one_image(self):
        img = np.ones((3, 32, 32), dtype=np.float32)
        det = ThresholdCloudDetector()
        result = det.detect(img)
        # Uniform image — Otsu may split arbitrarily
        assert result.cloud_mask.dtype == bool

    def test_very_small_image(self):
        img = np.random.rand(3, 4, 4).astype(np.float32)
        det = ThresholdCloudDetector(CloudDetectionConfig(min_component_area=1))
        result = det.detect(img)
        assert result.cloud_mask.shape == (4, 4)

    def test_large_kernel(self):
        img = np.random.rand(3, 64, 64).astype(np.float32)
        img[:, 20:40, 20:40] = 0.95
        det = ThresholdCloudDetector(CloudDetectionConfig(
            morph_open_kernel=11, morph_close_kernel=15, min_component_area=10,
        ))
        result = det.detect(img)
        assert result.cloud_mask.shape == (64, 64)

    def test_png_color_values(self, synthetic_cloud_image, tmp_path):
        det = ThresholdCloudDetector(CloudDetectionConfig(min_component_area=10))
        png_path = tmp_path / "colors.png"
        det.detect_file(
            None, output_png=png_path,
        ) if False else det._save_png_overlay(
            det.detect(synthetic_cloud_image).cloud_mask, png_path,
        )
        png = cv2.imread(str(png_path))
        # Only two colors expected: red and green
        # Check unique channels
        r, g, b = png[:, :, 2], png[:, :, 1], png[:, :, 0]
        # Red pixels: R=220, G=40, B=40
        red_mask = (r == 220) & (g == 40) & (b == 40)
        # Green pixels: R=34, G=139, B=34
        green_mask = (r == 34) & (g == 139) & (b == 34)
        assert (red_mask | green_mask).all()
