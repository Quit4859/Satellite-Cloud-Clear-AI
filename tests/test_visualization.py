"""Tests for visualization module (Phase 12)."""

from pathlib import Path

import numpy as np
import pytest

from utils.visualization import (
    _normalize,
    _to_uint8,
    create_before_after,
    create_cloud_mask_overlay,
    create_difference_image,
    create_false_color,
    create_histogram_comparison,
    create_rgb_preview,
    create_temporal_timeline,
    export_geotiff,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image():
    return np.random.rand(6, 64, 64).astype(np.float32)


@pytest.fixture
def sample_mask():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_normalize(self):
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = _normalize(arr)
        assert out[0] == pytest.approx(0.0, abs=0.01)
        assert out[-1] == pytest.approx(1.0, abs=0.01)

    def test_normalize_constant(self):
        arr = np.ones((16, 16), dtype=np.float32) * 5.0
        out = _normalize(arr)
        assert out.sum() == 0.0

    def test_to_uint8(self):
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = _to_uint8(arr)
        assert out.dtype == np.uint8
        assert out[-1] == 255


# ---------------------------------------------------------------------------
# RGB Preview
# ---------------------------------------------------------------------------

class TestRGBPreview:
    def test_returns_uint8(self, sample_image, tmp_path):
        out = create_rgb_preview(sample_image, output_path=tmp_path / "rgb.png")
        assert out.dtype == np.uint8
        assert out.shape == (64, 64, 3)
        assert (tmp_path / "rgb.png").exists()

    def test_custom_bands(self, sample_image):
        out = create_rgb_preview(sample_image, bands=(5, 3, 1))
        assert out.shape == (64, 64, 3)

    def test_no_save(self, sample_image):
        out = create_rgb_preview(sample_image)
        assert out.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# False Color
# ---------------------------------------------------------------------------

class TestFalseColor:
    def test_returns_uint8(self, sample_image, tmp_path):
        out = create_false_color(sample_image, output_path=tmp_path / "fc.png")
        assert out.dtype == np.uint8
        assert (tmp_path / "fc.png").exists()

    def test_fewer_bands(self):
        img = np.random.rand(3, 32, 32).astype(np.float32)
        out = create_false_color(img, bands=(2, 1, 0))
        assert out.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# Cloud mask overlay
# ---------------------------------------------------------------------------

class TestCloudOverlay:
    def test_overlay_rgb(self, sample_image, sample_mask, tmp_path):
        out = create_cloud_mask_overlay(sample_image, sample_mask,
                                        output_path=tmp_path / "overlay.png")
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8
        assert (tmp_path / "overlay.png").exists()

    def test_overlay_grayscale(self, sample_mask, tmp_path):
        gray = np.random.rand(64, 64).astype(np.float32)
        out = create_cloud_mask_overlay(gray, sample_mask)
        assert out.shape == (64, 64, 3)

    def test_overlay_single_band_3d(self):
        img = np.random.rand(1, 32, 32).astype(np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[5:10, 5:10] = True
        out = create_cloud_mask_overlay(img, mask)
        assert out.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# Before / After
# ---------------------------------------------------------------------------

class TestBeforeAfter:
    def test_2d(self):
        before = np.random.rand(32, 32).astype(np.float32)
        after = np.random.rand(32, 32).astype(np.float32)
        out = create_before_after(before, after, output_path=None)
        assert out.ndim == 3
        assert out.shape[0] == 32

    def test_3d(self, sample_image, tmp_path):
        after = sample_image + 0.01
        out = create_before_after(sample_image, after, output_path=tmp_path / "ba.png")
        assert (tmp_path / "ba.png").exists()


# ---------------------------------------------------------------------------
# Difference image
# ---------------------------------------------------------------------------

class TestDifferenceImage:
    def test_heatmap(self, sample_image, tmp_path):
        rec = sample_image + 0.1
        out = create_difference_image(sample_image, rec, output_path=tmp_path / "diff.png")
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_2d_input(self):
        ref = np.random.rand(32, 32).astype(np.float32)
        rec = ref + 0.05
        out = create_difference_image(ref, rec)
        assert out.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

class TestHistogram:
    def test_creates_file(self, sample_image, tmp_path):
        rec = sample_image + 0.02
        create_histogram_comparison(sample_image, rec, output_path=tmp_path / "hist.png")
        assert (tmp_path / "hist.png").exists()

    def test_2d_input(self, tmp_path):
        a = np.random.rand(32, 32).astype(np.float32)
        b = a + 0.1
        create_histogram_comparison(a, b, output_path=tmp_path / "h.png")
        assert (tmp_path / "h.png").exists()


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

class TestTimeline:
    def test_creates_file(self, tmp_path):
        imgs = [np.random.rand(32, 32).astype(np.float32) for _ in range(4)]
        labels = ["2024-01", "2024-02", "2024-03", "2024-04"]
        create_temporal_timeline(imgs, labels, output_path=tmp_path / "timeline.png")
        assert (tmp_path / "timeline.png").exists()

    def test_single_image(self, tmp_path):
        create_temporal_timeline(
            [np.random.rand(16, 16).astype(np.float32)],
            ["single"],
            output_path=tmp_path / "one.png",
        )
        assert (tmp_path / "one.png").exists()


# ---------------------------------------------------------------------------
# GeoTIFF export
# ---------------------------------------------------------------------------

class TestExportGeoTIFF:
    def test_export(self, tmp_path):
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine

        data = np.random.rand(3, 32, 32).astype(np.float32)
        out = export_geotiff(data, tmp_path / "out.tif",
                             Affine.identity(), CRS.from_epsg(4326))
        assert out.exists()
        with rasterio.open(out) as src:
            assert src.count == 3
            assert src.crs == CRS.from_epsg(4326)

    def test_single_band(self, tmp_path):
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine

        data = np.random.rand(32, 32).astype(np.float32)
        out = export_geotiff(data, tmp_path / "single.tif",
                             Affine.identity(), CRS.from_epsg(4326))
        with rasterio.open(out) as src:
            assert src.count == 1
