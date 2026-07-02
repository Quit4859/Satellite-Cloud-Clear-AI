"""Comprehensive tests for the temporal reconstruction engine."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds

from models.temporal_reconstruction import (
    ReconstructionConfig,
    ReconstructionResult,
    ReplacementStats,
    TemporalReconstruction,
    reconstruct_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_geotiff(
    path: Path,
    data: np.ndarray,
    crs: str = "EPSG:4326",
    bounds: tuple = (0.0, 0.0, 1.0, 1.0),
    nodata: float = None,
) -> Path:
    """Write a GeoTIFF with given data."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    n_bands, height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    kwargs = {}
    if nodata is not None:
        kwargs["nodata"] = nodata
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=n_bands, dtype=data.dtype, crs=crs, transform=transform,
        **kwargs,
    ) as dst:
        dst.write(data)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def three_band_images():
    """Three 3-band images with known overlap regions."""
    shape = (3, 64, 64)
    current = np.random.rand(*shape).astype(np.float32) * 0.5
    previous = np.random.rand(*shape).astype(np.float32) * 0.5
    next_img = np.random.rand(*shape).astype(np.float32) * 0.5
    return current, previous, next_img


@pytest.fixture
def cloud_mask_partial():
    """Cloud mask covering ~25% of pixels (top-left quadrant)."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[:32, :32] = True
    return mask


@pytest.fixture
def cloud_mask_full():
    """Cloud mask covering the entire image."""
    return np.ones((64, 64), dtype=bool)


@pytest.fixture
def cloud_mask_empty():
    """No clouds."""
    return np.zeros((64, 64), dtype=bool)


@pytest.fixture
def config():
    return ReconstructionConfig()


@pytest.fixture
def engine(config):
    return TemporalReconstruction(config)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestReconstructionConfig:
    def test_defaults(self):
        cfg = ReconstructionConfig()
        assert cfg.invalid_data_value is None
        assert cfg.max_cloud_fraction == 1.0
        assert cfg.min_valid_fraction == 0.0
        assert cfg.output_dtype == "float32"
        assert cfg.compress == "lzw"

    def test_custom_values(self):
        cfg = ReconstructionConfig(
            invalid_data_value=0.0,
            max_cloud_fraction=0.8,
            min_valid_fraction=0.2,
            output_dtype="float64",
        )
        assert cfg.invalid_data_value == 0.0
        assert cfg.max_cloud_fraction == 0.8
        assert cfg.min_valid_fraction == 0.2
        assert cfg.output_dtype == "float64"

    def test_invalid_cloud_fraction_zero(self):
        with pytest.raises(ValueError, match="max_cloud_fraction"):
            ReconstructionConfig(max_cloud_fraction=0.0)

    def test_invalid_cloud_fraction_above_one(self):
        with pytest.raises(ValueError, match="max_cloud_fraction"):
            ReconstructionConfig(max_cloud_fraction=1.5)

    def test_invalid_valid_fraction_one(self):
        with pytest.raises(ValueError, match="min_valid_fraction"):
            ReconstructionConfig(min_valid_fraction=1.0)

    def test_invalid_valid_fraction_negative(self):
        with pytest.raises(ValueError, match="min_valid_fraction"):
            ReconstructionConfig(min_valid_fraction=-0.1)


# ---------------------------------------------------------------------------
# ReplacementStats
# ---------------------------------------------------------------------------


class TestReplacementStats:
    def test_full_replacement(self):
        s = ReplacementStats(
            total_pixels=100,
            cloudy_pixels=20,
            replaced_from_previous=15,
            replaced_from_next=5,
            unresolved_pixels=0,
        )
        assert s.cloud_fraction == pytest.approx(0.2)
        assert s.replacement_rate == pytest.approx(1.0)

    def test_partial_replacement(self):
        s = ReplacementStats(
            total_pixels=100,
            cloudy_pixels=20,
            replaced_from_previous=5,
            replaced_from_next=5,
            unresolved_pixels=10,
        )
        assert s.cloud_fraction == pytest.approx(0.2)
        assert s.replacement_rate == pytest.approx(0.5)

    def test_no_clouds(self):
        s = ReplacementStats(
            total_pixels=100,
            cloudy_pixels=0,
            replaced_from_previous=0,
            replaced_from_next=0,
            unresolved_pixels=0,
        )
        assert s.cloud_fraction == 0.0
        assert s.replacement_rate == 1.0

    def test_to_dict(self):
        s = ReplacementStats(
            total_pixels=100, cloudy_pixels=10,
            replaced_from_previous=8, replaced_from_next=2,
            unresolved_pixels=0,
        )
        d = s.to_dict()
        assert d["cloud_fraction"] == 0.1
        assert d["replacement_rate"] == 1.0
        assert d["unresolved_pixels"] == 0

    def test_zero_total_pixels(self):
        s = ReplacementStats(
            total_pixels=0, cloudy_pixels=0,
            replaced_from_previous=0, replaced_from_next=0,
            unresolved_pixels=0,
        )
        assert s.cloud_fraction == 0.0
        assert s.replacement_rate == 1.0


# ---------------------------------------------------------------------------
# ReconstructionResult
# ---------------------------------------------------------------------------


class TestReconstructionResult:
    def test_properties(self):
        result = ReconstructionResult(
            reconstructed_image=np.random.rand(3, 64, 64).astype(np.float32),
            unresolved_mask=np.zeros((64, 64), dtype=bool),
            band_stats=[
                ReplacementStats(100, 10, 8, 2, 0),
                ReplacementStats(100, 10, 8, 2, 0),
                ReplacementStats(100, 10, 8, 2, 0),
            ],
        )
        assert result.n_bands == 3
        assert result.total_cloud_fraction == pytest.approx(0.1)
        assert result.overall_replacement_rate == pytest.approx(1.0)
        assert result.total_unresolved == 0

    def test_summary(self):
        result = ReconstructionResult(
            reconstructed_image=np.random.rand(2, 32, 32).astype(np.float32),
            unresolved_mask=np.ones((32, 32), dtype=bool),
            band_stats=[
                ReplacementStats(1024, 512, 200, 100, 212),
                ReplacementStats(1024, 512, 200, 100, 212),
            ],
        )
        s = result.summary()
        assert s["total_unresolved_pixels"] == 1024
        assert len(s["per_band"]) == 2

    def test_empty_stats(self):
        result = ReconstructionResult(
            reconstructed_image=np.random.rand(1, 16, 16).astype(np.float32),
            unresolved_mask=np.zeros((16, 16), dtype=bool),
            band_stats=[],
        )
        assert result.total_cloud_fraction == 0.0
        assert result.overall_replacement_rate == 1.0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


class TestReadGeotiff:
    def test_read_rgb(self, engine, tmp_path):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        path = _make_geotiff(tmp_path / "img.tif", data)
        result, meta = engine.read_geotiff(path)
        assert result.shape == (3, 64, 64)
        assert result.dtype == np.float32
        assert meta["crs"] == CRS.from_epsg(4326)

    def test_read_single_band(self, engine, tmp_path):
        data = np.random.rand(64, 64).astype(np.float32)
        path = _make_geotiff(tmp_path / "single.tif", data)
        result, meta = engine.read_geotiff(path)
        assert result.shape == (1, 64, 64)

    def test_file_not_found(self, engine, tmp_path):
        with pytest.raises(FileNotFoundError):
            engine.read_geotiff(tmp_path / "missing.tif")

    def test_not_geotiff(self, engine, tmp_path):
        bad = tmp_path / "image.png"
        bad.write_bytes(b"not a tiff")
        with pytest.raises(ValueError, match="Not a GeoTIFF"):
            engine.read_geotiff(bad)


class TestSaveGeotiff:
    def test_save_and_reopen(self, engine, tmp_path):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        out = engine.save_geotiff(data, tmp_path / "out.tif", transform, CRS.from_epsg(4326))
        assert out.exists()
        with rasterio.open(out) as src:
            assert src.count == 3
            assert src.width == 64
            assert src.crs == CRS.from_epsg(4326)

    def test_save_single_band(self, engine, tmp_path):
        data = np.random.rand(64, 64).astype(np.float32)
        out = engine.save_geotiff(
            data, tmp_path / "single.tif",
            Affine.identity(), CRS.from_epsg(4326),
        )
        with rasterio.open(out) as src:
            assert src.count == 1

    def test_save_creates_parent_dirs(self, engine, tmp_path):
        data = np.random.rand(2, 16, 16).astype(np.float32)
        out = engine.save_geotiff(
            data, tmp_path / "sub" / "dir" / "out.tif",
            Affine.identity(), CRS.from_epsg(4326),
        )
        assert out.exists()

    def test_save_with_nodata(self, engine, tmp_path):
        data = np.random.rand(2, 16, 16).astype(np.float32)
        out = engine.save_geotiff(
            data, tmp_path / "nodata.tif",
            Affine.identity(), CRS.from_epsg(4326), nodata=-9999.0,
        )
        with rasterio.open(out) as src:
            assert src.nodata == -9999.0

    def test_output_dtype(self, engine, tmp_path):
        cfg = ReconstructionConfig(output_dtype="float64")
        eng = TemporalReconstruction(cfg)
        data = np.random.rand(2, 16, 16).astype(np.float32)
        out = eng.save_geotiff(
            data, tmp_path / "f64.tif",
            Affine.identity(), CRS.from_epsg(4326),
        )
        with rasterio.open(out) as src:
            assert src.dtypes[0] == "float64"


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------


class TestBuildValidMask:
    def test_all_valid(self, engine):
        img = np.random.rand(3, 32, 32).astype(np.float32)
        mask = engine._build_valid_mask(img)
        assert mask.all()

    def test_nan_invalid(self, engine):
        img = np.random.rand(3, 32, 32).astype(np.float32)
        img[0, 5, 5] = np.nan
        mask = engine._build_valid_mask(img)
        assert not mask[5, 5]
        assert mask.sum() == 32 * 32 - 1

    def test_inf_invalid(self, engine):
        img = np.random.rand(3, 32, 32).astype(np.float32)
        img[1, 10, 10] = np.inf
        mask = engine._build_valid_mask(img)
        assert not mask[10, 10]

    def test_nodata_value_invalid(self, engine):
        img = np.random.rand(3, 32, 32).astype(np.float32)
        img[2, 0, 0] = -9999.0
        mask = engine._build_valid_mask(img, nodata_value=-9999.0)
        assert not mask[0, 0]

    def test_single_band_nodata(self, engine):
        img = np.random.rand(1, 32, 32).astype(np.float32)
        img[0, 0, 0] = 0.0
        mask = engine._build_valid_mask(img, nodata_value=0.0)
        assert not mask[0, 0]


# ---------------------------------------------------------------------------
# Core reconstruction
# ---------------------------------------------------------------------------


class TestReconstruct:
    def test_no_clouds_passthrough(self, engine, three_band_images, cloud_mask_empty):
        current, previous, next_img = three_band_images
        result = engine.reconstruct(current, previous, next_img, cloud_mask_empty)
        np.testing.assert_array_equal(result.reconstructed_image, current)
        assert result.total_unresolved == 0
        assert result.overall_replacement_rate == 1.0

    def test_all_clouds_use_previous(self, engine, three_band_images, cloud_mask_full):
        current, previous, next_img = three_band_images
        result = engine.reconstruct(current, previous, next_img, cloud_mask_full)
        np.testing.assert_array_equal(result.reconstructed_image, previous)
        assert result.overall_replacement_rate == 1.0

    def test_partial_cloud_replacement(self, engine, three_band_images, cloud_mask_partial):
        current, previous, next_img = three_band_images
        result = engine.reconstruct(current, previous, next_img, cloud_mask_partial)
        # Cloudy region should match previous
        for b in range(3):
            np.testing.assert_array_equal(
                result.reconstructed_image[b, :32, :32],
                previous[b, :32, :32],
            )
        # Clear region should stay as current
        np.testing.assert_array_equal(
            result.reconstructed_image[0, 40, 40],
            current[0, 40, 40],
        )
        assert result.total_unresolved == 0

    def test_falls_back_to_next(self, engine, three_band_images, cloud_mask_partial):
        current, previous, next_img = three_band_images
        # Make previous invalid in cloudy region
        prev_bad = previous.copy()
        prev_bad[:, :32, :32] = np.nan
        result = engine.reconstruct(current, prev_bad, next_img, cloud_mask_partial)
        # Cloudy region should use next
        for b in range(3):
            np.testing.assert_array_equal(
                result.reconstructed_image[b, :32, :32],
                next_img[b, :32, :32],
            )

    def test_unresolved_when_both_invalid(self, engine, three_band_images, cloud_mask_partial):
        current, previous, next_img = three_band_images
        prev_bad = previous.copy()
        prev_bad[:, :32, :32] = np.nan
        next_bad = next_img.copy()
        next_bad[:, :32, :32] = np.nan
        result = engine.reconstruct(current, prev_bad, next_bad, cloud_mask_partial)
        assert result.total_unresolved == 32 * 32
        assert result.unresolved_mask[:32, :32].all()
        assert not result.unresolved_mask[40, 40]

    def test_single_band_input(self, engine):
        ref = np.random.rand(64, 64).astype(np.float32)
        prev = np.random.rand(64, 64).astype(np.float32)
        nxt = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=bool)
        mask[0:16, 0:16] = True
        result = engine.reconstruct(ref, prev, nxt, mask)
        assert result.reconstructed_image.ndim == 3
        assert result.reconstructed_image.shape[0] == 1

    def test_multispectral(self, engine):
        """7-band image reconstruction."""
        bands = 7
        ref = np.random.rand(bands, 32, 32).astype(np.float32)
        prev = np.random.rand(bands, 32, 32).astype(np.float32)
        nxt = np.random.rand(bands, 32, 32).astype(np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[0:8, 0:8] = True
        result = engine.reconstruct(ref, prev, nxt, mask)
        assert result.reconstructed_image.shape == (7, 32, 32)
        assert len(result.band_stats) == 7

    def test_shape_mismatch_raises(self, engine):
        a = np.random.rand(3, 64, 64).astype(np.float32)
        b = np.random.rand(3, 32, 32).astype(np.float32)
        mask = np.zeros((64, 64), dtype=bool)
        with pytest.raises(ValueError, match="Shape mismatch"):
            engine.reconstruct(a, b, b, mask)

    def test_cloud_mask_shape_mismatch(self, engine):
        a = np.random.rand(3, 64, 64).astype(np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        with pytest.raises(ValueError, match="Cloud mask shape"):
            engine.reconstruct(a, a, a, mask)

    def test_nodata_value_affects_validity(self, engine):
        """Nodata sentinel marks pixels as invalid."""
        shape = (3, 32, 32)
        current = np.random.rand(*shape).astype(np.float32)
        previous = np.full(shape, 42.0, dtype=np.float32)
        # Only a small patch has valid data; rest is nodata
        previous[:, 5:10, 5:10] = np.random.rand(3, 5, 5).astype(np.float32)
        next_img = np.full(shape, 42.0, dtype=np.float32)
        # Cloud mask covers a large region, much larger than the valid patch
        mask = np.zeros((32, 32), dtype=bool)
        mask[:20, :20] = True

        result = engine.reconstruct(
            current, previous, next_img, mask,
            prev_nodata=42.0, next_nodata=42.0,
        )
        # 5:10,5:10 should be replaced from prev (25 pixels)
        # The rest of 0:20,0:20 is cloudy with both prev/next nodata → unresolved
        assert result.total_unresolved > 0
        prev_stat = result.band_stats[0]
        assert prev_stat.replaced_from_previous == 25
        assert prev_stat.unresolved_pixels == 20 * 20 - 25

    def test_statistics_consistency(self, engine, three_band_images, cloud_mask_partial):
        current, previous, next_img = three_band_images
        result = engine.reconstruct(current, previous, next_img, cloud_mask_partial)
        for stat in result.band_stats:
            total = (
                stat.replaced_from_previous
                + stat.replaced_from_next
                + stat.unresolved_pixels
            )
            assert total == stat.cloudy_pixels

    def test_output_is_float32(self, engine, three_band_images, cloud_mask_partial):
        current, previous, next_img = three_band_images
        result = engine.reconstruct(current, previous, next_img, cloud_mask_partial)
        assert result.reconstructed_image.dtype == np.float32


# ---------------------------------------------------------------------------
# File-level reconstruction
# ---------------------------------------------------------------------------


class TestReconstructFromFiles:
    def test_end_to_end(self, engine, tmp_path):
        shape = (3, 64, 64)
        current = np.random.rand(*shape).astype(np.float32)
        previous = np.random.rand(*shape).astype(np.float32)
        next_img = np.random.rand(*shape).astype(np.float32)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True

        cur_path = _make_geotiff(tmp_path / "current.tif", current)
        prev_path = _make_geotiff(tmp_path / "previous.tif", previous)
        next_path = _make_geotiff(tmp_path / "next.tif", next_img)
        mask_path = _make_geotiff(tmp_path / "mask.tif", mask.astype(np.uint8))

        out = tmp_path / "cloud_free.tif"
        result = engine.reconstruct_from_files(
            cur_path, prev_path, next_path, mask_path,
            output_path=out,
        )

        assert out.exists()
        assert result.reconstructed_image.shape == shape
        assert result.metadata["crs"] == CRS.from_epsg(4326)

        with rasterio.open(out) as src:
            assert src.count == 3
            assert src.crs == CRS.from_epsg(4326)

    def test_unresolved_output(self, engine, tmp_path):
        shape = (3, 32, 32)
        current = np.random.rand(*shape).astype(np.float32)
        prev_bad = np.full(shape, np.nan, dtype=np.float32)
        next_bad = np.full(shape, np.nan, dtype=np.float32)
        mask = np.ones((32, 32), dtype=bool)

        cur_path = _make_geotiff(tmp_path / "c.tif", current)
        prev_path = _make_geotiff(tmp_path / "p.tif", prev_bad)
        next_path = _make_geotiff(tmp_path / "n.tif", next_bad)
        mask_path = _make_geotiff(tmp_path / "m.tif", mask.astype(np.uint8))

        unresolved_out = tmp_path / "unresolved.tif"
        result = engine.reconstruct_from_files(
            cur_path, prev_path, next_path, mask_path,
            unresolved_output_path=unresolved_out,
        )

        assert unresolved_out.exists()
        assert result.total_unresolved == 32 * 32

    def test_array_cloud_mask(self, engine, tmp_path):
        shape = (3, 32, 32)
        current = np.random.rand(*shape).astype(np.float32)
        previous = np.random.rand(*shape).astype(np.float32)
        next_img = np.random.rand(*shape).astype(np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[0:8, 0:8] = True

        cur_path = _make_geotiff(tmp_path / "c.tif", current)
        prev_path = _make_geotiff(tmp_path / "p.tif", previous)
        next_path = _make_geotiff(tmp_path / "n.tif", next_img)

        result = engine.reconstruct_from_files(
            cur_path, prev_path, next_path, mask,
        )
        assert result.reconstructed_image.shape == shape

    def test_file_not_found(self, engine, tmp_path):
        cur = _make_geotiff(tmp_path / "c.tif", np.zeros((1, 8, 8), dtype=np.float32))
        prev = _make_geotiff(tmp_path / "p.tif", np.zeros((1, 8, 8), dtype=np.float32))
        nxt = _make_geotiff(tmp_path / "n.tif", np.zeros((1, 8, 8), dtype=np.float32))
        mask = _make_geotiff(tmp_path / "m.tif", np.zeros((8, 8), dtype=np.uint8))

        with pytest.raises(FileNotFoundError):
            engine.reconstruct_from_files(cur, tmp_path / "missing.tif", nxt, mask)

    def test_nodata_propagation(self, engine, tmp_path):
        """CRS and transform are preserved in the output."""
        shape = (3, 32, 32)
        bounds = (73.0, 20.0, 74.0, 21.0)
        current = np.random.rand(*shape).astype(np.float32)
        previous = np.random.rand(*shape).astype(np.float32)
        next_img = np.random.rand(*shape).astype(np.float32)
        mask = np.zeros((32, 32), dtype=bool)

        cur_path = _make_geotiff(tmp_path / "c.tif", current, bounds=bounds, crs="EPSG:32643")
        prev_path = _make_geotiff(tmp_path / "p.tif", previous, bounds=bounds, crs="EPSG:32643")
        next_path = _make_geotiff(tmp_path / "n.tif", next_img, bounds=bounds, crs="EPSG:32643")
        mask_path = _make_geotiff(tmp_path / "m.tif", mask.astype(np.uint8), bounds=bounds)

        out = tmp_path / "out.tif"
        engine.reconstruct_from_files(
            cur_path, prev_path, next_path, mask_path, output_path=out,
        )
        with rasterio.open(out) as src:
            assert src.crs == CRS.from_epsg(32643)
            assert isinstance(src.transform, Affine)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


class TestReconstructBatch:
    def _make_batch_dir(self, tmp_path: Path, n: int = 3):
        """Create a batch directory with n triplets."""
        d = tmp_path / "batch"
        d.mkdir()
        shape = (3, 32, 32)
        for i in range(n):
            current = np.random.rand(*shape).astype(np.float32)
            previous = np.random.rand(*shape).astype(np.float32)
            next_img = np.random.rand(*shape).astype(np.float32)
            mask = np.zeros((32, 32), dtype=bool)
            mask[5:15, 5:15] = True

            _make_geotiff(d / f"scene_{i:02d}.tif", current)
            _make_geotiff(d / f"scene_{i:02d}_prev.tif", previous)
            _make_geotiff(d / f"scene_{i:02d}_next.tif", next_img)
            _make_geotiff(d / f"scene_{i:02d}_cloud_mask.tif", mask.astype(np.uint8))
        return d

    def test_batch_all_reconstructed(self, tmp_path):
        batch_dir = self._make_batch_dir(tmp_path, 3)
        out = tmp_path / "batch_out"
        results = reconstruct_batch(batch_dir, out)
        assert len(results) == 3
        for r in results:
            assert r.reconstructed_image.shape[0] == 3

    def test_batch_outputs_exist(self, tmp_path):
        batch_dir = self._make_batch_dir(tmp_path, 2)
        out = tmp_path / "batch_out"
        reconstruct_batch(batch_dir, out)
        cloud_free = list(out.glob("*_cloud_free.tif"))
        unresolved = list(out.glob("*_unresolved.tif"))
        assert len(cloud_free) == 2
        assert len(unresolved) == 2

    def test_batch_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        out = tmp_path / "batch_out"
        results = reconstruct_batch(empty, out)
        assert results == []

    def test_batch_incomplete_triplet_skipped(self, tmp_path):
        d = tmp_path / "batch"
        d.mkdir()
        shape = (3, 16, 16)
        # Only current + mask, missing prev/next
        _make_geotiff(d / "scene_00.tif", np.random.rand(*shape).astype(np.float32))
        _make_geotiff(
            d / "scene_00_cloud_mask.tif",
            np.zeros((16, 16), dtype=np.uint8),
        )
        out = tmp_path / "batch_out"
        results = reconstruct_batch(d, out)
        assert results == []

    def test_batch_preserves_crs(self, tmp_path):
        d = tmp_path / "batch"
        d.mkdir()
        shape = (3, 16, 16)
        for i in range(2):
            _make_geotiff(
                d / f"s{i}.tif",
                np.random.rand(*shape).astype(np.float32),
                crs="EPSG:4326",
            )
            _make_geotiff(
                d / f"s{i}_prev.tif",
                np.random.rand(*shape).astype(np.float32),
                crs="EPSG:4326",
            )
            _make_geotiff(
                d / f"s{i}_next.tif",
                np.random.rand(*shape).astype(np.float32),
                crs="EPSG:4326",
            )
            _make_geotiff(
                d / f"s{i}_cloud_mask.tif",
                np.zeros((16, 16), dtype=np.uint8),
            )
        out = tmp_path / "batch_out"
        reconstruct_batch(d, out)
        for f in out.glob("*_cloud_free.tif"):
            with rasterio.open(f) as src:
                assert src.crs == CRS.from_epsg(4326)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_pixel(self, engine):
        ref = np.random.rand(3, 1, 1).astype(np.float32)
        prev = np.random.rand(3, 1, 1).astype(np.float32)
        nxt = np.random.rand(3, 1, 1).astype(np.float32)
        mask = np.ones((1, 1), dtype=bool)
        result = engine.reconstruct(ref, prev, nxt, mask)
        assert result.reconstructed_image.shape == (3, 1, 1)

    def test_large_image(self, engine):
        """Reconstruction on a larger image completes without error."""
        h, w = 512, 512
        ref = np.random.rand(3, h, w).astype(np.float32)
        prev = np.random.rand(3, h, w).astype(np.float32)
        nxt = np.random.rand(3, h, w).astype(np.float32)
        mask = np.random.rand(h, w) > 0.8
        result = engine.reconstruct(ref, prev, nxt, mask)
        assert result.reconstructed_image.shape == (3, h, w)

    def test_all_nan_temporal_images(self, engine):
        shape = (3, 32, 32)
        current = np.random.rand(*shape).astype(np.float32)
        prev = np.full(shape, np.nan, dtype=np.float32)
        nxt = np.full(shape, np.nan, dtype=np.float32)
        mask = np.ones((32, 32), dtype=bool)
        result = engine.reconstruct(current, prev, nxt, mask)
        assert result.total_unresolved == 32 * 32
        assert result.overall_replacement_rate == 0.0

    def test_current_always_preserved_where_clear(self, engine):
        """Clear-sky pixels in current must never be modified."""
        shape = (3, 32, 32)
        current = np.full(shape, 0.42, dtype=np.float32)
        prev = np.full(shape, 0.99, dtype=np.float32)
        nxt = np.full(shape, 0.88, dtype=np.float32)
        mask = np.zeros((32, 32), dtype=bool)
        result = engine.reconstruct(current, prev, nxt, mask)
        np.testing.assert_array_equal(result.reconstructed_image, current)

    def test_mixed_validity_per_band(self, engine):
        """Different bands may have different validity patterns."""
        shape = (3, 32, 32)
        current = np.random.rand(*shape).astype(np.float32)
        previous = np.random.rand(*shape).astype(np.float32)
        next_img = np.random.rand(*shape).astype(np.float32)

        # Band 0 of prev is all invalid
        previous[0, :, :] = np.nan
        # Band 1,2 of next are invalid in the cloud region
        next_img[1:, :10, :10] = np.nan

        mask = np.zeros((32, 32), dtype=bool)
        mask[:10, :10] = True

        result = engine.reconstruct(current, previous, next_img, mask)
        # Band 0: prev invalid → uses next (valid)
        # Bands 1,2: prev valid → uses prev
        assert result.reconstructed_image.shape == shape

    def test_reconstruct_returns_copy_not_view(self, engine, three_band_images, cloud_mask_partial):
        current, previous, next_img = three_band_images
        result = engine.reconstruct(current, previous, next_img, cloud_mask_partial)
        # Modifying result should not affect originals
        result.reconstructed_image[0, 40, 40] = -1.0
        assert current[0, 40, 40] != -1.0
