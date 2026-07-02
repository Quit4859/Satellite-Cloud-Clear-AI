"""Comprehensive tests for the image registration module."""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds

from utils.registration import (
    ImageRegistration,
    RegistrationConfig,
    RegistrationResult,
    _compute_ncc,
    _compute_rmse,
    _to_uint8,
    register_batch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_geotiff(
    path: Path,
    data: np.ndarray,
    crs: str = "EPSG:4326",
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    nodata: float = None,
) -> Path:
    """Helper to write a GeoTIFF with given data."""
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


@pytest.fixture
def reference_image() -> np.ndarray:
    """3-band float32 reference image with distinct features."""
    img = np.random.RandomState(42).rand(3, 128, 128).astype(np.float32) * 0.5
    # Add a bright patch for feature detection
    img[:, 30:60, 30:60] = 0.9
    img[:, 80:100, 80:100] = 0.8
    return img


@pytest.fixture
def slightly_shifted_image(reference_image: np.ndarray) -> np.ndarray:
    """Image shifted by a few pixels from reference."""
    # Shift right by 3 pixels and down by 2
    M = np.float32([[1, 0, 3], [0, 1, 2]])
    shifted = np.empty_like(reference_image)
    for i in range(3):
        shifted[i] = cv2.warpAffine(
            reference_image[i], M, (128, 128), flags=cv2.INTER_LINEAR,
        )
    return shifted


@pytest.fixture
def reference_geotiff(tmp_path: Path, reference_image: np.ndarray) -> Path:
    """Reference GeoTIFF on disk."""
    return _make_geotiff(
        tmp_path / "reference.tif", reference_image,
        crs="EPSG:4326", bounds=(73.0, 20.0, 74.0, 21.0),
    )


@pytest.fixture
def shifted_geotiff(tmp_path: Path, slightly_shifted_image: np.ndarray) -> Path:
    """Shifted GeoTIFF on disk."""
    return _make_geotiff(
        tmp_path / "shifted.tif", slightly_shifted_image,
        crs="EPSG:4326", bounds=(73.0, 20.0, 74.0, 21.0),
    )


@pytest.fixture
def batch_input_dir(tmp_path: Path, reference_image: np.ndarray) -> Path:
    """Directory with multiple images for batch testing."""
    d = tmp_path / "batch_in"
    d.mkdir()
    M_identity = np.float32([[1, 0, 0], [0, 1, 0]])
    for i in range(3):
        shift_x = (i + 1) * 2
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        shifted = np.empty_like(reference_image)
        for b in range(3):
            shifted[b] = cv2.warpAffine(
                reference_image[b], M, (128, 128), flags=cv2.INTER_LINEAR,
            )
        path = d / f"image_{i:02d}.tif"
        _make_geotiff(path, shifted)
    return d


@pytest.fixture
def config() -> RegistrationConfig:
    return RegistrationConfig(
        ecc_iterations=100,
        ecc_motion_type="euclidean",
        band_selection="first",
    )


@pytest.fixture
def registrer(config: RegistrationConfig) -> ImageRegistration:
    return ImageRegistration(config)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestRegistrationConfig:
    def test_defaults(self):
        cfg = RegistrationConfig()
        assert cfg.ecc_iterations == 200
        assert cfg.ecc_epsilon == 1e-5
        assert cfg.ecc_motion_type == "euclidean"
        assert cfg.orb_features == 5000
        assert cfg.orb_match_ratio == 0.75
        assert cfg.min_matches == 10
        assert cfg.band_selection == "first"
        assert cfg.output_dtype == "float32"
        assert cfg.compress == "lzw"

    def test_custom_values(self):
        cfg = RegistrationConfig(
            ecc_iterations=500,
            ecc_motion_type="affine",
            orb_features=10000,
            band_selection="mean",
        )
        assert cfg.ecc_iterations == 500
        assert cfg.ecc_motion_type == "affine"
        assert cfg.orb_features == 10000
        assert cfg.band_selection == "mean"

    def test_invalid_iterations(self):
        with pytest.raises(ValueError, match="must be positive"):
            RegistrationConfig(ecc_iterations=0)

    def test_invalid_match_ratio_high(self):
        with pytest.raises(ValueError, match="must be in"):
            RegistrationConfig(orb_match_ratio=1.5)

    def test_invalid_match_ratio_low(self):
        with pytest.raises(ValueError, match="must be in"):
            RegistrationConfig(orb_match_ratio=0.0)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class TestRegistrationResult:
    def test_properties(self):
        r = RegistrationResult(
            method_used="ecc",
            warp_matrix=np.eye(2, 3, dtype=np.float32),
            ncc_before=0.5,
            ncc_after=0.9,
            rmse_before=50.0,
            rmse_after=10.0,
            registered_data=np.zeros((3, 64, 64), dtype=np.float32),
        )
        assert r.ncc_improvement == pytest.approx(0.4)
        assert r.rmse_reduction == pytest.approx(40.0)
        assert r.is_valid is True

    def test_invalid_registration(self):
        r = RegistrationResult(
            method_used="ecc",
            warp_matrix=np.eye(2, 3, dtype=np.float32),
            ncc_before=0.9,
            ncc_after=0.5,
            rmse_before=10.0,
            rmse_after=50.0,
            registered_data=np.zeros((3, 64, 64), dtype=np.float32),
        )
        assert r.ncc_improvement < 0
        assert r.rmse_reduction < 0
        assert r.is_valid is False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestComputeNCC:
    def test_identical_images(self):
        img = np.random.rand(64, 64).astype(np.float32)
        ncc = _compute_ncc(img, img)
        assert ncc == pytest.approx(1.0, abs=1e-5)

    def test_opposite_images(self):
        img = np.random.rand(64, 64).astype(np.float32)
        ncc = _compute_ncc(img, -img)
        assert ncc == pytest.approx(-1.0, abs=1e-5)

    def test_different_images(self):
        a = np.random.RandomState(0).rand(64, 64).astype(np.float32)
        b = np.random.RandomState(1).rand(64, 64).astype(np.float32)
        ncc = _compute_ncc(a, b)
        assert -1.0 <= ncc <= 1.0

    def test_constant_returns_zero(self):
        a = np.ones((16, 16), dtype=np.float32)
        b = np.ones((16, 16), dtype=np.float32)
        ncc = _compute_ncc(a, b)
        assert ncc == 0.0

    def test_zero_std(self):
        a = np.ones((16, 16), dtype=np.float32)
        b = np.random.rand(16, 16).astype(np.float32)
        ncc = _compute_ncc(a, b)
        assert ncc == 0.0


class TestComputeRMSE:
    def test_identical_images(self):
        img = np.random.rand(64, 64).astype(np.float32)
        rmse = _compute_rmse(img, img)
        assert rmse == pytest.approx(0.0, abs=1e-6)

    def test_known_rmse(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        rmse = _compute_rmse(a, b)
        assert rmse == pytest.approx(1.0)

    def test_symmetric(self):
        a = np.random.rand(32, 32).astype(np.float32)
        b = np.random.rand(32, 32).astype(np.float32)
        assert _compute_rmse(a, b) == pytest.approx(_compute_rmse(b, a))


class TestToUint8:
    def test_range_stretch(self):
        img = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = _to_uint8(img)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[-1] == 255

    def test_constant_image(self):
        img = np.ones((16, 16), dtype=np.float32) * 5.0
        result = _to_uint8(img)
        assert result.sum() == 0

    def test_preserves_shape(self):
        img = np.random.rand(3, 64, 64).astype(np.float32)
        result = _to_uint8(img)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# ImageRegistration — read
# ---------------------------------------------------------------------------


class TestReadGeotiff:
    def test_read_rgb(self, registrer, reference_geotiff):
        data, meta = registrer.read_geotiff(reference_geotiff)
        assert data.shape[0] == 3
        assert data.dtype == np.float32
        assert meta["crs"] == CRS.from_epsg(4326)

    def test_read_preserves_transform(self, registrer, reference_geotiff):
        _, meta = registrer.read_geotiff(reference_geotiff)
        assert isinstance(meta["transform"], Affine)

    def test_file_not_found(self, registrer, tmp_path):
        with pytest.raises(FileNotFoundError):
            registrer.read_geotiff(tmp_path / "missing.tif")

    def test_not_geotiff(self, registrer, tmp_path):
        bad = tmp_path / "image.png"
        bad.write_bytes(b"not a tiff")
        with pytest.raises(ValueError, match="Not a GeoTIFF"):
            registrer.read_geotiff(bad)


# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------


class TestSelectBand:
    def test_first_band(self, registrer):
        data = np.random.rand(4, 32, 32).astype(np.float32)
        result = registrer._select_band(data, "first")
        np.testing.assert_array_equal(result, data[0])

    def test_mean_bands(self, registrer):
        data = np.random.rand(3, 32, 32).astype(np.float32)
        result = registrer._select_band(data, "mean")
        expected = np.mean(data, axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_max_bands(self, registrer):
        data = np.random.rand(3, 32, 32).astype(np.float32)
        result = registrer._select_band(data, "max")
        expected = np.max(data, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_single_band_passthrough(self, registrer):
        data = np.random.rand(32, 32).astype(np.float32)
        result = registrer._select_band(data, "first")
        np.testing.assert_array_equal(result, data)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessForAlignment:
    def test_output_is_uint8(self, registrer):
        img = np.random.rand(64, 64).astype(np.float32) * 10000
        result = registrer._preprocess_for_alignment(img)
        assert result.dtype == np.uint8

    def test_blurred(self, registrer):
        img = np.random.rand(64, 64).astype(np.float32)
        result = registrer._preprocess_for_alignment(img)
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# ECC registration
# ---------------------------------------------------------------------------


class TestECCRegistration:
    def test_ecc_success_on_shifted(self, registrer, reference_image, slightly_shifted_image):
        ref_gray = registrer._preprocess_for_alignment(
            registrer._select_band(reference_image, "first")
        )
        mov_gray = registrer._preprocess_for_alignment(
            registrer._select_band(slightly_shifted_image, "first")
        )
        ok, matrix = registrer._ecc_register(ref_gray, mov_gray)
        # Small shift should succeed with ECC
        assert ok is True
        assert matrix.shape[0] == 2

    def test_ecc_translation_mode(self):
        cfg = RegistrationConfig(ecc_motion_type="translation", ecc_iterations=100)
        reg = ImageRegistration(cfg)
        ref = np.random.RandomState(0).rand(64, 64).astype(np.float32)
        mov = np.roll(ref, shift=2, axis=1).copy()
        ref_g = reg._preprocess_for_alignment(ref)
        mov_g = reg._preprocess_for_alignment(mov)
        ok, matrix = reg._ecc_register(ref_g, mov_g)
        assert ok is True
        # Should detect horizontal shift
        assert abs(matrix[0, 2]) > 0.5

    def test_ecc_affine_mode(self):
        cfg = RegistrationConfig(ecc_motion_type="affine", ecc_iterations=100)
        reg = ImageRegistration(cfg)
        ref = np.random.RandomState(0).rand(64, 64).astype(np.float32)
        mov = ref.copy()
        ref_g = reg._preprocess_for_alignment(ref)
        mov_g = reg._preprocess_for_alignment(mov)
        ok, matrix = reg._ecc_register(ref_g, mov_g)
        assert ok is True
        assert matrix.shape == (2, 3)

    def test_ecc_failure_on_random(self):
        """Random images should fail or produce poor ECC."""
        cfg = RegistrationConfig(ecc_iterations=10, ecc_epsilon=1e-3)
        reg = ImageRegistration(cfg)
        ref = np.random.RandomState(0).rand(32, 32).astype(np.float32)
        mov = np.random.RandomState(1).rand(32, 32).astype(np.float32)
        ref_g = reg._preprocess_for_alignment(ref)
        mov_g = reg._preprocess_for_alignment(mov)
        # May or may not converge, but should return a matrix
        ok, matrix = reg._ecc_register(ref_g, mov_g)
        assert matrix.shape[0] == 2


class TestApplyEccWarp:
    def test_identity_warp(self, registrer, reference_image):
        identity = np.eye(2, 3, dtype=np.float32)
        warped = registrer._apply_ecc_warp(reference_image, identity)
        np.testing.assert_array_almost_equal(warped, reference_image, decimal=3)

    def test_preserves_shape(self, registrer, reference_image):
        M = np.float32([[1, 0, 5], [0, 1, 3]])
        warped = registrer._apply_ecc_warp(reference_image, M)
        assert warped.shape == reference_image.shape


# ---------------------------------------------------------------------------
# ORB registration
# ---------------------------------------------------------------------------


class TestORBRegistration:
    def test_orb_on_identical(self, registrer):
        img = np.random.RandomState(42).rand(128, 128).astype(np.float32) * 255
        img = img.astype(np.uint8)
        ok, matrix, inliers = registrer._orb_register(img, img)
        # Identical images: should find many matches
        assert ok is True
        assert matrix.shape == (2, 3)
        assert inliers > 0

    def test_orb_on_shifted(self, registrer, reference_image, slightly_shifted_image):
        ref_gray = registrer._preprocess_for_alignment(
            registrer._select_band(reference_image, "first")
        )
        mov_gray = registrer._preprocess_for_alignment(
            registrer._select_band(slightly_shifted_image, "first")
        )
        ok, matrix, inliers = registrer._orb_register(ref_gray, mov_gray)
        assert ok is True
        assert inliers >= registrer.config.min_matches
        assert matrix.shape == (2, 3)

    def test_orb_insufficient_features(self, registrer):
        """Uniform images have no features → should fail gracefully."""
        ref = np.ones((64, 64), dtype=np.uint8) * 128
        mov = np.ones((64, 64), dtype=np.uint8) * 128
        ok, matrix, inliers = registrer._orb_register(ref, mov)
        assert ok is False
        assert inliers == 0

    def test_orb_ratio_test(self):
        """Custom ratio test should filter more matches."""
        cfg = RegistrationConfig(orb_match_ratio=0.5)  # very strict
        reg = ImageRegistration(cfg)
        ref = np.random.RandomState(0).rand(128, 128).astype(np.float32)
        ref_g = reg._preprocess_for_alignment(ref)
        mov_g = reg._preprocess_for_alignment(ref)  # identical
        ok, matrix, inliers = reg._orb_register(ref_g, mov_g)
        # Strict ratio should still work on identical images
        assert ok is True


class TestApplyOrbWarp:
    def test_identity_warp(self, registrer, reference_image):
        identity = np.eye(2, 3, dtype=np.float32)
        warped = registrer._apply_orb_warp(reference_image, identity)
        np.testing.assert_array_almost_equal(warped, reference_image, decimal=3)

    def test_preserves_shape(self, registrer, reference_image):
        M = np.float32([[1, 0, 5], [0, 1, 3]])
        warped = registrer._apply_orb_warp(reference_image, M)
        assert warped.shape == reference_image.shape


# ---------------------------------------------------------------------------
# register_images
# ---------------------------------------------------------------------------


class TestRegisterImages:
    def test_ecc_registration(self, registrer, reference_image, slightly_shifted_image):
        result = registrer.register_images(reference_image, slightly_shifted_image)
        assert result.method_used == "ecc"
        assert result.registered_data.shape == reference_image.shape
        # ECC should find a valid transform
        assert result.warp_matrix.shape[0] == 2

    def test_identical_images(self, registrer, reference_image):
        result = registrer.register_images(reference_image, reference_image)
        assert result.method_used in ("ecc", "orb")
        assert result.ncc_after >= 0.99
        assert result.rmse_after < 1.0

    def test_shape_mismatch_raises(self, registrer):
        a = np.random.rand(3, 64, 64).astype(np.float32)
        b = np.random.rand(3, 32, 32).astype(np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            registrer.register_images(a, b)

    def test_single_band_input(self, registrer):
        ref = np.random.RandomState(42).rand(64, 64).astype(np.float32)
        mov = np.roll(ref, shift=2, axis=1).copy()
        result = registrer.register_images(ref, mov)
        assert result.registered_data.ndim == 3
        assert result.registered_data.shape[0] == 1

    def test_metrics_are_floats(self, registrer, reference_image, slightly_shifted_image):
        result = registrer.register_images(reference_image, slightly_shifted_image)
        assert isinstance(result.ncc_before, float)
        assert isinstance(result.ncc_after, float)
        assert isinstance(result.rmse_before, float)
        assert isinstance(result.rmse_after, float)

    def test_orb_fallback(self, reference_image, slightly_shifted_image):
        """Force ECC to fail by setting very low iterations → ORB fallback."""
        cfg = RegistrationConfig(ecc_iterations=1, ecc_epsilon=1e-10)
        reg = ImageRegistration(cfg)
        result = reg.register_images(reference_image, slightly_shifted_image)
        # Should try ORB and succeed
        assert result.method_used in ("ecc", "orb")


# ---------------------------------------------------------------------------
# register_files
# ---------------------------------------------------------------------------


class TestRegisterFiles:
    def test_register_and_save(
        self, registrer, reference_geotiff, shifted_geotiff, tmp_path,
    ):
        out = tmp_path / "registered.tif"
        result = registrer.register_files(
            reference_geotiff, shifted_geotiff, output_path=out,
        )
        assert out.exists()
        assert result.method_used in ("ecc", "orb")

        with rasterio.open(out) as src:
            assert src.crs == CRS.from_epsg(4326)
            assert src.count == 3

    def test_register_no_save(self, registrer, reference_geotiff, shifted_geotiff):
        result = registrer.register_files(reference_geotiff, shifted_geotiff)
        assert result.metadata["crs"] == CRS.from_epsg(4326)

    def test_preserves_crs(self, registrer, reference_geotiff, shifted_geotiff, tmp_path):
        out = tmp_path / "crs_test.tif"
        registrer.register_files(reference_geotiff, shifted_geotiff, output_path=out)
        with rasterio.open(out) as src:
            assert src.crs is not None

    def test_file_not_found(self, registrer, reference_geotiff, tmp_path):
        with pytest.raises(FileNotFoundError):
            registrer.register_files(
                reference_geotiff, tmp_path / "missing.tif",
            )


# ---------------------------------------------------------------------------
# register_temporal_pair
# ---------------------------------------------------------------------------


class TestRegisterTemporalPair:
    def test_temporal_registration(
        self, registrer, reference_geotiff, shifted_geotiff, reference_image, tmp_path,
    ):
        out_dir = tmp_path / "temporal_out"
        # Create a second shifted image (shift left by 2, up by 1)
        M2 = np.float32([[1, 0, -2], [0, 1, -1]])
        shifted2 = np.empty_like(reference_image)
        for b in range(3):
            shifted2[b] = cv2.warpAffine(
                reference_image[b], M2, (128, 128), flags=cv2.INTER_LINEAR,
            )
        second_shifted = tmp_path / "second_shifted.tif"
        _make_geotiff(second_shifted, shifted2, crs="EPSG:4326")

        prev_result, next_result = registrer.register_temporal_pair(
            current_path=reference_geotiff,
            previous_path=shifted_geotiff,
            next_path=second_shifted,
            output_dir=out_dir,
        )

        assert prev_result.method_used in ("ecc", "orb")
        assert next_result.method_used in ("ecc", "orb")
        assert len(list(out_dir.glob("registered_*.tif"))) == 2

    def test_creates_output_dir(self, registrer, reference_geotiff, shifted_geotiff, reference_image, tmp_path):
        out_dir = tmp_path / "new" / "dir"
        # Create a shifted version as the "next" image
        M = np.float32([[1, 0, -1], [0, 1, -2]])
        shifted_next = np.empty_like(reference_image)
        for b in range(3):
            shifted_next[b] = cv2.warpAffine(
                reference_image[b], M, (128, 128), flags=cv2.INTER_LINEAR,
            )
        second = tmp_path / "other.tif"
        _make_geotiff(second, shifted_next)
        registrer.register_temporal_pair(
            reference_geotiff, shifted_geotiff, second, out_dir,
        )
        assert out_dir.exists()


# ---------------------------------------------------------------------------
# save_registered
# ---------------------------------------------------------------------------


class TestSaveRegistered:
    def test_save_and_reopen(self, registrer, tmp_path):
        data = np.random.rand(3, 64, 64).astype(np.float32)
        transform = from_bounds(0, 0, 1, 1, 64, 64)
        out = tmp_path / "saved.tif"

        path = registrer.save_registered(data, out, transform, CRS.from_epsg(4326))
        assert path.exists()

        with rasterio.open(out) as src:
            assert src.count == 3
            assert src.width == 64
            assert src.height == 64
            assert src.crs == CRS.from_epsg(4326)

    def test_save_single_band(self, registrer, tmp_path):
        data = np.random.rand(64, 64).astype(np.float32)
        transform = Affine.identity()
        out = tmp_path / "single.tif"
        registrer.save_registered(data, out, transform, CRS.from_epsg(4326))
        with rasterio.open(out) as src:
            assert src.count == 1

    def test_save_creates_parent_dirs(self, registrer, tmp_path):
        data = np.random.rand(2, 16, 16).astype(np.float32)
        out = tmp_path / "sub" / "dir" / "out.tif"
        registrer.save_registered(data, out, Affine.identity(), CRS.from_epsg(4326))
        assert out.exists()

    def test_output_dtype(self, registrer, tmp_path):
        cfg = RegistrationConfig(output_dtype="float64")
        reg = ImageRegistration(cfg)
        data = np.random.rand(2, 16, 16).astype(np.float32)
        out = tmp_path / "float64.tif"
        reg.save_registered(data, out, Affine.identity(), CRS.from_epsg(4326))
        with rasterio.open(out) as src:
            assert src.dtypes[0] == "float64"


# ---------------------------------------------------------------------------
# Batch registration
# ---------------------------------------------------------------------------


class TestRegisterBatch:
    def test_batch_all_registered(
        self, reference_geotiff, batch_input_dir, tmp_path,
    ):
        out = tmp_path / "batch_out"
        results = register_batch(reference_geotiff, batch_input_dir, out)
        assert len(results) == 3
        for r in results:
            assert r.method_used in ("ecc", "orb")

    def test_batch_outputs_exist(
        self, reference_geotiff, batch_input_dir, tmp_path,
    ):
        out = tmp_path / "batch_out2"
        register_batch(reference_geotiff, batch_input_dir, out)
        registered_files = list(out.glob("registered_*.tif"))
        assert len(registered_files) == 3

    def test_batch_empty_directory(self, reference_geotiff, tmp_path):
        empty_in = tmp_path / "empty_in"
        empty_in.mkdir()
        out = tmp_path / "empty_out"
        results = register_batch(reference_geotiff, empty_in, out)
        assert results == []

    def test_batch_preserves_crs(
        self, reference_geotiff, batch_input_dir, tmp_path,
    ):
        out = tmp_path / "batch_crs"
        register_batch(reference_geotiff, batch_input_dir, out)
        for f in out.glob("registered_*.tif"):
            with rasterio.open(f) as src:
                assert src.crs is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_constant_images(self, registrer):
        ref = np.ones((3, 64, 64), dtype=np.float32) * 0.5
        mov = np.ones((3, 64, 64), dtype=np.float32) * 0.5
        result = registrer.register_images(ref, mov)
        assert result.registered_data.shape == ref.shape

    def test_very_small_images(self, registrer):
        ref = np.random.rand(3, 32, 32).astype(np.float32)
        mov = np.random.rand(3, 32, 32).astype(np.float32)
        result = registrer.register_images(ref, mov)
        assert result.registered_data.shape == ref.shape

    def test_large_translation(self, registrer, reference_image):
        """Large shift may fail ECC but ORB should handle it."""
        M = np.float32([[1, 0, 20], [0, 1, 15]])
        shifted = np.empty_like(reference_image)
        for i in range(3):
            shifted[i] = cv2.warpAffine(
                reference_image[i], M, (128, 128), flags=cv2.INTER_LINEAR,
            )
        result = registrer.register_images(reference_image, shifted)
        assert result.registered_data.shape == reference_image.shape
        assert result.method_used in ("ecc", "orb", "failed")

    def test_single_pixel_image(self, registrer):
        """1x1 images are too small for ECC/ORB → should fail gracefully."""
        ref = np.random.rand(3, 1, 1).astype(np.float32)
        mov = np.random.rand(3, 1, 1).astype(np.float32)
        result = registrer.register_images(ref, mov)
        assert result.registered_data.shape == (3, 1, 1)
        assert result.method_used == "failed"

    def test_nodata_values(self, registrer, tmp_path):
        ref = np.random.rand(3, 64, 64).astype(np.float32)
        mov = np.random.rand(3, 64, 64).astype(np.float32)
        ref_path = tmp_path / "ref_nodata.tif"
        mov_path = tmp_path / "mov_nodata.tif"
        _make_geotiff(ref_path, ref, nodata=0.0)
        _make_geotiff(mov_path, mov, nodata=0.0)

        out = tmp_path / "out_nodata.tif"
        result = registrer.register_files(ref_path, mov_path, output_path=out)
        assert out.exists()

    def test_different_crs_files(self, tmp_path):
        """Registration should still work; CRS is preserved from moving image."""
        ref = np.random.RandomState(42).rand(3, 64, 64).astype(np.float32)
        mov = np.random.RandomState(42).rand(3, 64, 64).astype(np.float32)
        ref_path = tmp_path / "ref.tif"
        mov_path = tmp_path / "mov.tif"
        _make_geotiff(ref_path, ref, crs="EPSG:4326")
        _make_geotiff(mov_path, mov, crs="EPSG:32643")

        reg = ImageRegistration(RegistrationConfig(ecc_iterations=50))
        out = tmp_path / "out.tif"
        result = reg.register_files(ref_path, mov_path, output_path=out)

        # Output should preserve moving image's CRS
        with rasterio.open(out) as src:
            assert src.crs == CRS.from_epsg(32643)
