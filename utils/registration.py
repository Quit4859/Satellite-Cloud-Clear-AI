"""Image registration module for satellite imagery alignment.

Aligns previous and next temporal satellite images with the current
reference image using ECC (Enhanced Correlation Coefficient) alignment
with automatic ORB feature-based fallback. Preserves all geospatial
metadata through the registration pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
from loguru import logger
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RegistrationConfig:
    """Configuration for image registration."""

    ecc_iterations: int = 200
    """Maximum number of ECC iterations."""

    ecc_epsilon: float = 1e-5
    """Convergence threshold for ECC."""

    ecc_motion_type: str = "euclidean"
    """ECC motion model: 'translation', 'euclidean', 'affine', or 'homography'."""

    orb_features: int = 5000
    """Maximum number of features to detect with ORB."""

    orb_match_ratio: float = 0.75
    """Lowe's ratio test threshold for ORB feature matching."""

    min_matches: int = 10
    """Minimum number of good matches required for ORB fallback."""

    gaussian_kernel: Tuple[int, int] = (5, 5)
    """Gaussian blur kernel size for preprocessing."""

    band_selection: str = "first"
    """Which band to use for registration: 'first', 'mean', or 'max'."""

    output_dtype: str = "float32"
    """Output data type for registered images."""

    compress: str = "lzw"
    """GeoTIFF compression method."""

    def __post_init__(self) -> None:
        if self.ecc_iterations <= 0:
            raise ValueError(f"ecc_iterations must be positive, got {self.ecc_iterations}")
        if not (0.0 < self.orb_match_ratio < 1.0):
            raise ValueError(f"orb_match_ratio must be in (0, 1), got {self.orb_match_ratio}")


@dataclass
class RegistrationResult:
    """Result of registering a single image pair."""

    method_used: str
    """Registration method that succeeded: 'ecc' or 'orb'."""

    warp_matrix: np.ndarray
    """Transformation matrix estimated by the registration."""

    ncc_before: float
    """Normalized cross-correlation before alignment."""

    ncc_after: float
    """Normalized cross-correlation after alignment."""

    rmse_before: float
    """Root mean square error before alignment."""

    rmse_after: float
    """Root mean square error after alignment."""

    registered_data: np.ndarray
    """The registered image array."""

    metadata: Dict = field(default_factory=dict)
    """Source metadata for saving."""

    num_inliers: int = 0
    """Number of inlier matches (ORB only)."""

    @property
    def ncc_improvement(self) -> float:
        """Improvement in NCC (positive = better alignment)."""
        return self.ncc_after - self.ncc_before

    @property
    def rmse_reduction(self) -> float:
        """Reduction in RMSE (positive = better alignment)."""
        return self.rmse_before - self.rmse_after

    @property
    def is_valid(self) -> bool:
        """Whether the registration improved alignment metrics."""
        return self.ncc_improvement >= 0 or self.rmse_reduction >= 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_ncc(reference: np.ndarray, target: np.ndarray) -> float:
    """Compute Normalized Cross-Correlation between two images.

    Args:
        reference: Reference image (H, W) float32.
        target: Target image (H, W) float32.

    Returns:
        NCC value in [-1, 1]. Higher is better alignment.
    """
    ref = reference.astype(np.float64).ravel()
    tgt = target.astype(np.float64).ravel()

    ref_mean = ref.mean()
    tgt_mean = tgt.mean()
    ref_std = ref.std()
    tgt_std = tgt.std()

    if ref_std < 1e-10 or tgt_std < 1e-10:
        return 0.0

    n = len(ref)
    ncc = np.sum((ref - ref_mean) * (tgt - tgt_mean)) / (n * ref_std * tgt_std)
    return float(np.clip(ncc, -1.0, 1.0))


def _compute_rmse(reference: np.ndarray, target: np.ndarray) -> float:
    """Compute Root Mean Square Error between two images.

    Args:
        reference: Reference image (H, W) float32.
        target: Target image (H, W) float32.

    Returns:
        RMSE value (lower is better).
    """
    diff = reference.astype(np.float64) - target.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float image to uint8 for OpenCV operations.

    Args:
        image: Float array of any range.

    Returns:
        uint8 array scaled to [0, 255].
    """
    img = image.copy()
    vmin, vmax = float(img.min()), float(img.max())
    if vmax - vmin < 1e-10:
        return np.zeros_like(img, dtype=np.uint8)
    img = ((img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Main registration class
# ---------------------------------------------------------------------------


class ImageRegistration:
    """Registers satellite images to align with a reference image.

    Supports ECC-based alignment with automatic ORB feature-based fallback
    when ECC fails. Preserves CRS and metadata through the pipeline.
    """

    SUPPORTED_EXTENSIONS = {".tif", ".tiff"}

    def __init__(self, config: Optional[RegistrationConfig] = None) -> None:
        self.config = config or RegistrationConfig()

    def read_geotiff(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """Read a GeoTIFF and return data with metadata.

        Args:
            file_path: Path to a GeoTIFF file.

        Returns:
            Tuple of (data array with shape (bands, H, W), metadata dict).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid GeoTIFF.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Not a GeoTIFF: {file_path.suffix}")

        with rasterio.open(file_path, "r") as src:
            data = src.read().astype(np.float32)
            metadata = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "file_path": str(file_path),
            }

        logger.info(
            f"Read {file_path.name}: {data.shape[0]} bands, "
            f"{data.shape[1]}x{data.shape[2]}, dtype={data.dtype}"
        )
        return data, metadata

    def _select_band(
        self, data: np.ndarray, method: str = "first"
    ) -> np.ndarray:
        """Select or collapse bands for registration.

        Args:
            data: Multi-band array (bands, H, W).
            method: Band selection strategy.

        Returns:
            2D (H, W) array.
        """
        if data.ndim == 2:
            return data

        if method == "mean":
            return np.mean(data, axis=0)
        elif method == "max":
            return np.max(data, axis=0)
        else:
            return data[0]

    def _preprocess_for_alignment(self, image: np.ndarray) -> np.ndarray:
        """Prepare a single-band image for OpenCV registration.

        Converts to uint8 and applies Gaussian blur to reduce noise.

        Args:
            image: 2D float array.

        Returns:
            Preprocessed uint8 array.
        """
        uint8 = _to_uint8(image)
        kx, ky = self.config.gaussian_kernel
        blurred = cv2.GaussianBlur(uint8, (kx, ky), 0)
        return blurred

    # ------------------------------------------------------------------
    # ECC registration
    # ------------------------------------------------------------------

    def _ecc_register(
        self,
        reference: np.ndarray,
        moving: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """Attempt ECC-based registration.

        Args:
            reference: Reference grayscale uint8 image (H, W).
            moving: Moving grayscale uint8 image (H, W), same size as reference.

        Returns:
            Tuple of (success: bool, warp_matrix).
        """
        h, w = reference.shape
        if h < 4 or w < 4:
            logger.warning(f"ECC: image too small ({h}x{w}), minimum is 4x4")
            return False, np.eye(2, 3, dtype=np.float32)

        motion_map = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
            "homography": cv2.MOTION_HOMOGRAPHY,
        }
        motion_type = motion_map.get(self.config.ecc_motion_type, cv2.MOTION_EUCLIDEAN)

        if motion_type == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, dtype=np.float32)
        elif motion_type == cv2.MOTION_AFFINE:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        elif motion_type == cv2.MOTION_EUCLIDEAN:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.config.ecc_iterations,
            self.config.ecc_epsilon,
        )

        try:
            cc, warp_matrix = cv2.findTransformECC(
                reference, moving, warp_matrix, motion_type, criteria,
            )
            logger.debug(f"ECC converged: cc={cc:.4f}, method={self.config.ecc_motion_type}")
            return True, warp_matrix
        except cv2.error as e:
            logger.warning(f"ECC failed: {e}")
            return False, warp_matrix

    def _apply_ecc_warp(
        self,
        data: np.ndarray,
        warp_matrix: np.ndarray,
    ) -> np.ndarray:
        """Apply ECC warp matrix to multi-band image.

        Args:
            data: Multi-band array (bands, H, W).
            warp_matrix: ECC transformation matrix.

        Returns:
            Warped array with same shape and dtype.
        """
        n_bands, h, w = data.shape
        motion_map = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
            "homography": cv2.MOTION_HOMOGRAPHY,
        }
        motion_type = motion_map.get(self.config.ecc_motion_type, cv2.MOTION_EUCLIDEAN)

        warped = np.empty_like(data)
        for i in range(n_bands):
            band = data[i].astype(np.float32)
            flags = cv2.WARP_INVERSE_MAP if motion_type == cv2.MOTION_HOMOGRAPHY else 0
            warped[i] = cv2.warpAffine(
                band, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR + flags,
            )
        return warped

    # ------------------------------------------------------------------
    # ORB registration
    # ------------------------------------------------------------------

    def _orb_register(
        self,
        reference: np.ndarray,
        moving: np.ndarray,
    ) -> Tuple[bool, np.ndarray, int]:
        """Attempt ORB feature-based registration.

        Args:
            reference: Reference grayscale uint8 image (H, W).
            moving: Moving grayscale uint8 image (H, W).

        Returns:
            Tuple of (success, warp_matrix, num_inliers).
        """
        h, w = reference.shape
        if h < 16 or w < 16:
            logger.warning(f"ORB: image too small ({h}x{w}), minimum is 16x16")
            return False, np.eye(2, 3, dtype=np.float32), 0

        orb = cv2.ORB_create(nfeatures=self.config.orb_features)

        kp1, des1 = orb.detectAndCompute(reference, None)
        kp2, des2 = orb.detectAndCompute(moving, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            logger.warning("ORB: insufficient features detected")
            return False, np.eye(2, 3, dtype=np.float32), 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.config.orb_match_ratio * n.distance:
                    good_matches.append(m)

        logger.debug(
            f"ORB: {len(kp1)} ref features, {len(kp2)} mov features, "
            f"{len(good_matches)} good matches"
        )

        if len(good_matches) < self.config.min_matches:
            logger.warning(
                f"ORB: only {len(good_matches)} matches (need {self.config.min_matches})"
            )
            return False, np.eye(2, 3, dtype=np.float32), 0

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_mov = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        warp_matrix, inliers = cv2.estimateAffinePartial2D(
            pts_mov, pts_ref, method=cv2.RANSAC,
        )

        if warp_matrix is None:
            logger.warning("ORB: affine estimation failed")
            return False, np.eye(2, 3, dtype=np.float32), 0

        n_inliers = int(inliers.sum()) if inliers is not None else 0
        logger.debug(f"ORB: estimated transform with {n_inliers} inliers")
        return True, warp_matrix, n_inliers

    def _apply_orb_warp(
        self,
        data: np.ndarray,
        warp_matrix: np.ndarray,
    ) -> np.ndarray:
        """Apply ORB warp matrix to multi-band image.

        Args:
            data: Multi-band array (bands, H, W).
            warp_matrix: Affine transformation matrix (2x3).

        Returns:
            Warped array with same shape and dtype.
        """
        n_bands, h, w = data.shape
        warped = np.empty_like(data)

        for i in range(n_bands):
            band = data[i].astype(np.float32)
            warped[i] = cv2.warpAffine(
                band, warp_matrix, (w, h),
                flags=cv2.INTER_LINEAR,
            )
        return warped

    # ------------------------------------------------------------------
    # High-level registration
    # ------------------------------------------------------------------

    def register_images(
        self,
        reference: np.ndarray,
        moving: np.ndarray,
    ) -> RegistrationResult:
        """Register a moving image to align with the reference.

        Tries ECC first; falls back to ORB if ECC fails.

        Args:
            reference: Reference multi-band array (bands, H, W) or (H, W).
            moving: Moving multi-band array, same spatial dimensions as reference.

        Returns:
            RegistrationResult with aligned image and quality metrics.
        """
        if reference.ndim == 2:
            reference = reference[np.newaxis, ...]
        if moving.ndim == 2:
            moving = moving[np.newaxis, ...]

        if reference.shape != moving.shape:
            raise ValueError(
                f"Shape mismatch: reference={reference.shape}, moving={moving.shape}"
            )

        ref_gray = self._preprocess_for_alignment(
            self._select_band(reference, self.config.band_selection)
        )
        mov_gray = self._preprocess_for_alignment(
            self._select_band(moving, self.config.band_selection)
        )

        ncc_before = _compute_ncc(ref_gray.astype(np.float32), mov_gray.astype(np.float32))
        rmse_before = _compute_rmse(ref_gray.astype(np.float32), mov_gray.astype(np.float32))

        logger.info(
            f"Pre-alignment: NCC={ncc_before:.4f}, RMSE={rmse_before:.4f}"
        )

        # Attempt ECC
        ecc_ok, warp_matrix = self._ecc_register(ref_gray, mov_gray)
        if ecc_ok:
            registered = self._apply_ecc_warp(moving, warp_matrix)
            method = "ecc"
            n_inliers = 0
        else:
            # Fallback to ORB
            logger.info("ECC failed, falling back to ORB feature matching")
            orb_ok, orb_matrix, n_inliers = self._orb_register(ref_gray, mov_gray)
            if orb_ok:
                warp_matrix = orb_matrix
                registered = self._apply_orb_warp(moving, warp_matrix)
                method = "orb"
            else:
                logger.error("Both ECC and ORB registration failed")
                return RegistrationResult(
                    method_used="failed",
                    warp_matrix=np.eye(2, 3, dtype=np.float32),
                    ncc_before=ncc_before,
                    ncc_after=ncc_before,
                    rmse_before=rmse_before,
                    rmse_after=rmse_before,
                    registered_data=moving.copy(),
                    num_inliers=0,
                )

        reg_gray = self._preprocess_for_alignment(
            self._select_band(registered, self.config.band_selection)
        )
        ncc_after = _compute_ncc(ref_gray.astype(np.float32), reg_gray.astype(np.float32))
        rmse_after = _compute_rmse(ref_gray.astype(np.float32), reg_gray.astype(np.float32))

        logger.info(
            f"Post-alignment ({method}): NCC={ncc_after:.4f}, RMSE={rmse_after:.4f}, "
            f"delta_NCC={ncc_after - ncc_before:+.4f}, delta_RMSE={rmse_before - rmse_after:+.4f}"
        )

        return RegistrationResult(
            method_used=method,
            warp_matrix=warp_matrix,
            ncc_before=ncc_before,
            ncc_after=ncc_after,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            registered_data=registered,
            num_inliers=n_inliers,
        )

    def register_files(
        self,
        reference_path: Union[str, Path],
        moving_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> RegistrationResult:
        """Register a GeoTIFF to align with a reference GeoTIFF.

        Args:
            reference_path: Path to the reference GeoTIFF.
            moving_path: Path to the moving GeoTIFF to be aligned.
            output_path: If given, save the registered image.

        Returns:
            RegistrationResult with aligned data and metadata.
        """
        ref_data, ref_meta = self.read_geotiff(reference_path)
        mov_data, mov_meta = self.read_geotiff(moving_path)

        result = self.register_images(ref_data, mov_data)
        result.metadata = mov_meta

        if output_path is not None:
            self.save_registered(
                result.registered_data,
                output_path,
                transform=mov_meta["transform"],
                crs=mov_meta["crs"],
                metadata=mov_meta,
            )

        return result

    def register_temporal_pair(
        self,
        current_path: Union[str, Path],
        previous_path: Union[str, Path],
        next_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> Tuple[RegistrationResult, RegistrationResult]:
        """Register previous and next images to the current reference.

        This is the primary entry point for the satellite cloud clear pipeline:
        align both temporal neighbors to the current LISS-IV image.

        Args:
            current_path: Path to the current (reference) GeoTIFF.
            previous_path: Path to the previous temporal GeoTIFF.
            next_path: Path to the next temporal GeoTIFF.
            output_dir: Directory for saving aligned outputs.

        Returns:
            Tuple of (previous_result, next_result).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prev_stem = Path(previous_path).stem
        next_stem = Path(next_path).stem

        prev_out = output_dir / f"registered_{prev_stem}.tif"
        next_out = output_dir / f"registered_{next_stem}.tif"

        logger.info(f"Registering temporal pair around {Path(current_path).name}")

        prev_result = self.register_files(
            reference_path=current_path,
            moving_path=previous_path,
            output_path=prev_out,
        )

        next_result = self.register_files(
            reference_path=current_path,
            moving_path=next_path,
            output_path=next_out,
        )

        logger.info(
            f"Temporal registration complete: "
            f"prev={prev_result.method_used}, next={next_result.method_used}"
        )

        return prev_result, next_result

    def save_registered(
        self,
        data: np.ndarray,
        output_path: Union[str, Path],
        transform: Optional[Affine] = None,
        crs: Optional[CRS] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Save registered image as a GeoTIFF preserving geospatial metadata.

        Args:
            data: Array with shape (bands, H, W) or (H, W).
            output_path: Destination file path.
            transform: Affine transform from source.
            crs: Coordinate reference system.
            metadata: Optional metadata dict.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_bands, height, width = data.shape
        nodata = np.nan if np.issubdtype(data.dtype, np.floating) else None

        profile = {
            "driver": "GTiff",
            "dtype": self.config.output_dtype,
            "width": width,
            "height": height,
            "count": n_bands,
            "crs": crs,
            "transform": transform,
            "compress": self.config.compress,
            "nodata": nodata,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(n_bands):
                dst.write(data[i].astype(self.config.output_dtype), i + 1)

        logger.info(f"Saved registered image → {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def register_batch(
    reference_path: Union[str, Path],
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "*.tif",
    config: Optional[RegistrationConfig] = None,
) -> List[RegistrationResult]:
    """Register all matching files to a single reference image.

    Args:
        reference_path: Path to the reference GeoTIFF.
        input_dir: Directory containing moving images.
        output_dir: Directory for registered outputs.
        pattern: Glob pattern for input files.
        config: Registration configuration.

    Returns:
        List of RegistrationResult objects.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reg = ImageRegistration(config)

    files = sorted(input_dir.glob(pattern))
    if not files:
        logger.warning(f"No files matching '{pattern}' in {input_dir}")
        return []

    logger.info(f"Batch registration: {len(files)} files against {Path(reference_path).name}")
    results: List[RegistrationResult] = []

    for f in files:
        try:
            out_path = output_dir / f"registered_{f.stem}.tif"
            result = reg.register_files(reference_path, f, output_path=out_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to register {f.name}: {e}")

    if results:
        avg_ncc = np.mean([r.ncc_after for r in results])
        logger.info(
            f"Batch complete: {len(results)}/{len(files)} registered, "
            f"mean NCC={avg_ncc:.4f}"
        )
    return results
