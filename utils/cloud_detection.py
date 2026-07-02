"""Cloud detection module for satellite imagery.

Provides an abstract CloudDetector interface and a concrete
ThresholdCloudDetector implementation using brightness thresholding,
Otsu segmentation, morphological filtering, and connected-component
analysis.  Designed so a future AI segmentation model can replace the
detector by subclassing CloudDetector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
from loguru import logger
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import Affine


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CloudDetectionConfig:
    """Configuration for threshold-based cloud detection."""

    brightness_threshold: Optional[int] = None
    """Fixed brightness threshold (0-255). When None, Otsu auto-selects."""

    use_otsu: bool = True
    """Use Otsu automatic thresholding instead of a fixed threshold."""

    morph_open_kernel: int = 3
    """Kernel size for morphological opening (noise removal)."""

    morph_close_kernel: int = 7
    """Kernel size for morphological closing (gap filling)."""

    morph_iterations: int = 1
    """Number of iterations for each morphological operation."""

    min_component_area: int = 50
    """Minimum connected-component area in pixels to keep."""

    composite_band: str = "mean"
    """How to collapse bands before thresholding: 'mean', 'max', 'rgb_max'."""


@dataclass
class CloudDetectionResult:
    """Output of cloud detection on a single image."""

    cloud_mask: np.ndarray
    """Boolean array (H, W) — True where cloud is detected."""

    coverage: float
    """Fraction of valid pixels classified as cloud (0.0 – 1.0)."""

    threshold_used: float
    """The brightness threshold that was applied."""

    component_count: int
    """Number of connected cloud components after filtering."""

    metadata: Dict = field(default_factory=dict)
    """Source file metadata carried through for saving."""

    @property
    def shape(self) -> Tuple[int, int]:
        return self.cloud_mask.shape


# ---------------------------------------------------------------------------
# Abstract base — swap this for an AI model later
# ---------------------------------------------------------------------------

class CloudDetector(ABC):
    """Abstract interface for cloud detectors.

    Subclass this and implement ``detect`` to plug in a neural-network-
    based segmenter.  The rest of the pipeline (saving, batch, coverage)
    works unchanged.
    """

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> CloudDetectionResult:
        """Run cloud detection on an image array.

        Args:
            image: (bands, H, W) or (H, W) float32 array.
            metadata: Optional metadata dict for pass-through.

        Returns:
            CloudDetectionResult with the binary mask and stats.
        """
        ...

    @abstractmethod
    def detect_file(
        self,
        file_path: Union[str, Path],
        output_mask: Optional[Union[str, Path]] = None,
        output_png: Optional[Union[str, Path]] = None,
    ) -> CloudDetectionResult:
        """Detect clouds in a GeoTIFF file and optionally save outputs.

        Args:
            file_path: Input GeoTIFF path.
            output_mask: If given, save the binary mask as a GeoTIFF.
            output_png: If given, save a visual overlay as a PNG.

        Returns:
            CloudDetectionResult.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete threshold-based detector
# ---------------------------------------------------------------------------

class ThresholdCloudDetector(CloudDetector):
    """Brightness-threshold + Otsu cloud detector.

    Pipeline per image:
        1. Collapse bands into a single brightness map.
        2. Threshold (fixed or Otsu).
        3. Morphological opening  → remove small noise.
        4. Morphological closing  → fill small holes.
        5. Connected-component filtering → drop tiny blobs.
        6. Compute coverage statistics.
    """

    def __init__(self, config: Optional[CloudDetectionConfig] = None) -> None:
        self.config = config or CloudDetectionConfig()

    # ------------------------------------------------------------------
    # Band compositing
    # ------------------------------------------------------------------

    def _to_brightness(self, image: np.ndarray) -> np.ndarray:
        """Collapse multi-band image to a single brightness map in [0, 255].

        Args:
            image: (bands, H, W) float array, any range.

        Returns:
            (H, W) uint8 array.
        """
        if image.ndim == 2:
            band = image
        else:
            if self.config.composite_band == "max":
                band = np.nanmax(image, axis=0)
            elif self.config.composite_band == "rgb_max" and image.shape[0] >= 3:
                band = np.nanmax(image[:3], axis=0)
            else:
                band = np.nanmean(image, axis=0)

        band = np.nan_to_num(band, nan=0.0)

        vmin, vmax = float(band.min()), float(band.max())
        if vmax - vmin < 1e-10:
            return np.zeros(band.shape, dtype=np.uint8)

        normalized = ((band - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        return normalized

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------

    def _threshold(self, brightness: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply brightness thresholding.

        When the image is uniform (zero variance), Otsu degenerates to
        threshold=0 or 255 and produces meaningless results.  In that
        case we fall back to the configured fixed threshold or an empty
        mask.

        Args:
            brightness: uint8 (H, W) array.

        Returns:
            Tuple of (binary mask bool array, threshold value used).
        """
        if self.config.use_otsu:
            # Guard: if variance is zero, Otsu is degenerate.
            if np.ptp(brightness) == 0:
                thresh_val = float(self.config.brightness_threshold or 255)
                return np.zeros(brightness.shape, dtype=bool), thresh_val

            thresh_val, binary = cv2.threshold(
                brightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            logger.debug(f"Otsu threshold selected: {thresh_val}")
        else:
            thresh_val = self.config.brightness_threshold or 180
            _, binary = cv2.threshold(brightness, thresh_val, 255, cv2.THRESH_BINARY)

        return binary.astype(bool), float(thresh_val)

    # ------------------------------------------------------------------
    # Morphology
    # ------------------------------------------------------------------

    def _morphological_open(self, mask: np.ndarray) -> np.ndarray:
        """Remove small noise blobs via erosion → dilation."""
        k = self.config.morph_open_kernel
        if k < 1:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(
            mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel,
            iterations=self.config.morph_iterations,
        )
        return opened.astype(bool)

    def _morphological_close(self, mask: np.ndarray) -> np.ndarray:
        """Fill small gaps inside cloud regions via dilation → erosion."""
        k = self.config.morph_close_kernel
        if k < 1:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(
            mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel,
            iterations=self.config.morph_iterations,
        )
        return closed.astype(bool)

    # ------------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------------

    def _filter_components(self, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Remove connected components smaller than min_component_area.

        Returns:
            Tuple of (filtered mask, number of remaining components).
        """
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8,
        )

        keep = np.zeros_like(mask, dtype=bool)
        count = 0
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.config.min_component_area:
                keep |= labels == i
                count += 1

        return keep, count

    # ------------------------------------------------------------------
    # Core detect
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> CloudDetectionResult:
        """Run the full threshold-based cloud detection pipeline.

        Args:
            image: (bands, H, W) or (H, W) float32 array.
            metadata: Optional source metadata dict.

        Returns:
            CloudDetectionResult.
        """
        brightness = self._to_brightness(image)
        mask, thresh = self._threshold(brightness)
        mask = self._morphological_open(mask)
        mask = self._morphological_close(mask)
        mask, n_components = self._filter_components(mask)

        coverage = self._compute_coverage(mask, image)

        return CloudDetectionResult(
            cloud_mask=mask,
            coverage=coverage,
            threshold_used=thresh,
            component_count=n_components,
            metadata=metadata or {},
        )

    def _compute_coverage(
        self, mask: np.ndarray, image: np.ndarray,
    ) -> float:
        """Compute cloud fraction among valid (non-NaN) pixels."""
        if image.ndim == 3:
            valid = ~np.all(np.isnan(image), axis=0)
        elif np.issubdtype(image.dtype, np.floating):
            valid = ~np.isnan(image)
        else:
            valid = np.ones(image.shape[:2] if image.ndim == 3 else image.shape, dtype=bool)

        total = int(valid.sum())
        if total == 0:
            return 0.0
        cloud_pixels = int((mask & valid).sum())
        return cloud_pixels / total

    # ------------------------------------------------------------------
    # File-level detection
    # ------------------------------------------------------------------

    def detect_file(
        self,
        file_path: Union[str, Path],
        output_mask: Optional[Union[str, Path]] = None,
        output_png: Optional[Union[str, Path]] = None,
    ) -> CloudDetectionResult:
        """Detect clouds in a GeoTIFF and optionally save mask + PNG.

        Args:
            file_path: Input GeoTIFF.
            output_mask: Path for the binary-mask GeoTIFF (uint8).
            output_png: Path for a visual overlay PNG.

        Returns:
            CloudDetectionResult.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with rasterio.open(file_path, "r") as src:
            image = src.read().astype(np.float32)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

        metadata = {
            "file_path": str(file_path),
            "width": profile["width"],
            "height": profile["height"],
            "count": profile["count"],
            "crs": crs,
            "transform": transform,
            "bounds": bounds,
        }

        result = self.detect(image, metadata=metadata)
        logger.info(
            f"{file_path.name}: coverage={result.coverage:.2%}, "
            f"components={result.component_count}, threshold={result.threshold_used:.1f}"
        )

        if output_mask is not None:
            self._save_mask_geo(result.cloud_mask, output_mask, profile)

        if output_png is not None:
            self._save_png_overlay(result.cloud_mask, output_png)

        return result

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_mask_geo(
        mask: np.ndarray,
        path: Union[str, Path],
        source_profile: Dict,
    ) -> Path:
        """Save binary cloud mask as a single-band uint8 GeoTIFF."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        profile = source_profile.copy()
        profile.update(
            dtype="uint8",
            count=1,
            nodata=None,
            compress="lzw",
        )

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)

        logger.info(f"Saved mask → {path}")
        return path

    def _save_png_overlay(
        self,
        mask: np.ndarray,
        path: Union[str, Path],
    ) -> Path:
        """Save a PNG visualization of the cloud mask.

        Shows clouds as a semi-transparent red overlay on a neutral
        background with green valid-pixel regions.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        h, w = mask.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Green for clear sky
        canvas[~mask] = [34, 139, 34]
        # Red for cloud
        canvas[mask] = [220, 40, 40]

        cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved PNG → {path}")
        return path


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def detect_clouds_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    detector: Optional[CloudDetector] = None,
    pattern: str = "*.tif",
) -> List[CloudDetectionResult]:
    """Run cloud detection on every matching file in a directory.

    Args:
        input_dir: Folder with GeoTIFFs.
        output_dir: Where to write masks and PNGs.
        detector: CloudDetector instance (defaults to ThresholdCloudDetector).
        pattern: Glob pattern for input files.

    Returns:
        List of CloudDetectionResult objects.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if detector is None:
        detector = ThresholdCloudDetector()

    files = sorted(input_dir.glob(pattern))
    if not files:
        logger.warning(f"No files matching '{pattern}' in {input_dir}")
        return []

    logger.info(f"Cloud detection on {len(files)} files from {input_dir}")
    results: List[CloudDetectionResult] = []

    for f in files:
        try:
            mask_path = output_dir / f"cloud_mask_{f.stem}.tif"
            png_path = output_dir / f"cloud_vis_{f.stem}.png"
            result = detector.detect_file(f, output_mask=mask_path, output_png=png_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed on {f.name}: {e}")

    logger.info(
        f"Batch complete: {len(results)}/{len(files)} processed, "
        f"mean coverage={np.mean([r.coverage for r in results]):.2%}"
    )
    return results
