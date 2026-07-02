"""Temporal reconstruction engine for cloud removal.

Generates cloud-free satellite imagery by replacing cloudy pixels with
observations from adjacent temporal images (previous and next acquisitions).
Follows a strict precedence: previous-first, then next, then unresolved.
Supports RGB, multispectral, batch processing, and GeoTIFF output.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from loguru import logger
from rasterio.crs import CRS
from rasterio.transform import Affine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ReconstructionConfig:
    """Configuration for temporal reconstruction."""

    invalid_data_value: Optional[float] = None
    """Pixel value treated as missing/nodata. If None, uses the file nodata."""

    max_cloud_fraction: float = 1.0
    """Reject images whose cloud fraction exceeds this threshold."""

    min_valid_fraction: float = 0.0
    """Minimum fraction of valid pixels required in temporal images."""

    output_dtype: str = "float32"
    """Data type for the reconstructed output."""

    compress: str = "lzw"
    """GeoTIFF compression method."""

    def __post_init__(self) -> None:
        if not 0.0 < self.max_cloud_fraction <= 1.0:
            raise ValueError(
                f"max_cloud_fraction must be in (0, 1], got {self.max_cloud_fraction}"
            )
        if not 0.0 <= self.min_valid_fraction < 1.0:
            raise ValueError(
                f"min_valid_fraction must be in [0, 1), got {self.min_valid_fraction}"
            )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class ReplacementStats:
    """Pixel-level replacement statistics for a single band."""

    total_pixels: int
    cloudy_pixels: int
    replaced_from_previous: int
    replaced_from_next: int
    unresolved_pixels: int

    @property
    def cloud_fraction(self) -> float:
        """Fraction of pixels that were cloudy."""
        if self.total_pixels == 0:
            return 0.0
        return self.cloudy_pixels / self.total_pixels

    @property
    def replacement_rate(self) -> float:
        """Fraction of cloudy pixels that were successfully replaced."""
        if self.cloudy_pixels == 0:
            return 1.0
        return (self.replaced_from_previous + self.replaced_from_next) / self.cloudy_pixels

    def to_dict(self) -> Dict:
        return {
            "total_pixels": self.total_pixels,
            "cloudy_pixels": self.cloudy_pixels,
            "replaced_from_previous": self.replaced_from_previous,
            "replaced_from_next": self.replaced_from_next,
            "unresolved_pixels": self.unresolved_pixels,
            "cloud_fraction": round(self.cloud_fraction, 6),
            "replacement_rate": round(self.replacement_rate, 6),
        }


@dataclass
class ReconstructionResult:
    """Full output of temporal reconstruction."""

    reconstructed_image: np.ndarray
    """Cloud-free image array (bands, H, W)."""

    unresolved_mask: np.ndarray
    """Boolean mask (H, W) — True where no temporal data was available."""

    band_stats: List[ReplacementStats]
    """Per-band replacement statistics."""

    metadata: Dict = field(default_factory=dict)
    """Source metadata carried through for saving."""

    @property
    def n_bands(self) -> int:
        return self.reconstructed_image.shape[0]

    @property
    def total_cloud_fraction(self) -> float:
        """Overall cloud fraction from the first band's stats."""
        if not self.band_stats:
            return 0.0
        return self.band_stats[0].cloud_fraction

    @property
    def overall_replacement_rate(self) -> float:
        """Average replacement rate across all bands."""
        if not self.band_stats:
            return 1.0
        return float(np.mean([s.replacement_rate for s in self.band_stats]))

    @property
    def total_unresolved(self) -> int:
        return int(self.unresolved_mask.sum())

    def summary(self) -> Dict:
        return {
            "shape": self.reconstructed_image.shape,
            "total_cloud_fraction": round(self.total_cloud_fraction, 6),
            "overall_replacement_rate": round(self.overall_replacement_rate, 6),
            "total_unresolved_pixels": self.total_unresolved,
            "per_band": [s.to_dict() for s in self.band_stats],
        }


# ---------------------------------------------------------------------------
# Core reconstruction engine
# ---------------------------------------------------------------------------


class TemporalReconstruction:
    """Replaces cloudy pixels with cloud-free observations from adjacent dates.

    Priority: previous image > next image > unresolved.
    Both the cloud mask and optional nodata masks on temporal images
    determine which pixels are eligible for replacement.
    """

    SUPPORTED_EXTENSIONS = {".tif", ".tiff"}

    def __init__(self, config: Optional[ReconstructionConfig] = None) -> None:
        self.config = config or ReconstructionConfig()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def read_geotiff(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """Read a GeoTIFF and return data with metadata.

        Args:
            file_path: Path to a GeoTIFF file.

        Returns:
            Tuple of (data array (bands, H, W) float32, metadata dict).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a supported GeoTIFF.
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

    def save_geotiff(
        self,
        data: np.ndarray,
        output_path: Union[str, Path],
        transform: Optional[Affine] = None,
        crs: Optional[CRS] = None,
        nodata: Optional[float] = None,
    ) -> Path:
        """Save an array as a GeoTIFF preserving geospatial metadata.

        Args:
            data: Array with shape (bands, H, W) or (H, W).
            output_path: Destination file path.
            transform: Affine transform from source.
            crs: Coordinate reference system.
            nodata: Nodata value to encode.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_bands, height, width = data.shape

        profile = {
            "driver": "GTiff",
            "dtype": self.config.output_dtype,
            "width": width,
            "height": height,
            "count": n_bands,
            "crs": crs,
            "transform": transform,
            "compress": self.config.compress,
        }
        if nodata is not None:
            profile["nodata"] = nodata

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(n_bands):
                dst.write(data[i].astype(self.config.output_dtype), i + 1)

        logger.info(f"Saved → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Validity helpers
    # ------------------------------------------------------------------

    def _build_valid_mask(
        self,
        image: np.ndarray,
        nodata_value: Optional[float] = None,
    ) -> np.ndarray:
        """Build a boolean validity mask for a multi-band image.

        A pixel is valid if it is finite and (when nodata_value is set)
        not equal to the nodata sentinel.

        Args:
            image: (bands, H, W) float array.
            nodata_value: Optional nodata sentinel.

        Returns:
            Boolean (H, W) array — True where the pixel is valid in ALL bands.
        """
        valid = np.isfinite(image)
        if nodata_value is not None:
            valid = valid & (image != nodata_value)

        # A pixel is valid only when every band is valid
        return valid.all(axis=0)

    def _resolve_nodata_value(
        self,
        metadata: Dict,
        explicit: Optional[float] = None,
    ) -> Optional[float]:
        """Determine the effective nodata value.

        Priority: explicit config > file metadata.
        """
        if explicit is not None:
            return explicit
        return metadata.get("nodata")

    # ------------------------------------------------------------------
    # Core pixel-wise reconstruction
    # ------------------------------------------------------------------

    def _compute_band_stats(
        self,
        cloud_mask: np.ndarray,
        prev_valid: np.ndarray,
        next_valid: np.ndarray,
        replaced_from_previous: np.ndarray,
        replaced_from_next: np.ndarray,
        unresolved: np.ndarray,
    ) -> ReplacementStats:
        """Compute replacement statistics for one band.

        Args:
            cloud_mask: Boolean (H, W) — True where cloudy.
            prev_valid: Boolean (H, W) — True where previous is valid.
            next_valid: Boolean (H, W) — True where next is valid.
            replaced_from_previous: Boolean (H, W) — actually replaced by prev.
            replaced_from_next: Boolean (H, W) — actually replaced by next.
            unresolved: Boolean (H, W) — cloudy but unreplaced.

        Returns:
            ReplacementStats for the band.
        """
        total = cloud_mask.size
        cloudy = int(cloud_mask.sum())
        n_prev = int(replaced_from_previous.sum())
        n_next = int(replaced_from_next.sum())
        n_unresolved = int(unresolved.sum())

        return ReplacementStats(
            total_pixels=total,
            cloudy_pixels=cloudy,
            replaced_from_previous=n_prev,
            replaced_from_next=n_next,
            unresolved_pixels=n_unresolved,
        )

    def reconstruct(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        next_image: np.ndarray,
        cloud_mask: np.ndarray,
        prev_nodata: Optional[float] = None,
        next_nodata: Optional[float] = None,
    ) -> ReconstructionResult:
        """Reconstruct a cloud-free image from three temporal observations.

        For every cloudy pixel in the current image:
          1. Use the previous image if its pixel is valid.
          2. Else use the next image if its pixel is valid.
          3. Else mark as unresolved.

        Args:
            current: Cloudy reference image (bands, H, W) or (H, W).
            previous: Previous temporal image, same shape.
            next_image: Next temporal image, same shape.
            cloud_mask: Boolean (H, W) — True where cloud is present.
            prev_nodata: Nodata value for the previous image.
            next_nodata: Nodata value for the next image.

        Returns:
            ReconstructionResult with cloud-free image and statistics.

        Raises:
            ValueError: If image shapes are inconsistent.
        """
        # Ensure 3-D
        if current.ndim == 2:
            current = current[np.newaxis, ...]
        if previous.ndim == 2:
            previous = previous[np.newaxis, ...]
        if next_image.ndim == 2:
            next_image = next_image[np.newaxis, ...]

        if not (current.shape == previous.shape == next_image.shape):
            raise ValueError(
                f"Shape mismatch: current={current.shape}, "
                f"previous={previous.shape}, next={next_image.shape}"
            )

        if cloud_mask.shape != current.shape[1:]:
            raise ValueError(
                f"Cloud mask shape {cloud_mask.shape} does not match "
                f"spatial dimensions {current.shape[1:]}"
            )

        n_bands, height, width = current.shape
        cloud_bool = cloud_mask.astype(bool)

        # Validity masks for temporal images (per-pixel, all-band check)
        prev_valid = self._build_valid_mask(previous, prev_nodata)
        next_valid = self._build_valid_mask(next_image, next_nodata)

        # Cloud-free pixels in current stay unchanged
        clear_mask = ~cloud_bool

        # Output starts as a copy of the current image
        reconstructed = current.copy()
        unresolved_mask = np.zeros((height, width), dtype=bool)

        band_stats: List[ReplacementStats] = []

        for b in range(n_bands):
            # Cloudy + prev valid → replace from prev
            use_prev = cloud_bool & prev_valid
            # Cloudy + not prev valid + next valid → replace from next
            use_next = cloud_bool & (~prev_valid) & next_valid
            # Cloudy + neither valid → unresolved
            band_unresolved = cloud_bool & (~prev_valid) & (~next_valid)

            reconstructed[b][use_prev] = previous[b][use_prev]
            reconstructed[b][use_next] = next_image[b][use_next]
            # unresolved pixels keep the cloudy pixel value (cannot fix)

            unresolved_mask |= band_unresolved

            band_stats.append(
                self._compute_band_stats(
                    cloud_mask=cloud_bool,
                    prev_valid=prev_valid,
                    next_valid=next_valid,
                    replaced_from_previous=use_prev,
                    replaced_from_next=use_next,
                    unresolved=band_unresolved,
                )
            )

            logger.debug(
                f"Band {b}: cloudy={cloud_bool.sum()}, "
                f"prev={int(use_prev.sum())}, next={int(use_next.sum())}, "
                f"unresolved={int(band_unresolved.sum())}"
            )

        total_unresolved = int(unresolved_mask.sum())
        if total_unresolved > 0:
            logger.warning(
                f"{total_unresolved} pixels could not be resolved "
                f"from either temporal image"
            )

        return ReconstructionResult(
            reconstructed_image=reconstructed,
            unresolved_mask=unresolved_mask,
            band_stats=band_stats,
        )

    # ------------------------------------------------------------------
    # File-level reconstruction
    # ------------------------------------------------------------------

    def reconstruct_from_files(
        self,
        current_path: Union[str, Path],
        previous_path: Union[str, Path],
        next_path: Union[str, Path],
        cloud_mask: Union[np.ndarray, Path],
        output_path: Optional[Union[str, Path]] = None,
        unresolved_output_path: Optional[Union[str, Path]] = None,
    ) -> ReconstructionResult:
        """Reconstruct using GeoTIFF files on disk.

        Args:
            current_path: Path to the current (cloudy) GeoTIFF.
            previous_path: Path to the previous temporal GeoTIFF.
            next_path: Path to the next temporal GeoTIFF.
            cloud_mask: Boolean array or path to a single-band GeoTIFF.
            output_path: If given, save the cloud-free GeoTIFF.
            unresolved_output_path: If given, save the unresolved mask.

        Returns:
            ReconstructionResult with output and metadata.
        """
        current_data, current_meta = self.read_geotiff(current_path)
        prev_data, prev_meta = self.read_geotiff(previous_path)
        next_data, next_meta = self.read_geotiff(next_path)

        # Resolve cloud mask
        if isinstance(cloud_mask, (str, Path)):
            mask_path = Path(cloud_mask)
            with rasterio.open(mask_path, "r") as src:
                cm = src.read(1).astype(bool)
        else:
            cm = cloud_mask.astype(bool)

        prev_nodata = self._resolve_nodata_value(prev_meta, self.config.invalid_data_value)
        next_nodata = self._resolve_nodata_value(next_meta, self.config.invalid_data_value)

        result = self.reconstruct(
            current=current_data,
            previous=prev_data,
            next_image=next_data,
            cloud_mask=cm,
            prev_nodata=prev_nodata,
            next_nodata=next_nodata,
        )

        result.metadata = current_meta

        if output_path is not None:
            self.save_geotiff(
                result.reconstructed_image,
                output_path,
                transform=current_meta.get("transform"),
                crs=current_meta.get("crs"),
            )

        if unresolved_output_path is not None:
            mask_out = result.unresolved_mask.astype(np.uint8)
            self.save_geotiff(
                mask_out[np.newaxis, ...],
                unresolved_output_path,
                transform=current_meta.get("transform"),
                crs=current_meta.get("crs"),
                nodata=255,
            )

        return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def reconstruct_batch(
    current_dir: Union[str, Path],
    output_dir: Union[str, Path],
    cloud_mask_pattern: str = "*_cloud_mask.tif",
    prev_suffix: str = "_prev",
    next_suffix: str = "_next",
    config: Optional[ReconstructionConfig] = None,
) -> List[ReconstructionResult]:
    """Batch-reconstruct multiple scenes in a directory.

    Expects the following naming convention inside ``current_dir``:
      - ``<name>.tif`` — current cloudy image
      - ``<name>_cloud_mask.tif`` — corresponding binary cloud mask
      - ``<name>_prev.tif`` — previous temporal image
      - ``<name>_next.tif`` — next temporal image

    Args:
        current_dir: Directory containing input triplets.
        output_dir: Directory for reconstructed outputs.
        cloud_mask_pattern: Glob pattern matching cloud mask files.
        prev_suffix: Suffix identifying the previous image.
        next_suffix: Suffix identifying the next image.
        config: Reconstruction configuration.

    Returns:
        List of ReconstructionResult objects.
    """
    current_dir = Path(current_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = TemporalReconstruction(config)

    # Discover cloud mask files
    mask_files = sorted(current_dir.glob(cloud_mask_pattern))
    if not mask_files:
        logger.warning(f"No cloud masks matching '{cloud_mask_pattern}' in {current_dir}")
        return []

    results: List[ReconstructionResult] = []

    for mask_file in mask_files:
        stem = mask_file.stem.replace("_cloud_mask", "")
        current_file = current_dir / f"{stem}.tif"
        prev_file = current_dir / f"{stem}{prev_suffix}.tif"
        next_file = current_dir / f"{stem}{next_suffix}.tif"

        if not all(f.exists() for f in [current_file, prev_file, next_file]):
            logger.warning(f"Incomplete triplet for '{stem}', skipping")
            continue

        out_file = output_dir / f"{stem}_cloud_free.tif"
        unresolved_file = output_dir / f"{stem}_unresolved.tif"

        try:
            result = engine.reconstruct_from_files(
                current_path=current_file,
                previous_path=prev_file,
                next_path=next_file,
                cloud_mask=mask_file,
                output_path=out_file,
                unresolved_output_path=unresolved_file,
            )
            results.append(result)
            logger.info(
                f"Reconstructed {stem}: "
                f"cloud_fraction={result.total_cloud_fraction:.3f}, "
                f"replacement_rate={result.overall_replacement_rate:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed to reconstruct {stem}: {e}")

    if results:
        avg_rate = np.mean([r.overall_replacement_rate for r in results])
        logger.info(
            f"Batch complete: {len(results)}/{len(mask_files)} scenes, "
            f"mean replacement rate={avg_rate:.4f}"
        )

    return results
