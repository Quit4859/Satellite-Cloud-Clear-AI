"""Production-ready satellite image preprocessing module.

Reads GeoTIFF images via Rasterio, preserves all geospatial metadata,
handles NoData values, normalizes to [0,1], resizes to configurable
dimensions, and saves processed outputs with full metadata retention.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from loguru import logger


@dataclass
class PreprocessingConfig:
    """Configuration for satellite image preprocessing."""

    target_size: Tuple[int, int] = (512, 512)
    normalize: bool = True
    nodata_value: Optional[float] = None
    clip_percentile: Optional[float] = 2.0
    dtype: str = "float32"
    resampling: Resampling = Resampling.bilinear
    compress: str = "lzw"

    def __post_init__(self) -> None:
        h, w = self.target_size
        if h <= 0 or w <= 0:
            raise ValueError(f"target_size must be positive, got {self.target_size}")


@dataclass
class PreprocessingResult:
    """Result of preprocessing a single image."""

    data: np.ndarray
    transform: Affine
    crs: Optional[CRS]
    metadata: Dict
    original_shape: Tuple[int, ...]
    nodata_mask: Optional[np.ndarray] = None

    @property
    def bands(self) -> int:
        return self.data.shape[0] if self.data.ndim == 3 else 1

    @property
    def height(self) -> int:
        return self.data.shape[-2]

    @property
    def width(self) -> int:
        return self.data.shape[-1]


class SatellitePreprocessor:
    """Reads, preprocesses, and saves GeoTIFF satellite images.

    Supports RGB and multispectral imagery from any sensor. Handles NoData
    values, normalizes pixel values, resizes to target dimensions, and
    preserves all geospatial metadata through the processing pipeline.
    """

    SUPPORTED_EXTENSIONS = {".tif", ".tiff"}

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()

    def read_image(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """Read a GeoTIFF and return data with full metadata.

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
                "band_descriptions": [src.descriptions[i] or f"band_{i+1}" for i in range(src.count)],
            }

        logger.info(
            f"Read {file_path.name}: {data.shape[0]} bands, "
            f"{data.shape[1]}x{data.shape[2]}, dtype={data.dtype}"
        )
        return data, metadata

    def handle_nodata(
        self,
        data: np.ndarray,
        nodata_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Replace NoData values with NaN and produce a validity mask.

        Args:
            data: Input array (bands, H, W) or (H, W).
            nodata_value: Explicit NoData sentinel. If None, uses the
                          value from metadata or treats NaN as the mask.

        Returns:
            Tuple of (cleaned data with NoData replaced by NaN,
                      boolean mask where True = valid pixel).
        """
        mask = np.ones(data.shape, dtype=bool)

        if nodata_value is not None:
            mask = data != nodata_value
            data = data.astype(np.float32)
            data[~mask] = np.nan
        elif np.issubdtype(data.dtype, np.floating):
            mask = ~np.isnan(data)

        return data, mask

    def clip_outliers(
        self,
        data: np.ndarray,
        percentile: float = 2.0,
    ) -> np.ndarray:
        """Clip pixel values to a percentile range to remove outliers.

        Only operates on valid (non-NaN) pixels.

        Args:
            data: Input array.
            percentile: Symmetric percentile to clip at (e.g., 2.0 clips
                        at 2nd and 98th percentiles).

        Returns:
            Clipped array.
        """
        valid = data[~np.isnan(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
        if valid.size == 0:
            return data

        low = np.percentile(valid, percentile)
        high = np.percentile(valid, 100 - percentile)

        result = data.copy()
        if np.issubdtype(result.dtype, np.floating):
            result = np.where(np.isnan(result), np.nan, np.clip(result, low, high))
        else:
            result = np.clip(result, low, high)
        return result

    def normalize_minmax(self, data: np.ndarray) -> np.ndarray:
        """Normalize valid pixels to [0, 1] using min-max scaling.

        NaN pixels remain NaN after normalization.

        Args:
            data: Input array.

        Returns:
            Normalized array in [0, 1].
        """
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)

        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return data

        data_min, data_max = valid.min(), valid.max()
        if data_max - data_min < 1e-10:
            logger.warning("Constant data encountered; returning zeros")
            return np.where(np.isnan(data), np.nan, 0.0)

        normalized = (data - data_min) / (data_max - data_min)
        return normalized

    def resize_image(
        self,
        data: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Resize image bands to target (height, width).

        Uses the configured resampling method. NaN pixels are preserved
        using nearest-neighbor masking after interpolation.

        Args:
            data: Array with shape (bands, H, W).
            target_size: (height, width) override; uses config if None.

        Returns:
            Resized array with shape (bands, target_h, target_w).
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required for resizing")

        target_h, target_w = target_size or self.config.target_size
        n_bands = data.shape[0] if data.ndim == 3 else 1
        resized = np.empty((n_bands, target_h, target_w), dtype=data.dtype)

        interp_map = {
            Resampling.nearest: cv2.INTER_NEAREST,
            Resampling.bilinear: cv2.INTER_LINEAR,
            Resampling.cubic: cv2.INTER_CUBIC,
        }
        interp_flag = interp_map.get(self.config.resampling, cv2.INTER_LINEAR)

        for i in range(n_bands):
            band = data[i] if data.ndim == 3 else data
            has_nan = np.issubdtype(band.dtype, np.floating) and np.any(np.isnan(band))

            if has_nan:
                valid_mask = ~np.isnan(band)
                band_clean = np.where(valid_mask, band, 0.0).astype(np.float32)
                resized_clean = cv2.resize(band_clean, (target_w, target_h), interpolation=interp_flag)
                resized_mask = cv2.resize(
                    valid_mask.astype(np.float32), (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                resized[i] = np.where(resized_mask > 0.5, resized_clean, np.nan)
            else:
                resized[i] = cv2.resize(band, (target_w, target_h), interpolation=interp_flag)

        return resized

    def save_image(
        self,
        data: np.ndarray,
        output_path: Union[str, Path],
        transform: Affine,
        crs: Optional[CRS],
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Save a processed array as a GeoTIFF.

        Preserves CRS, transform, and writes with LZW compression.
        NaN values are stored as the raster nodata value.

        Args:
            data: Array with shape (bands, H, W).
            output_path: Destination file path.
            transform: Affine transform (adjusted for resize if needed).
            crs: Coordinate reference system.
            metadata: Optional extra metadata to store as tags.

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
            "dtype": self.config.dtype,
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
                band = data[i].astype(self.config.dtype)
                dst.write(band, i + 1)

            if metadata:
                for key in ("band_descriptions",):
                    if key in metadata and isinstance(metadata[key], list):
                        for idx, desc in enumerate(metadata[key]):
                            dst.set_band_description(idx + 1, str(desc))

        logger.info(f"Saved {output_path.name}: {n_bands} bands, {height}x{width}")
        return output_path

    def _compute_resized_transform(
        self,
        original_transform: Affine,
        original_shape: Tuple[int, int],
        target_shape: Tuple[int, int],
    ) -> Affine:
        """Compute the affine transform after resizing.

        Args:
            original_transform: The source affine transform.
            original_shape: (original_height, original_width).
            target_shape: (target_height, target_width).

        Returns:
            Adjusted affine transform.
        """
        orig_h, orig_w = original_shape
        tgt_h, tgt_w = target_shape
        scale_x = orig_w / tgt_w
        scale_y = orig_h / tgt_h
        return Affine(
            original_transform.a * scale_x,
            original_transform.b,
            original_transform.c,
            original_transform.d,
            original_transform.e * scale_y,
            original_transform.f,
        )

    def preprocess(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        bands: Optional[List[int]] = None,
    ) -> PreprocessingResult:
        """Full preprocessing pipeline: read → NoData → normalize → resize → save.

        Args:
            file_path: Input GeoTIFF path.
            output_path: Where to save the result. If None, returns
                         in-memory result only.
            bands: 1-based band indices to select (None = all).

        Returns:
            PreprocessingResult with processed data and metadata.
        """
        data, metadata = self.read_image(file_path)

        if bands is not None:
            data = data[np.array(bands) - 1]
            metadata["count"] = data.shape[0]
            if "band_descriptions" in metadata:
                metadata["band_descriptions"] = [
                    metadata["band_descriptions"][b - 1] for b in bands
                ]

        original_shape = data.shape

        data, nodata_mask = self.handle_nodata(
            data, nodata_value=self.config.nodata_value or metadata.get("nodata")
        )

        if self.config.clip_percentile is not None:
            data = self.clip_outliers(data, percentile=self.config.clip_percentile)

        if self.config.normalize:
            data = self.normalize_minmax(data)

        resized = self.resize_image(data)

        resized_transform = self._compute_resized_transform(
            metadata["transform"],
            (metadata["height"], metadata["width"]),
            (self.config.target_size[0], self.config.target_size[1]),
        )

        result = PreprocessingResult(
            data=resized,
            transform=resized_transform,
            crs=metadata["crs"],
            metadata=metadata,
            original_shape=original_shape,
            nodata_mask=nodata_mask,
        )

        if output_path is not None:
            self.save_image(
                data=resized,
                output_path=output_path,
                transform=resized_transform,
                crs=metadata["crs"],
                metadata=metadata,
            )

        return result

    def preprocess_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.tif",
        bands: Optional[List[int]] = None,
    ) -> List[Path]:
        """Batch preprocess all matching GeoTIFFs in a folder.

        Args:
            input_dir: Directory containing input images.
            output_dir: Directory for processed outputs.
            pattern: Glob pattern to match input files.
            bands: 1-based band indices to select (None = all).

        Returns:
            List of paths to successfully saved outputs.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob(pattern))
        if not files:
            logger.warning(f"No files matching '{pattern}' in {input_dir}")
            return []

        logger.info(f"Batch preprocessing {len(files)} files from {input_dir}")
        saved: List[Path] = []

        for file_path in files:
            try:
                out_path = output_dir / f"processed_{file_path.name}"
                self.preprocess(file_path, output_path=out_path, bands=bands)
                saved.append(out_path)
            except Exception as e:
                logger.error(f"Failed to preprocess {file_path.name}: {e}")

        logger.info(f"Batch complete: {len(saved)}/{len(files)} files processed")
        return saved

    def get_image_info(self, file_path: Union[str, Path]) -> Dict:
        """Return a summary dict of a GeoTIFF without loading pixel data.

        Args:
            file_path: Path to a GeoTIFF.

        Returns:
            Dictionary with shape, band count, CRS, resolution, nodata, etc.
        """
        file_path = Path(file_path)
        with rasterio.open(file_path, "r") as src:
            pixel_size_x = abs(src.transform.a)
            pixel_size_y = abs(src.transform.e)
            return {
                "file": file_path.name,
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": src.dtypes[0],
                "crs": str(src.crs) if src.crs else None,
                "pixel_size": (pixel_size_x, pixel_size_y),
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "nodata": src.nodata,
                "compress": src.compression,
            }


def preprocess_single(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    **kwargs,
) -> PreprocessingResult:
    """Convenience function to preprocess a single GeoTIFF.

    Args:
        input_path: Source GeoTIFF.
        output_path: Destination path.
        target_size: (height, width) for the output.
        **kwargs: Additional PreprocessingConfig fields.

    Returns:
        PreprocessingResult with processed data.
    """
    config = PreprocessingConfig(target_size=target_size, **kwargs)
    processor = SatellitePreprocessor(config)
    return processor.preprocess(input_path, output_path=output_path)


def preprocess_folder(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    **kwargs,
) -> List[Path]:
    """Convenience function to batch-preprocess a folder of GeoTIFFs.

    Args:
        input_dir: Directory with source GeoTIFFs.
        output_dir: Directory for outputs.
        target_size: (height, width) for outputs.
        **kwargs: Additional PreprocessingConfig fields.

    Returns:
        List of paths to saved outputs.
    """
    config = PreprocessingConfig(target_size=target_size, **kwargs)
    processor = SatellitePreprocessor(config)
    return processor.preprocess_batch(input_dir, output_dir)
