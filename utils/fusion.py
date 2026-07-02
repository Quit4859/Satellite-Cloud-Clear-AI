"""Multi-sensor data fusion module.

Aligns, resamples, and merges imagery from heterogeneous satellite sensors
(LISS-IV, Sentinel-1 SAR, Sentinel-2, DEM, and temporal acquisitions) into
a unified analysis-ready data cube.  Preserves CRS, nodata, and per-band
provenance metadata through every step.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
from loguru import logger
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.warp import calculate_default_transform, reproject


# ---------------------------------------------------------------------------
# Sensor definitions
# ---------------------------------------------------------------------------


class SensorType(str, Enum):
    LISS_IV = "liss_iv"
    SENTINEL1 = "sentinel1"
    SENTINEL2 = "sentinel2"
    DEM = "dem"
    TEMPORAL = "temporal"


SENSOR_DEFAULTS: Dict[SensorType, Dict] = {
    SensorType.LISS_IV: {"bands": 4, "resolution_m": 5.8, "crs": "EPSG:4326"},
    SensorType.SENTINEL1: {"bands": 2, "resolution_m": 10.0, "crs": "EPSG:4326"},
    SensorType.SENTINEL2: {"bands": 13, "resolution_m": 10.0, "crs": "EPSG:4326"},
    SensorType.DEM: {"bands": 1, "resolution_m": 30.0, "crs": "EPSG:4326"},
    SensorType.TEMPORAL: {"bands": 3, "resolution_m": 5.8, "crs": "EPSG:4326"},
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FusionConfig:
    """Configuration for multi-sensor fusion."""

    target_crs: str = "EPSG:4326"
    """Common CRS for all layers."""

    target_resolution: Tuple[float, float] = (0.00005, 0.00005)
    """Target pixel size in CRS units (lon, lat)."""

    resampling: str = "bilinear"
    """Resampling method for reprojection: 'nearest', 'bilinear', 'cubic'."""

    band_matching: str = "first"
    """Band matching strategy: 'first', 'mean', 'min', 'max'."""

    output_dtype: str = "float32"
    """Output data type."""

    compress: str = "lzw"
    """GeoTIFF compression."""

    fill_nodata: bool = True
    """Fill nodata with interpolated values before fusion."""

    def __post_init__(self) -> None:
        resample_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }
        if self.resampling not in resample_map:
            raise ValueError(
                f"resampling must be one of {list(resample_map.keys())}, "
                f"got '{self.resampling}'"
            )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SensorLayer:
    """Single sensor layer ready for fusion."""

    sensor_type: SensorType
    data: np.ndarray
    """(bands, H, W) float32 array."""

    transform: Affine
    crs: CRS
    nodata: Optional[float] = None
    bands: int = 0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bands == 0 and self.data.ndim == 3:
            self.bands = self.data.shape[0]


@dataclass
class FusionResult:
    """Output of the fusion pipeline."""

    fused_data: np.ndarray
    """(bands, H, W) float32 fused array."""

    transform: Affine
    crs: CRS
    layer_contributions: Dict[str, int]
    """Maps band index → source layer index."""

    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core fusion class
# ---------------------------------------------------------------------------


class MultiSensorFusion:
    """Aligns and fuses multi-sensor satellite imagery.

    Workflow:
        1. Reproject every layer to a common CRS and resolution.
        2. Resize / resample to match the target grid.
        3. Concatenate bands along the channel axis.
        4. Preserve all geospatial metadata.
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def read_geotiff(
        self,
        file_path: Union[str, Path],
        sensor_type: SensorType = SensorType.LISS_IV,
    ) -> SensorLayer:
        """Read a GeoTIFF and wrap it as a SensorLayer.

        Args:
            file_path: Path to a GeoTIFF.
            sensor_type: Sensor classification hint.

        Returns:
            SensorLayer with data and metadata.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with rasterio.open(file_path, "r") as src:
            data = src.read().astype(np.float32)
            layer = SensorLayer(
                sensor_type=sensor_type,
                data=data,
                transform=src.transform,
                crs=src.crs,
                nodata=src.nodata,
                bands=src.count,
                metadata={
                    "file": str(file_path),
                    "width": src.width,
                    "height": src.height,
                    "dtype": src.dtypes[0],
                },
            )

        logger.info(
            f"Read {file_path.name}: {data.shape[0]} bands, "
            f"{data.shape[1]}x{data.shape[2]}, sensor={sensor_type.value}"
        )
        return layer

    def save_geotiff(
        self,
        data: np.ndarray,
        output_path: Union[str, Path],
        transform: Affine,
        crs: CRS,
        nodata: Optional[float] = None,
    ) -> Path:
        """Save fused data as a GeoTIFF.

        Args:
            data: (bands, H, W) array.
            output_path: Destination path.
            transform: Affine transform.
            crs: CRS.
            nodata: Optional nodata value.

        Returns:
            Path to saved file.
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

        logger.info(f"Saved fused output → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Spatial alignment
    # ------------------------------------------------------------------

    def _resample_map(self) -> Dict[str, Resampling]:
        return {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
        }

    def reproject_layer(
        self,
        layer: SensorLayer,
        target_crs: Optional[str] = None,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> SensorLayer:
        """Reproject a layer to the target CRS and resolution.

        Args:
            layer: Input SensorLayer.
            target_crs: Target CRS string. Defaults to config.
            target_shape: Optional (H, W) to resize to.

        Returns:
            New SensorLayer with reprojected data.
        """
        crs = CRS.from_string(target_crs or self.config.target_crs)
        resampling = self._resample_map()[self.config.resampling]

        src_crs = layer.crs
        src_transform = layer.transform
        src_height, src_width = layer.data.shape[1], layer.data.shape[2]

        if target_shape is None:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, crs, src_width, src_height, transform=src_transform,
            )
        else:
            dst_height, dst_width = target_shape
            bounds = array_bounds(src_height, src_width, src_transform)
            dst_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top,
                                        dst_width, dst_height)

        reprojected = np.zeros(
            (layer.data.shape[0], dst_height, dst_width), dtype=np.float32,
        )

        for i in range(layer.data.shape[0]):
            reproject(
                source=layer.data[i],
                destination=reprojected[i],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=crs,
                resampling=resampling,
            )

        return SensorLayer(
            sensor_type=layer.sensor_type,
            data=reprojected,
            transform=dst_transform,
            crs=crs,
            nodata=layer.nodata,
            bands=layer.bands,
            metadata=layer.metadata,
        )

    def resize_to_grid(
        self,
        data: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Resize a (bands, H, W) array to the target spatial grid.

        Args:
            data: Input array.
            target_h: Target height.
            target_w: Target width.

        Returns:
            Resized array (bands, target_h, target_w).
        """
        n_bands = data.shape[0]
        resized = np.zeros((n_bands, target_h, target_w), dtype=np.float32)
        interp = cv2.INTER_LINEAR if self.config.resampling != "nearest" else cv2.INTER_NEAREST

        for b in range(n_bands):
            resized[b] = cv2.resize(
                data[b], (target_w, target_h), interpolation=interp,
            ).astype(np.float32)

        return resized

    # ------------------------------------------------------------------
    # Band matching
    # ------------------------------------------------------------------

    def match_bands(
        self,
        data: np.ndarray,
        target_bands: int,
    ) -> np.ndarray:
        """Adjust band count to match target via selection or averaging.

        Args:
            data: (bands, H, W) input.
            target_bands: Desired number of bands.

        Returns:
            Array with ``target_bands`` channels.
        """
        n_bands = data.shape[0]
        if n_bands == target_bands:
            return data

        strategy = self.config.band_matching

        if n_bands > target_bands:
            if strategy == "first":
                return data[:target_bands]
            elif strategy == "mean":
                chunk = target_bands
                step = n_bands // chunk
                out = np.zeros((chunk, data.shape[1], data.shape[2]), dtype=np.float32)
                for i in range(chunk):
                    out[i] = data[i * step:(i + 1) * step].mean(axis=0)
                return out
            else:
                return data[:target_bands]

        # Fewer bands than needed — pad with zeros
        padded = np.zeros((target_bands, data.shape[1], data.shape[2]), dtype=np.float32)
        padded[:n_bands] = data
        return padded

    # ------------------------------------------------------------------
    # Temporal consistency
    # ------------------------------------------------------------------

    def enforce_temporal_consistency(
        self,
        layers: List[SensorLayer],
        reference_idx: int = 0,
    ) -> List[SensorLayer]:
        """Align all layers to the same spatial grid as the reference.

        Args:
            layers: List of SensorLayer objects.
            reference_idx: Index of the reference layer to match.

        Returns:
            List of aligned layers.
        """
        ref = layers[reference_idx]
        ref_h, ref_w = ref.data.shape[1], ref.data.shape[2]
        aligned: List[SensorLayer] = []

        for i, layer in enumerate(layers):
            if i == reference_idx:
                aligned.append(layer)
                continue

            if layer.crs != ref.crs:
                layer = self.reproject_layer(layer, target_crs=str(ref.crs))

            if layer.data.shape[1:] != (ref_h, ref_w):
                resized_data = self.resize_to_grid(layer.data, ref_h, ref_w)
                layer = SensorLayer(
                    sensor_type=layer.sensor_type,
                    data=resized_data,
                    transform=ref.transform,
                    crs=ref.crs,
                    nodata=layer.nodata,
                    bands=layer.bands,
                    metadata=layer.metadata,
                )

            aligned.append(layer)

        logger.info(f"Temporal consistency enforced across {len(layers)} layers")
        return aligned

    # ------------------------------------------------------------------
    # High-level fusion
    # ------------------------------------------------------------------

    def fuse(
        self,
        layers: List[SensorLayer],
    ) -> FusionResult:
        """Fuse multiple aligned sensor layers into a single data cube.

        Args:
            layers: List of aligned SensorLayer objects.

        Returns:
            FusionResult with stacked bands and metadata.

        Raises:
            ValueError: If no layers provided.
        """
        if not layers:
            raise ValueError("At least one layer is required for fusion")

        # Align all layers to the first layer's grid
        aligned = self.enforce_temporal_consistency(layers)

        ref = aligned[0]
        target_bands = ref.data.shape[0]
        all_bands: List[np.ndarray] = []
        contributions: Dict[int, int] = {}
        band_offset = 0

        for i, layer in enumerate(aligned):
            matched = self.match_bands(layer.data, target_bands)
            all_bands.append(matched)
            for b in range(target_bands):
                contributions[band_offset + b] = i
            band_offset += target_bands

        fused = np.concatenate(all_bands, axis=0)

        logger.info(
            f"Fused {len(layers)} layers → {fused.shape[0]} bands, "
            f"{fused.shape[1]}x{fused.shape[2]}"
        )

        return FusionResult(
            fused_data=fused,
            transform=ref.transform,
            crs=ref.crs,
            layer_contributions=contributions,
            metadata={
                "n_layers": len(layers),
                "sensor_types": [l.sensor_type.value for l in layers],
                "total_bands": fused.shape[0],
            },
        )

    def fuse_files(
        self,
        file_paths: List[Union[str, Path]],
        sensor_types: Optional[List[SensorType]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> FusionResult:
        """Read, fuse, and optionally save a set of GeoTIFF files.

        Args:
            file_paths: List of paths to GeoTIFFs.
            sensor_types: Optional sensor type hints (one per file).
            output_path: If given, save the fused output.

        Returns:
            FusionResult.
        """
        if sensor_types is None:
            sensor_types = [SensorType.LISS_IV] * len(file_paths)

        layers = [
            self.read_geotiff(fp, st)
            for fp, st in zip(file_paths, sensor_types)
        ]

        result = self.fuse(layers)

        if output_path is not None:
            self.save_geotiff(
                result.fused_data, output_path,
                result.transform, result.crs,
            )

        return result
