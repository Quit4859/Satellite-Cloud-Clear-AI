"""Visualization module for satellite imagery analysis.

Generates publication-quality previews: RGB composites, false-color
composites, cloud mask overlays, before/after comparisons, difference
images, histogram analysis, and temporal timelines.  Exports as PNG
or GeoTIFF.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio
from loguru import logger
from rasterio.crs import CRS
from rasterio.transform import Affine


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _normalize(arr: np.ndarray, percentile: Tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    """Percentile-stretch to [0, 1] for display."""
    low = np.percentile(arr, percentile[0])
    high = np.percentile(arr, percentile[1])
    if high - low < 1e-10:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert [0, 1] float to [0, 255] uint8."""
    return (arr * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# RGB & False Color
# ---------------------------------------------------------------------------


def create_rgb_preview(
    image: np.ndarray,
    bands: Tuple[int, int, int] = (2, 1, 0),
    output_path: Optional[Union[str, Path]] = None,
    title: str = "RGB Preview",
) -> np.ndarray:
    """Generate an RGB composite from a multi-band image.

    Args:
        image: (B, H, W) float array.
        bands: Indices for R, G, B channels.
        output_path: If given, save as PNG.
        title: Title for saved image.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    r = _normalize(image[bands[0]])
    g = _normalize(image[bands[1]])
    b = _normalize(image[bands[2]])
    rgb = np.stack([r, g, b], axis=-1)

    if output_path is not None:
        _save_png(_to_uint8(rgb), output_path, title)

    return rgb


def create_false_color(
    image: np.ndarray,
    bands: Tuple[int, int, int] = (7, 3, 2),
    output_path: Optional[Union[str, Path]] = None,
    title: str = "False Color",
) -> np.ndarray:
    """Generate a false-color composite (common for Sentinel-2 NIR-Red-Green).

    Args:
        image: (B, H, W) float array.
        bands: Channel indices for the false-color assignment.
        output_path: If given, save as PNG.
        title: Title.

    Returns:
        (H, W, 3) uint8 array.
    """
    n_bands = image.shape[0]
    actual = tuple(min(b, n_bands - 1) for b in bands)

    r = _normalize(image[actual[0]])
    g = _normalize(image[actual[1]])
    b = _normalize(image[actual[2]])
    fc = np.stack([r, g, b], axis=-1)

    if output_path is not None:
        _save_png(_to_uint8(fc), output_path, title)

    return fc


# ---------------------------------------------------------------------------
# Cloud mask overlay
# ---------------------------------------------------------------------------


def create_cloud_mask_overlay(
    image: np.ndarray,
    cloud_mask: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (255, 0, 0),
    title: str = "Cloud Mask Overlay",
) -> np.ndarray:
    """Overlay a cloud mask on top of an RGB composite.

    Args:
        image: (B, H, W) float array.
        cloud_mask: (H, W) boolean mask.
        output_path: If given, save as PNG.
        alpha: Overlay transparency.
        color: RGB color for cloud pixels.
        title: Title.

    Returns:
        (H, W, 3) uint8 RGB overlay.
    """
    if image.ndim == 3 and image.shape[0] >= 3:
        rgb = np.stack([
            _normalize(image[2]),
            _normalize(image[1]),
            _normalize(image[0]),
        ], axis=-1)
    elif image.ndim == 3:
        gray = _normalize(image[0])
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        gray = _normalize(image)
        rgb = np.stack([gray, gray, gray], axis=-1)

    overlay = (rgb * 255).astype(np.uint8)
    mask = cloud_mask.astype(bool)

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask,
            np.clip(overlay[:, :, c].astype(np.float32) * (1 - alpha) + color[c] * alpha, 0, 255).astype(np.uint8),
            overlay[:, :, c],
        )

    if output_path is not None:
        _save_png(overlay, output_path, title)

    return overlay


# ---------------------------------------------------------------------------
# Before / After comparison
# ---------------------------------------------------------------------------


def create_before_after(
    before: np.ndarray,
    after: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    titles: Tuple[str, str] = ("Before", "After"),
    title: str = "Before / After Comparison",
) -> np.ndarray:
    """Side-by-side before/after comparison.

    Args:
        before: Reference image (H, W) or (B, H, W).
        after: Reconstructed image.
        output_path: If given, save as PNG.
        titles: Subplot titles.
        title: Overall title.

    Returns:
        (H, 2*W+gap, 3) uint8 comparison image.
    """
    b = _to_uint8(_normalize(before[0] if before.ndim == 3 else before))
    a = _to_uint8(_normalize(after[0] if after.ndim == 3 else after))

    h, w = b.shape
    gap = 4
    canvas = np.ones((h, 2 * w + gap, 3), dtype=np.uint8) * 200
    canvas[:, :w] = np.stack([b, b, b], axis=-1)
    canvas[:, w + gap:] = np.stack([a, a, a], axis=-1)

    if output_path is not None:
        _save_png(canvas, output_path, title)

    return canvas


# ---------------------------------------------------------------------------
# Difference image
# ---------------------------------------------------------------------------


def create_difference_image(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    colormap: int = cv2.COLORMAP_JET,
    title: str = "Difference Image",
) -> np.ndarray:
    """Generate a heatmap of absolute differences.

    Args:
        reference: (H, W) or (B, H, W) ground truth.
        reconstructed: Same shape as reference.
        output_path: If given, save as PNG.
        colormap: OpenCV colormap constant.
        title: Title.

    Returns:
        (H, W, 3) uint8 heatmap.
    """
    ref = reference[0] if reference.ndim == 3 else reference
    rec = reconstructed[0] if reconstructed.ndim == 3 else reconstructed

    diff = np.abs(ref.astype(np.float64) - rec.astype(np.float64))
    diff_norm = _normalize(diff)
    diff_u8 = _to_uint8(diff_norm)
    heatmap = cv2.applyColorMap(diff_u8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if output_path is not None:
        _save_png(heatmap, output_path, title)

    return heatmap


# ---------------------------------------------------------------------------
# Histogram comparison
# ---------------------------------------------------------------------------


def create_histogram_comparison(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Histogram Comparison",
    bins: int = 256,
) -> None:
    """Plot overlapping histograms of reference vs reconstructed.

    Args:
        reference: (H, W) or (B, H, W) ground truth.
        reconstructed: Same shape.
        output_path: Destination PNG path.
        title: Plot title.
        bins: Number of histogram bins.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = reference[0] if reference.ndim == 3 else reference
    rec = reconstructed[0] if reconstructed.ndim == 3 else reconstructed

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ref.ravel(), bins=bins, alpha=0.6, label="Reference", color="steelblue", density=True)
    ax.hist(rec.ravel(), bins=bins, alpha=0.6, label="Reconstructed", color="coral", density=True)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Histogram saved → {output_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Temporal timeline
# ---------------------------------------------------------------------------


def create_temporal_timeline(
    images: List[np.ndarray],
    labels: List[str],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Temporal Timeline",
) -> None:
    """Create a horizontal strip of temporal images with labels.

    Args:
        images: List of (H, W) or (B, H, W) arrays.
        labels: Corresponding date / label strings.
        output_path: Destination PNG.
        title: Overall title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, img, lbl in zip(axes, images, labels):
        display = img[0] if img.ndim == 3 else img
        ax.imshow(_normalize(display), cmap="gray")
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Timeline saved → {output_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# GeoTIFF export
# ---------------------------------------------------------------------------


def export_geotiff(
    data: np.ndarray,
    output_path: Union[str, Path],
    transform: Optional[Affine] = None,
    crs: Optional[CRS] = None,
    dtype: str = "float32",
    compress: str = "lzw",
) -> Path:
    """Export a visualisation array as a GeoTIFF.

    Args:
        data: (bands, H, W) or (H, W) array.
        output_path: Destination .tif path.
        transform: Affine transform.
        crs: Coordinate reference system.
        dtype: Output data type.
        compress: Compression method.

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
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": n_bands,
        "crs": crs,
        "transform": transform,
        "compress": compress,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for i in range(n_bands):
            dst.write(data[i].astype(dtype), i + 1)

    logger.info(f"GeoTIFF exported → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_png(data: np.ndarray, path: Union[str, Path], title: str = "") -> None:
    """Write a uint8 RGB array as PNG with optional title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(data)
    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PNG saved → {path}")
