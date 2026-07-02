"""Evaluation metrics and reporting for cloud removal.

Computes pixel-level quality metrics (PSNR, SSIM, MAE, RMSE) alongside
domain-specific measures (cloud coverage, replacement rate, unresolved
percentage).  Exports results as JSON, CSV, and visual comparison images.
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ImageMetrics:
    """Pixel-level quality metrics between reference and reconstructed."""

    psnr: float
    """Peak Signal-to-Noise Ratio (dB). Higher is better."""

    ssim: float
    """Structural Similarity Index in [-1, 1]. Higher is better."""

    mae: float
    """Mean Absolute Error. Lower is better."""

    rmse: float
    """Root Mean Square Error. Lower is better."""

    def to_dict(self) -> Dict[str, float]:
        return {"psnr": round(self.psnr, 4), "ssim": round(self.ssim, 6),
                "mae": round(self.mae, 6), "rmse": round(self.rmse, 6)}


@dataclass
class CloudMetrics:
    """Domain-specific cloud removal metrics."""

    original_cloud_coverage: float
    """Fraction of cloudy pixels in the input (0.0–1.0)."""

    resolved_coverage: float
    """Fraction of originally cloudy pixels that were replaced."""

    replacement_percentage: float
    """Percentage of cloudy pixels successfully replaced."""

    unresolved_percentage: float
    """Percentage of cloudy pixels that could not be fixed."""

    unresolved_count: int
    """Absolute count of unresolved pixels."""

    total_cloudy: int
    """Total number of cloudy pixels in the mask."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_cloud_coverage": round(self.original_cloud_coverage, 6),
            "resolved_coverage": round(self.resolved_coverage, 6),
            "replacement_percentage": round(self.replacement_percentage, 4),
            "unresolved_percentage": round(self.unresolved_percentage, 4),
            "unresolved_count": self.unresolved_count,
            "total_cloudy": self.total_cloudy,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation output for a single scene."""

    image_name: str
    image_metrics: ImageMetrics
    cloud_metrics: Optional[CloudMetrics] = None
    per_band_metrics: Dict[int, ImageMetrics] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "image_name": self.image_name,
            "image_metrics": self.image_metrics.to_dict(),
            "per_band": {str(k): v.to_dict() for k, v in self.per_band_metrics.items()},
            "metadata": self.metadata,
        }
        if self.cloud_metrics is not None:
            d["cloud_metrics"] = self.cloud_metrics.to_dict()
        return d


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _to_float64(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64)


def compute_psnr(reference: np.ndarray, reconstructed: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio.

    Args:
        reference: Ground-truth image.
        reconstructed: Predicted image.
        maximum pixel value (assumes [0, max_val]).

    Returns:
        PSNR in dB.  Returns np.inf for identical images.
    """
    ref = _to_float64(reference)
    rec = _to_float64(reconstructed)
    mse = float(np.mean((ref - rec) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / mse))


def compute_ssim(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    window_size: int = 11,
) -> float:
    """Structural Similarity Index (simplified single-channel).

    Uses a sliding-window approach without scikit-image dependency.

    Args:
        reference: 2-D ground-truth.
        reconstructed: 2-D prediction.
        window_size: Local window size.

    Returns:
        SSIM value in [-1, 1].
    """
    ref = _to_float64(reference)
    rec = _to_float64(reconstructed)

    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    k = cv2.getGaussianKernel(window_size, 1.5)
    window = np.outer(k, k)

    mu_r = cv2.filter2D(ref, -1, window)
    mu_c = cv2.filter2D(rec, -1, window)

    mu_r2 = mu_r ** 2
    mu_c2 = mu_c ** 2
    mu_rc = mu_r * mu_c

    sigma_r2 = cv2.filter2D(ref ** 2, -1, window) - mu_r2
    sigma_c2 = cv2.filter2D(rec ** 2, -1, window) - mu_c2
    sigma_rc = cv2.filter2D(ref * rec, -1, window) - mu_rc

    ssim_map = ((2 * mu_rc + C1) * (2 * sigma_rc + C2)) / \
               ((mu_r2 + mu_c2 + C1) * (sigma_r2 + sigma_c2 + C2))

    return float(np.mean(ssim_map))


def compute_mae(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(reference.astype(np.float64) - reconstructed.astype(np.float64))))


def compute_rmse(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((reference.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)))


def compute_cloud_metrics(
    cloud_mask: np.ndarray,
    unresolved_mask: Optional[np.ndarray] = None,
    original: Optional[np.ndarray] = None,
    reconstructed: Optional[np.ndarray] = None,
) -> CloudMetrics:
    """Compute domain-specific cloud removal metrics.

    Args:
        cloud_mask: Boolean (H, W) — True where cloud.
        unresolved_mask: Boolean (H, W) — True where unresolved.
        original: Original cloudy image (unused, for API symmetry).
        reconstructed: Reconstructed image (unused, for API symmetry).

    Returns:
        CloudMetrics instance.
    """
    mask = cloud_mask.astype(bool)
    total = int(mask.sum())

    unresolved = np.zeros_like(mask) if unresolved_mask is None else unresolved_mask.astype(bool)
    n_unresolved = int((mask & unresolved).sum())

    n_resolved = total - n_unresolved
    img_area = mask.size

    return CloudMetrics(
        original_cloud_coverage=total / img_area if img_area > 0 else 0.0,
        resolved_coverage=n_resolved / total if total > 0 else 1.0,
        replacement_percentage=(n_resolved / total * 100.0) if total > 0 else 100.0,
        unresolved_percentage=(n_unresolved / total * 100.0) if total > 0 else 0.0,
        unresolved_count=n_unresolved,
        total_cloudy=total,
    )


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def evaluate(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    cloud_mask: Optional[np.ndarray] = None,
    unresolved_mask: Optional[np.ndarray] = None,
    image_name: str = "unnamed",
) -> EvaluationReport:
    """Run all metrics and produce a full evaluation report.

    Args:
        reference: Ground-truth cloud-free image (B, H, W) or (H, W).
        reconstructed: Predicted cloud-free image, same shape.
        cloud_mask: Optional boolean (H, W) cloud mask.
        unresolved_mask: Optional boolean (H, W) unresolved mask.
        image_name: Name tag for the report.

    Returns:
        EvaluationReport with all computed metrics.
    """
    if reference.ndim == 2:
        reference = reference[np.newaxis, ...]
    if reconstructed.ndim == 2:
        reconstructed = reconstructed[np.newaxis, ...]

    # Global metrics (use first band)
    ref_b0 = reference[0]
    rec_b0 = reconstructed[0]
    global_metrics = ImageMetrics(
        psnr=compute_psnr(ref_b0, rec_b0),
        ssim=compute_ssim(ref_b0, rec_b0),
        mae=compute_mae(ref_b0, rec_b0),
        rmse=compute_rmse(ref_b0, rec_b0),
    )

    # Per-band
    per_band = {}
    for b in range(reference.shape[0]):
        per_band[b] = ImageMetrics(
            psnr=compute_psnr(reference[b], reconstructed[b]),
            ssim=compute_ssim(reference[b], reconstructed[b]),
            mae=compute_mae(reference[b], reconstructed[b]),
            rmse=compute_rmse(reference[b], reconstructed[b]),
        )

    # Cloud metrics
    cloud_m = None
    if cloud_mask is not None:
        cloud_m = compute_cloud_metrics(cloud_mask, unresolved_mask)

    report = EvaluationReport(
        image_name=image_name,
        image_metrics=global_metrics,
        cloud_metrics=cloud_m,
        per_band_metrics=per_band,
        metadata={"shape": list(reference.shape)},
    )

    logger.info(
        f"Evaluated {image_name}: PSNR={global_metrics.psnr:.2f}, "
        f"SSIM={global_metrics.ssim:.4f}, RMSE={global_metrics.rmse:.4f}"
    )
    return report


# ---------------------------------------------------------------------------
# Report export
# ---------------------------------------------------------------------------


def save_json_report(report: EvaluationReport, output_path: Union[str, Path]) -> Path:
    """Save an evaluation report as JSON.

    Args:
        report: EvaluationReport to serialise.
        output_path: Destination file path.

    Returns:
        Path to saved JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    logger.info(f"JSON report saved → {output_path}")
    return output_path


def save_csv_report(
    reports: List[EvaluationReport],
    output_path: Union[str, Path],
) -> Path:
    """Export a list of evaluation reports as a CSV table.

    Args:
        reports: List of EvaluationReport objects.
        output_path: Destination CSV path.

    Returns:
        Path to saved CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in reports:
        row = {"image_name": r.image_name}
        row.update(r.image_metrics.to_dict())
        if r.cloud_metrics is not None:
            row.update(r.cloud_metrics.to_dict())
        rows.append(row)

    if not rows:
        return output_path

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"CSV report saved → {output_path}")
    return output_path


def save_comparison_image(
    reference: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Union[str, Path],
    title: str = "Before / After",
) -> Path:
    """Create a side-by-side visual comparison PNG.

    Args:
        reference: Ground-truth (H, W) or (B, H, W).
        reconstructed: Prediction, same shape.
        output_path: Destination PNG.
        title: Plot title.

    Returns:
        Path to saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if reference.ndim == 3:
        reference = reference[0]
    if reconstructed.ndim == 3:
        reconstructed = reconstructed[0]

    vmin = min(reference.min(), reconstructed.min())
    vmax = max(reference.max(), reconstructed.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(reference, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Reference")
    axes[0].axis("off")

    axes[1].imshow(reconstructed, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    diff = np.abs(reference - reconstructed)
    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Difference")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Comparison image saved → {output_path}")
    return output_path
