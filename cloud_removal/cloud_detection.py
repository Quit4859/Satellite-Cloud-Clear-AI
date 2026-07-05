"""
cloud_detection.py
===================
Cloud probability / mask generation: SCL classification, s2cloudless
(core/thin/adaptive), plus border and halo detection, all merged into one
final pre-shadow cloud mask.
"""
from .common import np, cv2, ndi, Optional, Tuple, Dict, Any


def generate_scl_mask(scl_array: Optional[np.ndarray],
                       cloud_classes: Tuple[int, ...] = (8, 9, 10, 11)) -> Optional[np.ndarray]:
    """Boolean mask marking every pixel whose SCL class is in `cloud_classes`."""
    if scl_array is None:
        return None
    scl = scl_array[0] if scl_array.ndim == 3 else scl_array
    return np.isin(scl, cloud_classes).astype(bool)

# =============================================================================
# STEP 4 / 12 (part 2) -- s2cloudless: core / thin / adaptive probability bands
# =============================================================================
class S2CloudlessProcessor:
    """Thin wrapper around `s2cloudless.S2PixelCloudDetector`.

    Expects reflectance for bands
    [B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12] scaled to [0, 1]. If
    the raster does not have exactly those 10 bands, a best-effort
    subset/pad keeps the notebook running end-to-end regardless.
    """

    REQUIRED_BANDS = 10

    def __init__(self, threshold: float = 0.22) -> None:
        self.threshold = threshold
        self._detector = None
        try:
            from s2cloudless import S2PixelCloudDetector
            self._detector = S2PixelCloudDetector(
                threshold=threshold, average_over=4, dilation_size=2, all_bands=False
            )
        except Exception as exc:
            print(f"  \u26a0 Could not initialize s2cloudless detector: {exc}")

    def _prepare_bands(self, normalized_array: np.ndarray) -> np.ndarray:
        bands, h, w = normalized_array.shape
        if bands >= self.REQUIRED_BANDS:
            selected = normalized_array[: self.REQUIRED_BANDS]
        else:
            pad = np.repeat(normalized_array[-1:], self.REQUIRED_BANDS - bands, axis=0)
            selected = np.concatenate([normalized_array, pad], axis=0)
        stack = np.moveaxis(selected, 0, -1)
        return stack[np.newaxis, ...].astype(np.float32)

    def generate_probability(self, normalized_array: np.ndarray) -> np.ndarray:
        if self._detector is None:
            print("  \u26a0 s2cloudless unavailable -- returning zero probability map.")
            return np.zeros(normalized_array.shape[1:], dtype=np.float32)
        bands = normalized_array.shape[0]
        if bands < self.REQUIRED_BANDS:
            # s2cloudless's model was trained on 10 specific real Sentinel-2
            # bands (B01,B02,B04,B05,B08,B8A,B09,B10,B11,B12 -- notably
            # including the SWIR/cirrus channels). Fabricating the missing
            # ones by repeating whatever band we do have (e.g. NIR) feeds
            # the model spectral data it was never trained on and reliably
            # inflates false-positive cloud probability over large, entirely
            # clear areas -- which is what was replacing most of the image
            # instead of just the actual clouds. With only `bands` band(s)
            # available, skip s2cloudless entirely and rely on the SCL
            # classification (if supplied) plus border/halo detection.
            print(f"  \u26a0 s2cloudless needs {self.REQUIRED_BANDS} real Sentinel-2 bands; only "
                  f"{bands} were provided. Skipping s2cloudless for this run (would otherwise "
                  f"fabricate data and over-flag clear areas as cloud) -- relying on the SCL "
                  f"classification instead.")
            return np.zeros(normalized_array.shape[1:], dtype=np.float32)
        batch = self._prepare_bands(normalized_array)
        return self._detector.get_cloud_probability_maps(batch)[0].astype(np.float32)


def _fallback_bright_cloud_mask(normalized_array: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Conservative last-resort cloud heuristic used ONLY when s2cloudless
    was skipped (fewer than its required 10 bands) AND no SCL classification
    was supplied. Flags pixels that are both very bright and unusually flat
    across bands (low per-pixel band spread) -- the two properties that
    reliably separate cloud/haze from bright rooftops or pavement, which
    tend to have more inter-band contrast. Deliberately conservative (high
    percentile) since a false positive here silently replaces real,
    up-to-date scene content -- the exact problem this fallback exists to
    avoid.
    """
    luminance = normalized_array.mean(axis=0)
    spread = normalized_array.std(axis=0)
    bright_thresh = np.percentile(luminance, percentile)
    flat_thresh = np.percentile(spread, 100 - percentile) if percentile < 100 else spread.max()
    return (luminance >= bright_thresh) & (spread <= flat_thresh)


def generate_cloud_masks(normalized_array: np.ndarray, config: "Config",
                          scl_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Derive CORE / THIN / ADAPTIVE cloud masks from one s2cloudless run.

    A much lower default threshold plus a permissive "thin" band directly
    addresses "only the brightest part of the cloud is removed" and "thin
    clouds are still visible" -- and the adaptive band catches faint cloud
    sitting over a locally bright or dark background that a single global
    threshold would miss.
    """
    processor = S2CloudlessProcessor(threshold=config.cloud_prob_threshold)
    bands_available = normalized_array.shape[0]
    s2cloudless_usable = processor._detector is not None and bands_available >= S2CloudlessProcessor.REQUIRED_BANDS
    prob = processor.generate_probability(normalized_array)

    if not s2cloudless_usable and scl_mask is None:
        # s2cloudless can't be trusted here (missing real 10-band data, or
        # the library itself isn't installed) AND there's no SCL
        # classification to rely on instead -- without SOME fallback, cloud
        # detection would silently find nothing at all. Only invoked in
        # this specific "nothing else available" case, so it never adds
        # noise on top of an already-reliable SCL mask.
        prob = _fallback_bright_cloud_mask(normalized_array).astype(np.float32)
        print("  \u2139 Using conservative brightness-based fallback for cloud detection "
              "(no SCL supplied and s2cloudless needs real 10-band data).")

    core_mask = prob >= config.cloud_prob_threshold
    thin_mask = (prob >= config.thin_cloud_threshold) & ~core_mask

    adaptive_mask = np.zeros_like(core_mask, dtype=bool)
    if config.use_adaptive_threshold:
        block = config.adaptive_block_size | 1
        local_mean = ndi.uniform_filter(prob, size=block)
        adaptive_mask = (prob >= (local_mean + config.adaptive_offset)) & ~core_mask
        thin_mask = thin_mask | adaptive_mask

    return {
        "probability": prob, "core_mask": core_mask, "thin_mask": thin_mask,
        "adaptive_mask": adaptive_mask, "s2_mask": core_mask | thin_mask,
    }

# =============================================================================
# STEP 4 / 12 (part 3) -- Combine SCL + s2cloudless, detect borders / halo
# =============================================================================
def combine_masks(scl_mask: Optional[np.ndarray], s2cloudless_mask: np.ndarray) -> np.ndarray:
    """SCL OR s2cloudless -- s2cloudless alone if no SCL was supplied."""
    if scl_mask is None:
        return s2cloudless_mask.astype(bool)
    if scl_mask.shape != s2cloudless_mask.shape:
        raise ValueError(f"Mask shape mismatch: SCL {scl_mask.shape} vs s2cloudless {s2cloudless_mask.shape}")
    return np.logical_or(scl_mask, s2cloudless_mask)


def detect_cloud_borders(mask: np.ndarray, ring_px: int = 3) -> np.ndarray:
    """Thin ring around every cloud blob (morphological gradient) -- the
    semi-transparent transition pixels a hard mask usually misses."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return (dilated - eroded).astype(bool)


def detect_cloud_halo(normalized_array: np.ndarray, mask: np.ndarray,
                       search_px: int = 10, brightness_margin: float = 0.05) -> np.ndarray:
    """Bright white haze/fog halo just outside the current mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * search_px + 1, 2 * search_px + 1))
    search_ring = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask
    luminance = normalized_array.mean(axis=0)
    local_bg = ndi.uniform_filter(luminance, size=max(search_px * 4, 21))
    return search_ring & (luminance > (local_bg + brightness_margin))


def merge_all_cloud_masks(scl_mask: Optional[np.ndarray], s2_masks: Dict[str, np.ndarray],
                           normalized_array: np.ndarray, config: "Config") -> Dict[str, np.ndarray]:
    """Final (pre-shadow) mask = SCL OR core OR thin OR adaptive OR border OR halo."""
    base = combine_masks(scl_mask, s2_masks["s2_mask"])
    border = detect_cloud_borders(base, config.border_ring_px)
    halo = detect_cloud_halo(normalized_array, base, config.halo_search_px, config.halo_brightness_margin)
    merged = base | border | halo
    return {"base_mask": base, "border_mask": border, "halo_mask": halo, "merged_mask": merged}
