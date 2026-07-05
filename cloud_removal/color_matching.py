"""
color_matching.py
==================
Local color normalization: Reinhard-style local mean/std transfer for
temporal-stack layers, plus per-blob histogram matching and a final local
brightness correction on the composited result.
"""
from .common import np, ndi, exposure, Optional, Tuple, Dict, Any, List


def local_mean_std(array: np.ndarray, weight: Optional[np.ndarray] = None,
                    window: int = 41) -> Tuple[np.ndarray, np.ndarray]:
    """Windowed local mean/std of `array`, optionally weighted (e.g. only
    over "clear" pixels). Fully vectorized via `ndimage.uniform_filter`."""
    if weight is None:
        weight = np.ones_like(array, dtype=np.float32)
    else:
        weight = weight.astype(np.float32)
    wsum = np.clip(ndi.uniform_filter(weight, size=window), 1e-6, None)
    mean = ndi.uniform_filter(array * weight, size=window) / wsum
    mean_sq = ndi.uniform_filter((array ** 2) * weight, size=window) / wsum
    variance = np.clip(mean_sq - mean ** 2, 1e-6, None)
    return mean, np.sqrt(variance)


def local_color_transfer(source: np.ndarray, target: np.ndarray, target_valid_mask: np.ndarray,
                          window: int = 41) -> np.ndarray:
    """Locally rescale `source`'s brightness/contrast, band by band, to
    match `target`'s tone in the same neighborhood -- computed only from
    `target`'s valid (clear) pixels. This is what stops a historical /
    reference pixel from "looking copied from another image": before it's
    ever used to fill a cloud gap, its tone is already pulled to match
    Image 1's surrounding clear-sky statistics.
    """
    adjusted = source.astype(np.float32).copy()
    n_bands = min(source.shape[0], target.shape[0])
    for b in range(n_bands):
        target_mean, target_std = local_mean_std(target[b].astype(np.float32), target_valid_mask.astype(np.float32), window)
        source_mean, source_std = local_mean_std(source[b].astype(np.float32), None, window)
        ratio = target_std / np.clip(source_std, 1e-6, None)
        adjusted[b] = (source[b].astype(np.float32) - source_mean) * ratio + target_mean
    return adjusted


def normalize_stack_layers(cloudy_array: np.ndarray, layers: List[Dict[str, Any]],
                            cloud_mask: np.ndarray, config: "Config") -> List[Dict[str, Any]]:
    """Apply `local_color_transfer` to every layer in the temporal stack so
    each one's tone already matches Image 1 before compositing/selection.
    """
    clear = ~cloud_mask
    normalized_layers = []
    for layer in layers:
        adjusted = local_color_transfer(layer["array"], cloudy_array, clear, config.local_color_window)
        new_layer = dict(layer)
        new_layer["array"] = adjusted
        normalized_layers.append(new_layer)
    return normalized_layers


def histogram_match_replaced(result: np.ndarray, cloudy_source: np.ndarray,
                              full_mask: np.ndarray, margin: int = 40) -> np.ndarray:
    """Histogram-match the replaced pixels of EACH cloud blob individually
    against the clear pixels immediately surrounding that specific blob --
    not the whole image and not a single global statistic.
    """
    matched = result.copy()
    n_bands = result.shape[0]
    labeled, n_blobs = ndi.label(full_mask)
    if n_blobs == 0:
        return matched

    objects = ndi.find_objects(labeled)
    height, width = full_mask.shape
    for blob_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        y0, y1 = max(slc[0].start - margin, 0), min(slc[0].stop + margin, height)
        x0, x1 = max(slc[1].start - margin, 0), min(slc[1].stop + margin, width)
        region_mask = (labeled[y0:y1, x0:x1] == blob_id)
        clear_local = ~full_mask[y0:y1, x0:x1]
        if clear_local.sum() < 20 or not region_mask.any():
            continue
        for b in range(n_bands):
            crop_result = matched[b, y0:y1, x0:x1]
            crop_cloudy = cloudy_source[b, y0:y1, x0:x1]
            if crop_cloudy[clear_local].size < 20:
                continue
            matched_crop = exposure.match_histograms(crop_result, crop_cloudy, channel_axis=None)
            crop_result[region_mask] = matched_crop[region_mask]
            matched[b, y0:y1, x0:x1] = crop_result
    return matched


def local_brightness_correction(result: np.ndarray, cloudy_source: np.ndarray,
                                 mask: np.ndarray, window: int = 41) -> np.ndarray:
    """Final brightness-only nudge toward the local clear-sky mean, mopping
    up any residual tonal offset the coarser per-blob match left behind."""
    corrected = result.astype(np.float32).copy()
    clear_weight = (~mask).astype(np.float32)
    for b in range(result.shape[0]):
        target_mean, _ = local_mean_std(cloudy_source[b].astype(np.float32), clear_weight, window)
        current_mean, _ = local_mean_std(result[b].astype(np.float32), mask.astype(np.float32), window)
        delta = target_mean - current_mean
        band = corrected[b]
        band[mask] = band[mask] + delta[mask]
        corrected[b] = band
    return corrected.astype(result.dtype)
