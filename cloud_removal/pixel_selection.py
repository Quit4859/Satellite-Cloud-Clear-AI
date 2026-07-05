"""
pixel_selection.py
====================
Selects replacement pixels for cloudy areas via weighted-median temporal
fusion across the registered temporal stack.
"""
from .common import np, Tuple, Dict, Any, List
from .color_matching import local_mean_std


def extract_cloud_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of `image` with every non-cloud pixel zeroed out."""
    out = np.zeros_like(image)
    out[:, mask] = image[:, mask]
    return out


def weighted_median_select(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel weighted median across the stack axis (axis 0), fully
    vectorized via `argsort` + cumulative weights -- no per-pixel Python loop.

    Args:
        values: (N, H, W) stacked band values.
        weights: (N, H, W) per-pixel, per-layer weights (0 = unavailable).

    Returns:
        (result (H, W), no_data (H, W) bool -- True where every layer's
        weight was ~0, i.e. nothing usable was found for that pixel).
    """
    order = np.argsort(values, axis=0)
    sorted_vals = np.take_along_axis(values, order, axis=0)
    sorted_weights = np.take_along_axis(weights, order, axis=0)
    cum_weights = np.cumsum(sorted_weights, axis=0)
    total_weights = cum_weights[-1]
    cutoff = 0.5 * total_weights
    reached = cum_weights >= cutoff[np.newaxis, ...]
    idx = np.argmax(reached, axis=0)
    result = np.take_along_axis(sorted_vals, idx[np.newaxis, ...], axis=0)[0]
    no_data = total_weights <= 1e-9
    return result, no_data


def select_temporal_replacement(cloudy_array: np.ndarray, layers: List[Dict[str, Any]],
                                 cloud_mask: np.ndarray, config: "Config"
                                 ) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """Fill every cloudy pixel with a weighted median drawn from every CLEAR
    observation of that pixel across the temporal stack (uploaded reference
    + historical Sentinel-2 imagery), preferring layers that are closer in
    date AND locally more spectrally similar to Image 1's own clear-sky
    neighborhood (so a poorly-matching layer is naturally down-weighted
    even if its date-distance weight is high).

    Bands beyond `config.stack_band_count` (which the historical download
    doesn't supply) fall back to the single highest-weight clear layer, or
    the original cloudy pixel if nothing is available.

    IMPORTANT: `cloudy_array` must already be in the SAME value domain as
    every `layer["array"]` (both are normalized reflectance in [0, 1] --
    see `normalize_image`). Mixing raw (0-10000) and normalized (0-1)
    arrays here silently crushes every replaced pixel toward zero.

    Returns:
        (composite, stats, unresolved_mask) where `unresolved_mask` is
        True for every cloud pixel that had NO usable observation in any
        layer for ANY of the first `stack_band_count` bands -- these are
        the pixels later offered to the optional AI-inpainting fallback.
    """
    n_bands, height, width = cloudy_array.shape
    stack_bands = min(config.stack_band_count, n_bands)
    composite = cloudy_array.astype(np.float32).copy()

    stats = {"layers_used": len(layers), "stack_bands": stack_bands, "unfilled_pixels": 0}
    if not layers:
        unresolved_mask = cloud_mask.copy()
        stats["unfilled_pixels"] = int(unresolved_mask.sum())
        return composite, stats, unresolved_mask

    clear_overall = ~cloud_mask
    target_local_mean = [
        local_mean_std(cloudy_array[b].astype(np.float32), clear_overall.astype(np.float32), config.local_color_window)[0]
        for b in range(stack_bands)
    ]

    any_no_data = np.ones((height, width), dtype=bool)
    for b in range(stack_bands):
        values = np.zeros((len(layers), height, width), dtype=np.float32)
        weights = np.zeros((len(layers), height, width), dtype=np.float32)
        for i, layer in enumerate(layers):
            layer_bands = layer["array"].shape[0]
            band_arr = layer["array"][b] if layer_bands > b else layer["array"][0]
            diff = np.abs(band_arr - target_local_mean[b])
            similarity = np.exp(-(diff ** 2) / (2 * (config.spectral_similarity_sigma ** 2)))
            weights[i] = layer["weight"] * similarity * layer["clear_mask"].astype(np.float32)
            values[i] = band_arr
        band_result, no_data = weighted_median_select(values, weights)
        any_no_data &= no_data
        band_out = composite[b]
        fillable = cloud_mask & ~no_data
        band_out[fillable] = band_result[fillable]
        composite[b] = band_out

    # Extra bands the historical stack doesn't cover: fall back to the
    # single best (highest-weight, clear-at-that-pixel) layer's value.
    if n_bands > stack_bands and layers:
        best_layer_idx = int(np.argmax([l["weight"] for l in layers]))
        best_layer = layers[best_layer_idx]
        for b in range(stack_bands, n_bands):
            source_band = best_layer["array"][b] if best_layer["array"].shape[0] > b else best_layer["array"][-1]
            band_out = composite[b]
            fillable = cloud_mask & best_layer["clear_mask"]
            band_out[fillable] = source_band[fillable]
            composite[b] = band_out

    unresolved_mask = cloud_mask & any_no_data
    stats["unfilled_pixels"] = int(unresolved_mask.sum())
    return composite, stats, unresolved_mask
