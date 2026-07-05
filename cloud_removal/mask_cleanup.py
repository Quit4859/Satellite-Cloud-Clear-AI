"""
mask_cleanup.py
================
Denoise (open/close/remove-small/fill-holes) and then adaptively expand a
boolean cloud mask (bigger clouds grow more, via a distance transform).
"""
from .common import np, cv2, ndi, remove_small_objects, remove_small_holes, Optional


def clean_mask(mask: np.ndarray, kernel_size: int = 5, min_object_size: int = 40,
                min_hole_size: int = 64, debug: bool = False,
                output_mgr: Optional["OutputManager"] = None) -> np.ndarray:
    """Denoise a boolean cloud mask WITHOUT shrinking it back down.

    Sequence: opening -> closing -> remove small objects -> fill small holes.
    Deliberately no final erode (a dilate-then-erode would cancel itself
    out and re-introduce the "mask too tight" problem this update fixes).
    `min_object_size` is kept small so genuinely small/isolated clouds
    survive -- only single-pixel noise is removed.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = mask.astype(np.uint8)

    def _save(name: str, arr: np.ndarray) -> None:
        if debug and output_mgr is not None:
            output_mgr.save_png(f"debug_clean_{name}.png", arr.astype(bool))

    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    _save("a_opened", opened)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    _save("b_closed", closed)
    no_small_objects = remove_small_objects(closed.astype(bool), min_size=min_object_size)
    _save("c_no_small_objects", no_small_objects)
    filled = remove_small_holes(no_small_objects, area_threshold=min_hole_size)
    _save("d_filled_holes", filled)
    return filled.astype(bool)


def expand_mask_adaptive(mask: np.ndarray, config: "Config", debug: bool = False,
                          output_mgr: Optional["OutputManager"] = None) -> np.ndarray:
    """Grow the mask by an amount that scales with each cloud's own size,
    using a distance transform (vectorized -- no per-pixel Python loop).

    For every connected blob: radius = clip(sqrt(area/pi) * dilation_scale,
    dilation_min_px, dilation_max_px). A pixel just outside the mask joins
    the expanded mask if it's within *its nearest blob's* target radius.
    """
    labeled, n_components = ndi.label(mask)
    if n_components == 0:
        return mask.copy()

    sizes = ndi.sum(np.ones_like(mask, dtype=np.float32), labeled, index=np.arange(1, n_components + 1))
    radii = np.clip(np.sqrt(sizes / np.pi) * config.dilation_scale,
                     config.dilation_min_px, config.dilation_max_px)
    radius_lookup = np.concatenate([[0.0], radii])

    dist, nearest_idx = ndi.distance_transform_edt(~mask, return_indices=True)
    nearest_label = labeled[tuple(nearest_idx)]
    nearest_radius = radius_lookup[nearest_label]
    expanded = mask | ((dist <= nearest_radius) & (nearest_label > 0))

    if debug and output_mgr is not None:
        output_mgr.save_png("debug_expand_radius_map.png", radius_lookup[labeled], cmap="viridis")
    return expanded
