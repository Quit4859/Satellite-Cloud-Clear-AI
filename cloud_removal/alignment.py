"""
alignment.py
============
Check whether two rasters share a grid, and reproject one onto the other's
grid if not.
"""
from .common import np, reproject, Resampling, Dict, Any


def check_alignment(cloudy_meta: Dict[str, Any], other_meta: Dict[str, Any]) -> bool:
    """Return True if two rasters share CRS, size and transform exactly."""
    same_crs = cloudy_meta["crs"] == other_meta["crs"]
    same_size = cloudy_meta["array"].shape[-2:] == other_meta["array"].shape[-2:]
    same_transform = cloudy_meta["transform"] == other_meta["transform"]
    return same_crs and same_size and same_transform


def reproject_to_match(source_meta: Dict[str, Any], target_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Reproject/resample `source_meta`'s array onto `target_meta`'s grid."""
    dst_crs = target_meta["crs"]
    dst_transform = target_meta["transform"]
    dst_height, dst_width = target_meta["array"].shape[-2:]
    n_bands = source_meta["array"].shape[0]

    dst_array = np.zeros((n_bands, dst_height, dst_width), dtype=source_meta["array"].dtype)
    for b in range(n_bands):
        reproject(
            source=source_meta["array"][b], destination=dst_array[b],
            src_transform=source_meta["transform"], src_crs=source_meta["crs"],
            dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.bilinear,
        )

    new_meta = dict(source_meta)
    new_meta["array"] = dst_array
    new_meta["crs"] = dst_crs
    new_meta["transform"] = dst_transform
    new_meta["resolution"] = target_meta["resolution"]
    profile = source_meta["profile"].copy()
    profile.update(crs=dst_crs, transform=dst_transform, width=dst_width, height=dst_height)
    new_meta["profile"] = profile
    return new_meta


def ensure_alignment(cloudy_meta: Dict[str, Any], other_meta: Dict[str, Any],
                      other_name: str, logger: "ProcessingLogger") -> Dict[str, Any]:
    """Verify `other_meta` is aligned with `cloudy_meta`; reproject if not."""
    if check_alignment(cloudy_meta, other_meta):
        logger.log(f"{other_name} is already aligned with the cloudy image.")
        return other_meta
    logger.log(f"{other_name} is NOT aligned with the cloudy image -- reprojecting/resampling.")
    aligned = reproject_to_match(other_meta, cloudy_meta)
    if aligned["array"].shape[-2:] != cloudy_meta["array"].shape[-2:]:
        raise ValueError(f"Failed to align {other_name}: shape mismatch persists after reprojection.")
    return aligned

