"""
export.py
==========
Final export: write the cloud-free GeoTIFF (preserving CRS/metadata) plus
a quick-look PNG.
"""
from .common import np, cv2, rasterio, Dict, Any


def save_geotiff(path: str, array: np.ndarray, profile: Dict[str, Any],
                  compression: str = "LZW") -> str:
    """Write `array` to `path` as a GeoTIFF, preserving CRS/transform/dtype
    and every other field already present in `profile`."""
    out_profile = profile.copy()
    out_profile.update(count=array.shape[0], dtype=array.dtype, compress=compression,
                        height=array.shape[1], width=array.shape[2])
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array)
    return path


def save_final_png(path: str, array: np.ndarray) -> str:
    """Save a 3-band (or single-band) quick-look PNG of the final result."""
    arr = array.astype(np.float32)
    rgb = arr[:3] if arr.shape[0] >= 3 else np.repeat(arr[:1], 3, axis=0)
    rgb = np.moveaxis(rgb, 0, -1)
    vmax = np.nanpercentile(rgb, 99) or 1.0
    rgb_u8 = (np.clip(rgb / vmax, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    return path
