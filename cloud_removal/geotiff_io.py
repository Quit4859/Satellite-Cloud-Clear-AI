"""
geotiff_io.py
=============
Basic GeoTIFF reading helper.
"""
from .common import os, rasterio, RasterioIOError, Dict, Any


def read_geotiff(path: str) -> Dict[str, Any]:
    """Read a GeoTIFF and return its array plus full spatial metadata.

    Raises:
        FileNotFoundError / RasterioIOError / ValueError on bad input.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"GeoTIFF not found: {path}")
    try:
        with rasterio.open(path) as src:
            if src.count == 0:
                raise ValueError(f"GeoTIFF has no bands: {path}")
            array = src.read()
            profile = src.profile.copy()
            crs = src.crs
            transform = src.transform
            resolution = src.res
            tags = src.tags()
    except RasterioIOError as exc:
        raise RasterioIOError(f"Corrupted or unreadable GeoTIFF ({path}): {exc}")

    return {
        "array": array, "profile": profile, "crs": crs, "transform": transform,
        "resolution": resolution, "meta": tags, "path": path,
    }

