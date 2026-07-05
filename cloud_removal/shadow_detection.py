"""
shadow_detection.py
====================
Cloud-shadow detection: SCL shadow class + dark-object detection +
sun-geometry projection from the cloud mask.
"""
from .common import np, cv2, ndi, Optional, Tuple


def estimate_shadow_offset(sun_azimuth_deg: Optional[float], sun_elevation_deg: Optional[float],
                            cloud_height_px: float, default: Tuple[int, int] = (15, -15)
                            ) -> Tuple[int, int]:
    """Estimate the (dy, dx) offset from a cloud to its shadow using sun
    geometry if known, else fall back to a fixed default direction."""
    if sun_azimuth_deg is None or sun_elevation_deg is None:
        return default
    elevation_rad = np.radians(max(sun_elevation_deg, 1.0))
    azimuth_rad = np.radians(sun_azimuth_deg)
    distance = cloud_height_px / np.tan(elevation_rad)
    dy = int(round(-distance * np.cos(azimuth_rad)))
    dx = int(round(-distance * np.sin(azimuth_rad)))
    return dy, dx


def detect_dark_objects(normalized_array: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    """Dark-object detection: pixels darker than `percentile` of scene luminance."""
    luminance = normalized_array.mean(axis=0)
    return luminance <= np.percentile(luminance, percentile)


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))
    if dy > 0: shifted[:dy, :] = False
    elif dy < 0: shifted[dy:, :] = False
    if dx > 0: shifted[:, :dx] = False
    elif dx < 0: shifted[:, dx:] = False
    return shifted


def detect_shadows(normalized_array: np.ndarray, scl_array: Optional[np.ndarray],
                    cloud_mask: np.ndarray, config: "Config") -> np.ndarray:
    """Combine SCL class 3, dark-object detection, cloud-geometry / sun-angle
    projection, and neighborhood analysis into one shadow mask, then grow
    it slightly before it's merged with the cloud mask.
    """
    scl_shadow = np.zeros_like(cloud_mask, dtype=bool)
    if scl_array is not None:
        scl = scl_array[0] if scl_array.ndim == 3 else scl_array
        scl_shadow = (scl == config.scl_shadow_class)

    dark = detect_dark_objects(normalized_array, config.shadow_dark_percentile)
    dy, dx = estimate_shadow_offset(config.sun_azimuth_deg, config.sun_elevation_deg, config.assumed_cloud_height_px)
    geometric_projection = _shift_mask(cloud_mask, dy, dx) & ~cloud_mask

    # A shadow must lie in the sun-projected direction from an actual cloud.
    # (Bug fix: this used to also blanket-include every "dark" pixel within
    # `shadow_neighbor_px` of ANY cloud regardless of direction, which -- on
    # a normal city scene full of naturally dark rivers/rooftops/tree
    # canopy near clouds -- flagged roughly `shadow_dark_percentile`% of the
    # ENTIRE image as "shadow", not just the true cast shadows. That's what
    # was replacing most of the picture with old reference-image color
    # instead of just the actual cloud/shadow area.)
    near_cloud_tight = ndi.distance_transform_edt(~cloud_mask) <= max(abs(dy), abs(dx)) + 5
    estimated_shadow = dark & geometric_projection & near_cloud_tight
    shadow = scl_shadow | estimated_shadow

    if config.shadow_grow_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (2 * config.shadow_grow_px + 1, 2 * config.shadow_grow_px + 1))
        shadow = cv2.dilate(shadow.astype(np.uint8), kernel, iterations=1).astype(bool)

    return shadow & ~cloud_mask
