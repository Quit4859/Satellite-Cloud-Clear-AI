"""
registration.py
================
Downloads and registers historical Sentinel-2 scenes onto the cloudy
image's exact grid: coarse reproject, then ORB-feature + RANSAC
sub-pixel refinement.
"""
from .common import (
    os, np, cv2, rasterio, reproject, Resampling, pc, tqdm, Optional, Tuple, Dict, Any, List,
)
from .normalization import normalize_image, denormalize_image

HISTORICAL_BAND_ASSETS = ["B04", "B03", "B02", "B08"]  # Red, Green, Blue, NIR


def _signed_href(item: Any, asset_key: str) -> Optional[str]:
    """Return a time-limited, signed URL for a STAC asset (Planetary
    Computer requires this for private/rate-limited blob storage)."""
    try:
        asset = item.assets[asset_key]
        return pc.sign(asset.href)
    except Exception:
        return None


def to_gray_u8(band_stack: np.ndarray) -> np.ndarray:
    """Convert a (bands,H,W) array to an 8-bit brightness proxy for feature
    matching (ORB works on 8-bit grayscale images)."""
    proxy = band_stack[:3].mean(axis=0) if band_stack.shape[0] >= 3 else band_stack[0]
    proxy = np.nan_to_num(proxy)
    vmax = np.nanpercentile(proxy, 99) or 1.0
    proxy = np.clip(proxy / vmax, 0, 1) * 255
    return proxy.astype(np.uint8)


def refine_alignment_orb(target_array: np.ndarray, moving_array: np.ndarray,
                          config: "Config") -> Tuple[np.ndarray, Optional[float]]:
    """Sub-pixel registration refinement via ORB feature matching + RANSAC
    homography estimation.

    `target_array` and `moving_array` are assumed already coarsely aligned
    to the same grid/CRS/resolution (by `reproject`); this corrects any
    *residual* pixel-level misalignment from differing orbit geometry or
    terrain parallax. Falls back to the coarse-aligned array unchanged if
    not enough reliable matches are found.

    Returns:
        (refined_array, mean_inlier_reprojection_error_px_or_None)
    """
    try:
        fixed_gray = to_gray_u8(target_array)
        moving_gray = to_gray_u8(moving_array)

        orb = cv2.ORB_create(nfeatures=config.orb_features)
        kp1, des1 = orb.detectAndCompute(fixed_gray, None)
        kp2, des2 = orb.detectAndCompute(moving_gray, None)
        if des1 is None or des2 is None or len(kp1) < config.min_match_count or len(kp2) < config.min_match_count:
            return moving_array, None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
        if len(good) < config.min_match_count:
            return moving_array, None

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, config.ransac_reproj_threshold
        )
        if homography is None:
            return moving_array, None

        h, w = fixed_gray.shape
        warped_bands = [
            cv2.warpPerspective(band.astype(np.float32), homography, (w, h), flags=cv2.INTER_LINEAR)
            for band in moving_array
        ]
        warped = np.stack(warped_bands, axis=0)

        reg_error = None
        if inlier_mask is not None and inlier_mask.sum() > 0:
            keep = inlier_mask.ravel().astype(bool)
            projected = cv2.perspectiveTransform(src_pts[keep], homography)
            residual = projected - dst_pts[keep]
            reg_error = float(np.sqrt((residual ** 2).sum(axis=2)).mean())

        return warped, reg_error
    except Exception as exc:
        print(f"  \u26a0 ORB/RANSAC refinement skipped: {exc}")
        return moving_array, None


def download_and_register_item(item: Any, cloudy_meta: Dict[str, Any], config: "Config",
                                 output_mgr: "OutputManager", save_preview: bool
                                 ) -> Optional[Dict[str, Any]]:
    """Stream the required bands of one STAC item, reproject them onto the
    cloudy image's exact grid (CRS + resolution + affine transform), then
    refine the alignment with ORB+RANSAC. Returns the aligned array + an
    SCL-derived clear-sky mask, or None if the item couldn't be read.
    """
    dst_crs = cloudy_meta["crs"]
    dst_transform = cloudy_meta["transform"]
    height, width = cloudy_meta["array"].shape[-2:]

    band_arrays = []
    for band_key in HISTORICAL_BAND_ASSETS:
        href = _signed_href(item, band_key)
        if href is None:
            return None
        try:
            with rasterio.open(href) as src:
                dst = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1), destination=dst,
                    src_crs=src.crs, src_transform=src.transform,
                    dst_crs=dst_crs, dst_transform=dst_transform,
                    resampling=Resampling.bilinear,
                )
        except Exception as exc:
            print(f"  \u26a0 Failed to read/reproject {band_key} of {item.id}: {exc}")
            return None
        band_arrays.append(dst)
    coarse_array = np.stack(band_arrays, axis=0)  # (4, H, W): B04,B03,B02,B08

    clear_mask = None
    scl_href = _signed_href(item, "SCL")
    if scl_href is not None:
        try:
            with rasterio.open(scl_href) as src:
                scl_dst = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1), destination=scl_dst,
                    src_crs=src.crs, src_transform=src.transform,
                    dst_crs=dst_crs, dst_transform=dst_transform,
                    resampling=Resampling.nearest,
                )
            not_clear_classes = tuple(config.scl_cloud_classes) + (config.scl_shadow_class,)
            clear_mask = ~np.isin(scl_dst.astype(np.uint8), not_clear_classes)
        except Exception as exc:
            print(f"  \u26a0 Failed to read SCL for {item.id}: {exc}")

    normalized_target = normalize_image(cloudy_meta["array"])
    normalized_coarse = normalize_image(coarse_array, src_max=10000.0)
    aligned_array, reg_error = refine_alignment_orb(normalized_target, normalized_coarse, config)
    aligned_array = denormalize_image(aligned_array, coarse_array.dtype, dst_max=10000.0)

    if save_preview:
        output_mgr.save_png(os.path.join("04_registered_images", f"{item.id}.png"), aligned_array[:3])
        output_mgr.save_png(os.path.join("03_downloaded_images", f"{item.id}_raw.png"), coarse_array[:3])

    return {
        "id": item.id,
        "datetime": item.datetime,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "array": normalize_image(aligned_array, src_max=10000.0),
        "clear_mask": clear_mask,
        "registration_error": reg_error,
    }


def download_and_register_all(items: List[Any], cloudy_meta: Dict[str, Any], config: "Config",
                                output_mgr: "OutputManager", logger: "ProcessingLogger"
                                ) -> List[Dict[str, Any]]:
    """Run `download_and_register_item` for every candidate STAC item."""
    results = []
    for i, item in enumerate(tqdm(items, desc="Registering historical images", unit="img")):
        res = download_and_register_item(item, cloudy_meta, config, output_mgr, save_preview=(i < 5))
        if res is None:
            logger.log(f"  \u26a0 Skipped {item.id} (could not read/register).")
            continue
        logger.record_downloaded_image(res["id"], res["datetime"], res["cloud_cover"], res["registration_error"])
        results.append(res)
    logger.log(f"Successfully registered {len(results)} / {len(items)} historical image(s).")
    return results
