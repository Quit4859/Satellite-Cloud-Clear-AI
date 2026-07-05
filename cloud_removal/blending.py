"""
blending.py
============
Seamless edge blending: Laplacian-pyramid multi-band blending (with a
Gaussian-feather fallback) plus optional OpenCV Poisson blending.
"""
from .common import np, cv2, ndi, filters, List


def feather_blend(matched: np.ndarray, cloudy: np.ndarray, mask: np.ndarray,
                   feather_radius: int = 15) -> np.ndarray:
    """Distance-transform Gaussian feathering -- robust fallback for `blend_edges`."""
    mask_f = mask.astype(np.float32)
    dist_inside = ndi.distance_transform_edt(mask_f)
    dist_outside = ndi.distance_transform_edt(1 - mask_f)
    alpha = np.zeros_like(mask_f)
    alpha[mask_f == 1] = np.clip(dist_inside[mask_f == 1] / feather_radius, 0, 1)
    alpha[mask_f == 0] = np.clip(1 - dist_outside[mask_f == 0] / feather_radius, 0, 1)
    alpha = filters.gaussian(alpha, sigma=feather_radius / 3.0)
    alpha3 = np.broadcast_to(alpha, matched.shape)
    blended = alpha3 * matched.astype(np.float32) + (1 - alpha3) * cloudy.astype(np.float32)
    return blended.astype(matched.dtype)


def _gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    pyr = [img.astype(np.float32)]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    gp = _gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        size = (gp[i].shape[1], gp[i].shape[0])
        expanded = cv2.pyrUp(gp[i + 1], dstsize=size)
        lp.append(gp[i] - expanded)
    lp.append(gp[-1])
    return lp


def _laplacian_pyramid_blend_band(band_a: np.ndarray, band_b: np.ndarray,
                                   mask: np.ndarray, levels: int = 5) -> np.ndarray:
    """Burt & Adelson multi-band blend: low frequencies blend over a wide
    area, high-frequency detail blends over a narrow one, so the seam
    disappears even across high-contrast boundaries."""
    la = _laplacian_pyramid(band_a, levels)
    lb = _laplacian_pyramid(band_b, levels)
    gm = _gaussian_pyramid(mask, levels)
    blended_pyr = []
    for i in range(levels + 1):
        m = gm[i]
        if m.shape != la[i].shape:
            m = cv2.resize(m, (la[i].shape[1], la[i].shape[0]))
        blended_pyr.append(la[i] * m + lb[i] * (1 - m))
    img = blended_pyr[-1]
    for i in range(levels - 1, -1, -1):
        size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + blended_pyr[i]
    return img


def blend_edges(matched: np.ndarray, cloudy: np.ndarray, mask: np.ndarray,
                 feather_radius: int = 15, pyramid_levels: int = 5) -> np.ndarray:
    """Blend replaced pixels into the original with no visible boundary,
    using Laplacian-pyramid multi-band blending; falls back to plain
    feathering if the pyramid blend fails for any reason."""
    try:
        smooth_mask = np.clip(filters.gaussian(mask.astype(np.float32), sigma=feather_radius / 2.0), 0.0, 1.0)
        blended = np.zeros_like(matched, dtype=np.float32)
        for b in range(matched.shape[0]):
            blended[b] = _laplacian_pyramid_blend_band(
                matched[b].astype(np.float32), cloudy[b].astype(np.float32), smooth_mask, levels=pyramid_levels
            )
        return blended.astype(matched.dtype)
    except Exception as exc:
        print(f"  \u26a0 Laplacian pyramid blend failed ({exc}); falling back to feathering.")
        return feather_blend(matched, cloudy, mask, feather_radius)


def poisson_blend(blended: np.ndarray, cloudy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV seamlessClone (Poisson blending) as a final touch-up, operating
    on an 8-bit 3-band proxy. Falls back silently to `blended` on failure."""
    try:
        n_bands = blended.shape[0]
        band_idx = list(range(min(3, n_bands)))
        while len(band_idx) < 3:
            band_idx.append(band_idx[-1])

        def to_u8(arr3):
            a = arr3.astype(np.float32)
            vmax = np.nanpercentile(a, 99) or 1.0
            return (np.clip(a / vmax, 0, 1) * 255).astype(np.uint8)

        src_rgb = to_u8(np.stack([blended[i] for i in band_idx], axis=-1))
        dst_rgb = to_u8(np.stack([cloudy[i] for i in band_idx], axis=-1))
        mask_u8 = mask.astype(np.uint8) * 255

        ys, xs = np.where(mask)
        if ys.size == 0:
            return blended
        center = (int(np.mean(xs)), int(np.mean(ys)))
        cloned = cv2.seamlessClone(
            cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2BGR),
            mask_u8, center, cv2.NORMAL_CLONE,
        )
        cloned_rgb = cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB).astype(np.float32)

        result = blended.copy().astype(np.float32)
        scale = np.nanpercentile(blended[band_idx], 99) or 255.0
        for i, b in enumerate(band_idx[:min(3, n_bands)]):
            band_result = result[b]
            band_result[mask] = cloned_rgb[..., i][mask] / 255.0 * scale
            result[b] = band_result
        return result.astype(blended.dtype)
    except Exception as exc:
        print(f"  \u26a0 Poisson blending skipped ({exc}); keeping the pyramid-blended result.")
        return blended
