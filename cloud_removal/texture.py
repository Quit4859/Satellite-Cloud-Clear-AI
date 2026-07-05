"""
texture.py
===========
Re-injects high-frequency structural detail into the replaced region so
it does not look like a blurry reconstruction.
"""
from .common import np, cv2


def refine_texture(blended: np.ndarray, source_for_detail: np.ndarray, mask: np.ndarray,
                    strength: float = 0.55, blur_sigma: float = 3.0) -> np.ndarray:
    """Re-inject high-frequency structural detail from `source_for_detail`
    (the temporal composite) into the replaced region so it doesn't look
    like a blurry reconstruction -- only inside `mask`; everywhere else is
    left completely untouched.
    """
    refined = blended.astype(np.float32).copy()
    for b in range(blended.shape[0]):
        low_freq = cv2.GaussianBlur(source_for_detail[b].astype(np.float32), (0, 0), sigmaX=blur_sigma)
        high_freq = source_for_detail[b].astype(np.float32) - low_freq
        band = refined[b]
        band[mask] = band[mask] + strength * high_freq[mask]
        refined[b] = band
    return refined.astype(blended.dtype)
