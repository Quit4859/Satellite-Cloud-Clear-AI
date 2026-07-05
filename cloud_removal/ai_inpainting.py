"""
ai_inpainting.py
==================
Optional AI-inpainting fallback (open-source LaMa) for the small residual
set of pixels the temporal stack itself could not fill.
"""
from .common import np, TORCH_AVAILABLE, LAMA_AVAILABLE, SimpleLama, get_device, Optional, Tuple, Any


class AIInpainter:
    """Lazily loads SimpleLama once, on whichever device is currently best."""
    _model = None
    _device = None

    @classmethod
    def get_model(cls) -> Optional[Any]:
        if not (TORCH_AVAILABLE and LAMA_AVAILABLE):
            return None
        device = get_device()
        if cls._model is None or cls._device != device:
            try:
                cls._model = SimpleLama(device=device)
                cls._device = device
            except Exception as exc:
                print(f"  \u26a0 Could not load LaMa model: {exc}")
                cls._model = None
        return cls._model


def ai_inpaint_residual(normalized_array: np.ndarray, unresolved_mask: np.ndarray,
                         logger: "ProcessingLogger") -> Tuple[np.ndarray, bool]:
    """Inpaint ONLY `unresolved_mask` pixels (those the temporal stack could
    not fill) using the open-source LaMa model, on GPU if available else CPU.

    `normalized_array` must be reflectance in [0, 1] (see `normalize_image`).
    Only the first 3 bands (assumed R,G,B) are touched by the AI model; any
    additional bands (e.g. NIR) are left exactly as the temporal step set
    them. Fails safe: on any error, or if the optional AI packages are not
    installed, the input array is returned unchanged.
    """
    if not unresolved_mask.any():
        return normalized_array, False

    model = AIInpainter.get_model()
    if model is None:
        logger.log("  \u26a0 AI inpainting unavailable (torch/simple-lama-inpainting not installed) "
                    "-- residual pixels kept as-is.")
        return normalized_array, False

    device = get_device()
    logger.log(f"AI-inpainting {int(unresolved_mask.sum())} residual pixel(s) on '{device}' "
               f"using open-source LaMa...")
    try:
        from PIL import Image
        n_bands = normalized_array.shape[0]
        rgb_idx = list(range(min(3, n_bands)))
        while len(rgb_idx) < 3:
            rgb_idx.append(rgb_idx[-1])
        rgb = np.moveaxis(np.stack([normalized_array[i] for i in rgb_idx], axis=0), 0, -1)
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

        pil_image = Image.fromarray(rgb_u8)
        pil_mask = Image.fromarray((unresolved_mask.astype(np.uint8) * 255))

        result = model(pil_image, pil_mask)
        result_arr = np.asarray(result).astype(np.float32) / 255.0

        out = normalized_array.copy()
        for i, b in enumerate(rgb_idx[:min(3, n_bands)]):
            band = out[b]
            band[unresolved_mask] = result_arr[..., i][unresolved_mask]
            out[b] = band
        logger.log(f"  \u2714 AI inpainting done on {device}.")
        return out, True
    except Exception as exc:
        logger.log(f"  \u26a0 AI inpainting failed ({exc}); keeping residual pixels as-is.")
        return normalized_array, False
