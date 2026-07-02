from typing import Tuple

import numpy as np
from loguru import logger


class Resizer:
    """Resizes satellite image data to target dimensions."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        interpolation: str = "bilinear",
        keep_aspect_ratio: bool = False,
    ):
        """Initialize the resizer.

        Args:
            target_size: Target (height, width) for resized images.
            interpolation: Interpolation method ('bilinear', 'nearest', 'bicubic').
            keep_aspect_ratio: If True, pad to maintain aspect ratio.
        """
        self.target_size = target_size
        self.interpolation = interpolation
        self.keep_aspect_ratio = keep_aspect_ratio

    def resize(self, data: np.ndarray) -> np.ndarray:
        """Resize image data.

        Args:
            data: Input array with shape (bands, height, width).

        Returns:
            Resized array with shape (bands, target_h, target_w).
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for resizing. Install opencv-python.")

        interp_flag = self._get_cv2_interpolation()
        bands, h, w = data.shape
        target_h, target_w = self.target_size

        resized_bands = []
        for i in range(bands):
            band = data[i]
            if self.keep_aspect_ratio:
                band = self._resize_keep_ratio(band, target_h, target_w, interp_flag)
            else:
                band = cv2.resize(band, (target_w, target_h), interpolation=interp_flag)
            resized_bands.append(band)

        return np.stack(resized_bands, axis=0)

    def _resize_keep_ratio(
        self,
        band: np.ndarray,
        target_h: int,
        target_w: int,
        interp_flag: int,
    ) -> np.ndarray:
        """Resize while maintaining aspect ratio with padding.

        Args:
            band: 2D array to resize.
            target_h: Target height.
            target_w: Target width.
            interp_flag: OpenCV interpolation flag.

        Returns:
            Resized and padded array.
        """
        import cv2

        h, w = band.shape
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(band, (new_w, new_h), interpolation=interp_flag)

        canvas = np.zeros((target_h, target_w), dtype=band.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    def _get_cv2_interpolation(self) -> int:
        """Get OpenCV interpolation flag.

        Returns:
            OpenCV interpolation constant.
        """
        import cv2

        methods = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }
        return methods.get(self.interpolation, cv2.INTER_LINEAR)

    def resize_to_multiple(
        self,
        data: np.ndarray,
        multiple: int = 16,
    ) -> np.ndarray:
        """Resize to dimensions that are multiples of a given number.

        Useful for models requiring specific input size constraints.

        Args:
            data: Input array with shape (bands, height, width).
            multiple: Target multiple for dimensions.

        Returns:
            Resized array.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for resizing.")

        _, h, w = data.shape
        new_h = (h // multiple) * multiple
        new_w = (w // multiple) * multiple

        if new_h == h and new_w == w:
            return data

        original_target = self.target_size
        self.target_size = (new_h, new_w)
        result = self.resize(data)
        self.target_size = original_target
        return result
