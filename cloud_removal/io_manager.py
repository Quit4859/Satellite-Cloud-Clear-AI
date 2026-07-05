"""
io_manager.py
=============
`OutputManager` -- creates a unique, timestamped run directory under
`outputs/` and saves every intermediate PNG/artifact into it.
"""
from .common import os, np, cv2, plt, datetime, Optional


class OutputManager:
    """Creates a unique, timestamped run directory and saves artifacts to it.

    A brand-new folder is created for every run so previous runs are never
    overwritten, e.g. ``outputs/run_20260704_143522/``.
    """

    def __init__(self, output_root: str = "outputs") -> None:
        self.output_root = output_root
        self.run_dir = self._create_output_folder()
        os.makedirs(self.path("03_downloaded_images"), exist_ok=True)
        os.makedirs(self.path("04_registered_images"), exist_ok=True)

    def _create_output_folder(self) -> str:
        """Create ``<output_root>/run_<timestamp>/`` and return its path."""
        os.makedirs(self.output_root, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_root, f"run_{stamp}")
        suffix = 1
        candidate = run_dir
        while os.path.exists(candidate):
            candidate = f"{run_dir}_{suffix}"
            suffix += 1
        os.makedirs(candidate)
        return candidate

    def path(self, filename: str) -> str:
        """Return the absolute path of ``filename`` inside the run folder."""
        full = os.path.join(self.run_dir, filename)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        return full

    @staticmethod
    def _to_displayable(array: np.ndarray) -> np.ndarray:
        """Convert an arbitrary array (mask, float image, multi-band) into a
        uint8 image suitable for PNG export."""
        arr = np.asarray(array)
        if arr.dtype == bool:
            return arr.astype(np.uint8) * 255
        arr = arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]
        vmax = np.nanpercentile(arr, 99) if np.isfinite(arr).any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        arr = np.clip(arr / vmax, 0, 1)
        return (arr * 255).astype(np.uint8)

    def save_png(self, filename: str, array: np.ndarray, cmap: Optional[str] = None) -> str:
        """Save `array` (mask / single-band / RGB) as a PNG. Never displayed."""
        out_path = self.path(filename)
        disp = self._to_displayable(array)
        if disp.ndim == 2:
            if cmap:
                plt.imsave(out_path, disp, cmap=cmap)
            else:
                cv2.imwrite(out_path, disp)
        else:
            cv2.imwrite(out_path, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
        return out_path
