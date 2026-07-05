"""
logging_utils.py
=================
Run-level logging: timestamped event log, per-run metrics, and a final
processing-log writer.
"""
from .common import time, datetime, importlib, Optional, Dict, Any, List


class ProcessingLogger:
    """Collects timestamped log messages and per-run metrics/records."""

    def __init__(self) -> None:
        self._lines: List[str] = []
        self._start_time = time.time()
        self._metrics: Dict[str, Any] = {}
        self._downloaded_images: List[Dict[str, Any]] = []

    def log(self, message: str) -> None:
        """Record a timestamped message and echo it to stdout."""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        self._lines.append(line)
        print(line)

    def set_metric(self, key: str, value: Any) -> None:
        """Store a scalar metric for the final log."""
        self._metrics[key] = value

    def record_downloaded_image(self, image_id: str, acquisition_date: Any,
                                 cloud_cover: Optional[float],
                                 registration_error: Optional[float]) -> None:
        """Record one historical image's metadata for the processing log."""
        self._downloaded_images.append({
            "id": image_id,
            "acquisition_date": str(acquisition_date),
            "cloud_cover": cloud_cover,
            "registration_error_px": registration_error,
        })

    def write(self, path: str, config: "Config", library_versions: Dict[str, str]) -> None:
        """Write the complete processing log, including all recorded metrics."""
        elapsed = time.time() - self._start_time
        with open(path, "w") as fh:
            fh.write("Sentinel-2 Multi-Temporal Cloud Removal -- Processing Log\n")
            fh.write("=" * 70 + "\n")
            fh.write(f"Run date/time      : {datetime.now().isoformat()}\n")
            fh.write(f"Execution time (s) : {elapsed:.2f}\n\n")
            fh.write("Configuration\n")
            fh.write("-" * 70 + "\n")
            for k, v in config.__dict__.items():
                fh.write(f"  {k:28s}: {v}\n")
            fh.write("\nMetrics\n")
            fh.write("-" * 70 + "\n")
            for k, v in self._metrics.items():
                fh.write(f"  {k:28s}: {v}\n")
            fh.write("\nDownloaded historical images\n")
            fh.write("-" * 70 + "\n")
            if not self._downloaded_images:
                fh.write("  (none -- reference-image-only mode, or download unavailable)\n")
            for rec in self._downloaded_images:
                fh.write(f"  {rec}\n")
            fh.write("\nLibrary versions\n")
            fh.write("-" * 70 + "\n")
            for k, v in library_versions.items():
                fh.write(f"  {k:28s}: {v}\n")
            fh.write("\nEvent log\n")
            fh.write("-" * 70 + "\n")
            for line in self._lines:
                fh.write(line + "\n")


def get_library_versions() -> Dict[str, str]:
    """Return the installed version string of every key dependency."""
    versions: Dict[str, str] = {}
    modules = ["rasterio", "numpy", "cv2", "skimage", "scipy", "tqdm"]
    for name in modules:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    for name in ("s2cloudless", "pystac_client", "planetary_computer"):
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions
