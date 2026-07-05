"""
install_dependencies.py
========================
Standalone helper that installs every third-party package this project
needs. Safe to re-run (idempotent) -- it skips anything already installed.

Usage:
    python install_dependencies.py

This intentionally has NO imports from the `cloud_removal` package, since
its whole job is to make sure that package's dependencies exist before you
try to import it.
"""
import sys
import subprocess
import importlib
from time import time

# Mapping of "import name" -> "pip package name" (they sometimes differ)
REQUIRED_PACKAGES = {
    "rasterio": "rasterio",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "skimage": "scikit-image",
    "scipy": "scipy",
    "s2cloudless": "s2cloudless",
    "ipywidgets": "ipywidgets",
    "tqdm": "tqdm",
    "shapely": "shapely",
    "pystac_client": "pystac-client",
    "planetary_computer": "planetary-computer",
}

# These two are installed with --no-deps and treated as strictly OPTIONAL.
# Reason: their normal dependency resolution can silently pull in a
# different NumPy/SciPy version than the one Colab already has loaded,
# which corrupts the CURRENT session (typical symptom: "ModuleNotFoundError:
# No module named 'numpy.char'" or "NumPy installation fails to pass simple
# sanity checks" the moment anything imports scipy/skimage afterward).
# Colab already ships every real runtime dependency these two need (torch,
# Pillow, requests, tqdm), so --no-deps is safe and prevents that failure
# class entirely.
NO_DEPS_PACKAGES = {
    "torch": "torch",
    "simple_lama_inpainting": "simple-lama-inpainting",
}


def _numpy_scipy_ok() -> bool:
    """Import numpy + scipy.ndimage in a FRESH subprocess (not this one).

    A subprocess reflects exactly what's actually installed on disk right
    now, with no risk of stale caching in the current kernel -- so this is
    a reliable way to detect whether the install step above just corrupted
    them, before we let the rest of the notebook touch them.
    """
    try:
        subprocess.check_call(
            [sys.executable, "-c", "import numpy, scipy.ndimage"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _repair_numpy_scipy() -> bool:
    """Force a clean, mutually-consistent reinstall of numpy+scipy."""
    print("  \u23f3 Repairing numpy/scipy (force-reinstalling a consistent pair)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall",
             "--no-cache-dir", "-q", "numpy", "scipy"]
        )
    except subprocess.CalledProcessError as exc:
        print(f"  \u2718 Repair attempt failed: {exc}")
        return False
    return _numpy_scipy_ok()


def install_packages(packages: dict = REQUIRED_PACKAGES,
                      no_deps_packages: dict = NO_DEPS_PACKAGES,
                      quiet: bool = True) -> None:
    """Install any package in `packages`/`no_deps_packages` not already
    importable, then verify numpy/scipy weren't corrupted in the process.

    Args:
        packages: mapping of {import_name: pip_package_name}, installed
            normally (pip resolves dependencies).
        no_deps_packages: same mapping, installed with --no-deps (see
            NO_DEPS_PACKAGES docstring above for why).
        quiet: suppress pip's normal stdout noise when True.
    """
    print("Installing packages...")
    for import_name, pip_name in packages.items():
        try:
            importlib.import_module(import_name)
            print(f"  \u2714 {pip_name} already installed")
        except ImportError:
            print(f"  \u23f3 installing {pip_name} ...")
            cmd = [sys.executable, "-m", "pip", "install", pip_name]
            if quiet:
                cmd.append("-q")
            try:
                subprocess.check_call(cmd)
                print(f"  \u2714 {pip_name} installed")
            except subprocess.CalledProcessError as exc:
                # pystac-client / planetary-computer are used for the
                # OPTIONAL automatic-download feature -- the rest of the
                # notebook still works (reference-image-only mode) if these
                # two fail to install, so we warn instead of raising.
                if import_name in ("pystac_client", "planetary_computer"):
                    print(f"  \u26a0 Could not install {pip_name} ({exc}); "
                          f"automatic historical-image download will be disabled.")
                else:
                    print(f"  \u2718 FAILED to install {pip_name}: {exc}")
                    raise

    for import_name, pip_name in no_deps_packages.items():
        try:
            importlib.import_module(import_name)
            print(f"  \u2714 {pip_name} already installed")
            continue
        except ImportError:
            pass
        print(f"  \u23f3 installing {pip_name} (--no-deps, protects numpy/scipy) ...")
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", pip_name]
        if quiet:
            cmd.append("-q")
        try:
            subprocess.check_call(cmd)
            print(f"  \u2714 {pip_name} installed")
        except subprocess.CalledProcessError as exc:
            print(f"  \u26a0 Could not install {pip_name} ({exc}); "
                  f"AI-based inpainting fallback will be disabled.")

    if not _numpy_scipy_ok():
        print("  \u26a0 numpy/scipy look inconsistent after installs -- attempting an automatic repair...")
        if not _repair_numpy_scipy():
            print("\n" + "=" * 70)
            print("\u2718 Could not automatically repair numpy/scipy.")
            print("  FIX: Runtime \u25b8 Restart session, then Runtime \u25b8 Run all again.")
            print("  Everything above is already installed, so the next run will")
            print("  not repeat this -- it will just work.")
            print("=" * 70)
            raise RuntimeError("numpy/scipy broken after install -- restart the Colab runtime and re-run.")
        print("  \u2714 numpy/scipy repaired.")

    print("\u2714 Done\n")




if __name__ == "__main__":
    _t0 = time()
    install_packages()
    print(f"(install step took {time() - _t0:.1f}s)")
