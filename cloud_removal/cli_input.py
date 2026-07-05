"""
cli_input.py
============
Resolves the three input file paths the pipeline needs (cloudy image,
optional clear reference, optional SCL raster).

Priority order:
  1. Paths passed in directly (e.g. from argparse in main.py).
  2. The Google Colab upload widget, if running inside Colab.
  3. Interactive `input()` prompts (plain local/PC usage).
"""
from .common import os, IN_COLAB, colab_files, Optional, Dict


def get_input_paths(cloudy: Optional[str] = None,
                     reference: Optional[str] = None,
                     scl: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Prompt the user for the cloudy image, plus an optional clear
    reference and an optional SCL raster. Falls back to manual path entry
    outside Colab.

    Returns:
        dict with keys "cloudy", "reference", "scl" ("reference"/"scl" may
        be None -- both are now optional given automatic historical download).
    """
    print("Resolving input images")
    paths: Dict[str, Optional[str]] = {"cloudy": cloudy, "reference": reference, "scl": scl}

    def _colab_upload(prompt: str) -> Optional[str]:
        print(prompt)
        try:
            uploaded = colab_files.upload()
        except Exception as exc:
            print(f"  \u2718 Upload widget failed: {exc}")
            return None
        if not uploaded:
            return None
        return list(uploaded.keys())[0]

    try:
        if paths["cloudy"]:
            # Paths were already supplied (e.g. via CLI args) -- just validate them.
            if not os.path.exists(paths["cloudy"]):
                raise FileNotFoundError(f"Cloudy image not found: {paths['cloudy']}")
            if paths["reference"] and not os.path.exists(paths["reference"]):
                print(f"  \u26a0 Reference path not found ({paths['reference']}); continuing without it.")
                paths["reference"] = None
            if paths["scl"] and not os.path.exists(paths["scl"]):
                print(f"  \u26a0 SCL path not found ({paths['scl']}); continuing without it.")
                paths["scl"] = None
        elif IN_COLAB:
            cloudy_path = _colab_upload("Please upload the CLOUDY Sentinel-2 GeoTIFF (required):")
            if not cloudy_path:
                raise ValueError("No cloudy image uploaded -- this file is required.")
            paths["cloudy"] = cloudy_path

            print("Optional: upload a CLEAR REFERENCE GeoTIFF (skip if you don't have one -- "
                  "historical Sentinel-2 imagery will be downloaded automatically instead):")
            paths["reference"] = _colab_upload("Upload reference GeoTIFF, or skip:")

            print("Optional: upload an SCL GeoTIFF for the cloudy image (skip to use s2cloudless only):")
            paths["scl"] = _colab_upload("Upload SCL GeoTIFF, or skip:")
        else:
            cloudy_path = input("Path to CLOUDY GeoTIFF (required): ").strip()
            ref_path = input("Path to REFERENCE GeoTIFF (optional, leave blank to skip): ").strip()
            scl_path = input("Path to SCL GeoTIFF (optional, leave blank to skip): ").strip()
            paths["cloudy"] = cloudy_path or None
            paths["reference"] = ref_path or None
            paths["scl"] = scl_path or None
            if not paths["cloudy"] or not os.path.exists(paths["cloudy"]):
                raise FileNotFoundError(f"Cloudy image not found: {paths['cloudy']}")
            if paths["reference"] and not os.path.exists(paths["reference"]):
                print(f"  \u26a0 Reference path not found ({paths['reference']}); continuing without it.")
                paths["reference"] = None
            if paths["scl"] and not os.path.exists(paths["scl"]):
                print(f"  \u26a0 SCL path not found ({paths['scl']}); continuing without it.")
                paths["scl"] = None
    except Exception as exc:
        print(f"  \u2718 Upload error: {exc}")
        raise

    print(f"  \u2714 Cloudy   : {paths['cloudy']}")
    print(f"  \u2714 Reference: {paths['reference'] if paths['reference'] else '(none -- will rely on historical download)'}")
    print(f"  \u2714 SCL      : {paths['scl'] if paths['scl'] else '(none -- s2cloudless only)'}")
    print("\u2714 Done\n")
    return paths
