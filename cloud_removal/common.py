"""
common.py
=========
Shared third-party imports, optional-dependency detection, and small
cross-cutting helpers used by every module in the `cloud_removal` package.

Import everything you need from here, e.g.:

    from .common import np, cv2, Optional, Dict, Any, get_device
"""
import os
import io
import gc
import json
import time
import shutil
import traceback
import warnings
import importlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # never pop up interactive windows, only save to disk
import matplotlib.pyplot as plt

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.errors import RasterioIOError

from scipy import ndimage as ndi
from skimage import morphology, exposure, filters
from skimage.morphology import remove_small_objects, remove_small_holes

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Detect whether we are actually running inside Google Colab. Locally
# (running this as a normal PC repo) this will simply be False, and every
# code path that depended on it degrades to the local/CLI behaviour.
try:
    from google.colab import files as colab_files  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    colab_files = None

# STAC / Planetary Computer access is OPTIONAL. If unavailable, the pipeline
# simply skips automatic historical-image download and relies on whatever
# the user provided (reference image / SCL).
try:
    import pystac_client
    import planetary_computer as pc
    STAC_AVAILABLE = True
except ImportError:
    STAC_AVAILABLE = False
    pystac_client = None
    pc = None

# Open-source AI inpainting (LaMa) is OPTIONAL and used only as a fallback
# for the (usually tiny) handful of pixels the temporal stack itself could
# not fill. Runs on GPU automatically if one is available, otherwise CPU.
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False
    SimpleLama = None


def get_device() -> str:
    """Return 'cuda' if a GPU is available to torch right now, else 'cpu'.

    Called fresh every time AI inpainting runs, so if a GPU becomes
    available mid-session the very next run will automatically use it.
    """
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def print_environment_summary() -> None:
    """Print a one-line summary of which optional features are active."""
    print("Imports OK. Running in Colab:", IN_COLAB,
          "| STAC download available:", STAC_AVAILABLE,
          "| AI inpainting available:", (TORCH_AVAILABLE and LAMA_AVAILABLE),
          "| compute device:", get_device())
