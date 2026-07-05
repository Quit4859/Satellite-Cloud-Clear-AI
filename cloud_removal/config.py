"""
config.py
=========
Global configuration dataclass for the cloud-removal pipeline.

Edit the values in `Config`, or construct your own `Config(...)` instance
(e.g. from CLI arguments in main.py) and pass it to `CloudRemovalPipeline`.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    """Container for every tunable parameter used by the pipeline."""

    # ---- USER INPUTS (edit these) -----------------------------------------
    # Target location. Provide EITHER latitude+longitude (a small bounding
    # box is built automatically around the point) OR an explicit bbox
    # (min_lon, min_lat, max_lon, max_lat). Leave both None to disable
    # automatic historical-image download.
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None

    # Target acquisition date "YYYY-MM-DD" of the cloudy image. None = today.
    target_date: Optional[str] = None

    # Historical-image search parameters
    max_historical_images: int = 15
    search_window_days: int = 365
    cloud_cover_threshold: float = 20.0

    # STAC catalog (Microsoft Planetary Computer, free & keyless for search)
    stac_catalog_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
    stac_collection: str = "sentinel-2-l2a"

    # Bounding-box half-width (degrees) used to build a small AOI box around
    # a single latitude/longitude point (~2 km at the equator)
    point_buffer_deg: float = 0.02

    # ---- Image registration -------------------------------------------------
    orb_features: int = 2000
    ransac_reproj_threshold: float = 5.0
    min_match_count: int = 10

    # ---- Cloud detection (s2cloudless) ---------------------------------------
    cloud_prob_threshold: float = 0.22
    thin_cloud_threshold: float = 0.10
    use_adaptive_threshold: bool = True
    adaptive_block_size: int = 101
    adaptive_offset: float = 0.06

    # ---- Mask cleanup / expansion -------------------------------------------
    morph_kernel_size: int = 5
    min_object_size: int = 40
    min_hole_size: int = 64
    border_ring_px: int = 3
    halo_search_px: int = 10
    halo_brightness_margin: float = 0.05
    dilation_scale: float = 0.30
    dilation_min_px: int = 3
    dilation_max_px: int = 25

    # SCL classes considered "cloud / thin-cloud / snow" (shadow handled
    # separately). 8=Medium Prob. Cloud, 9=High Prob. Cloud,
    # 10=Thin Cirrus, 11=Snow/Ice
    scl_cloud_classes: Tuple[int, ...] = (8, 9, 10, 11)
    scl_shadow_class: int = 3

    # ---- Shadow detection ----------------------------------------------------
    shadow_dark_percentile: float = 20.0
    shadow_neighbor_px: int = 60
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    assumed_cloud_height_px: float = 25.0
    shadow_grow_px: int = 3

    # ---- Temporal compositing --------------------------------------------------
    # Number of leading bands (in Red, Green, Blue, NIR order) that the
    # downloaded historical imagery actually supplies. Any additional bands
    # in the uploaded cloudy GeoTIFF beyond this count fall back to the
    # single best available clear observation instead of a weighted median.
    stack_band_count: int = 4
    spectral_similarity_sigma: float = 0.08
    uploaded_reference_weight: float = 1.5

    # ---- Local color matching --------------------------------------------------
    local_color_window: int = 41
    local_match_margin: int = 40

    # ---- Blending -----------------------------------------------------------
    feather_radius: int = 15
    pyramid_levels: int = 5
    use_poisson_blending: bool = True

    # ---- Texture refinement -------------------------------------------------
    texture_refinement_strength: float = 0.55
    texture_blur_sigma: float = 3.0

    # ---- AI inpainting fallback (open-source LaMa) --------------------------
    # Used ONLY for the residual pixels the temporal stack (Steps 1-7)
    # genuinely could not fill (no clear observation of that pixel in any
    # layer). Runs on GPU automatically if `get_device()` finds one,
    # otherwise CPU -- no manual switch needed.
    use_ai_inpainting: bool = True

    # ---- Misc -----------------------------------------------------------------
    debug_mode: bool = True
    output_root: str = "outputs"
    export_compression: str = "LZW"



CONFIG = Config()
