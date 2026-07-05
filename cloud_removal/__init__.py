"""
cloud_removal
=============
Automatic Multi-Temporal Sentinel-2 Cloud Removal System.

Removes clouds, thin clouds/cirrus, haze and cloud shadows from a
Sentinel-2 GeoTIFF by reconstructing the affected pixels from a temporal
stack: an optional user-supplied clear reference image, plus historical
Sentinel-2 scenes automatically downloaded for the same coordinates from
Microsoft Planetary Computer's STAC catalog.

Typical usage:

    from cloud_removal import Config, CloudRemovalPipeline
    from cloud_removal.cli_input import get_input_paths

    config = Config(latitude=48.85, longitude=2.35)
    paths = get_input_paths(cloudy="cloudy.tif")
    pipeline = CloudRemovalPipeline(config)
    results = pipeline.run(paths)
"""
from .config import Config, CONFIG
from .pipeline import CloudRemovalPipeline
from .common import print_environment_summary

__all__ = ["Config", "CONFIG", "CloudRemovalPipeline", "print_environment_summary"]

__version__ = "1.0.0"
