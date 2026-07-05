"""
stac_search.py
==============
Searches Microsoft Planetary Computer STAC catalog for historical,
mostly-clear Sentinel-2 L2A scenes near the target coordinates/date.
"""
from .common import STAC_AVAILABLE, pystac_client, datetime, timedelta, Optional, List, Any


def resolve_target_datetime(config: "Config") -> datetime:
    """Parse `config.target_date`, defaulting to "now" if not supplied."""
    if config.target_date:
        return datetime.fromisoformat(config.target_date).replace(tzinfo=None)
    return datetime.utcnow()


def resolve_bbox(config: "Config") -> Optional[List[float]]:
    """Build a search bounding box from either `bbox` or lat/lon + buffer."""
    if config.bbox is not None:
        return list(config.bbox)
    if config.latitude is not None and config.longitude is not None:
        d = config.point_buffer_deg
        return [config.longitude - d, config.latitude - d,
                 config.longitude + d, config.latitude + d]
    return None


def search_historical_images(config: "Config", logger: "ProcessingLogger") -> List[Any]:
    """Search the STAC catalog for the best historical Sentinel-2 scenes.

    Returns:
        A list of pystac Items, sorted by (date-distance, cloud cover),
        truncated to `config.max_historical_images`. Empty list if STAC is
        unavailable, unconfigured (no coordinates), or the search fails --
        in all such cases the pipeline simply continues without historical
        imagery (reference-image-only, or original-pixels-only, mode).
    """
    if not STAC_AVAILABLE:
        logger.log("STAC packages unavailable -- skipping automatic historical-image download.")
        return []

    bbox = resolve_bbox(config)
    if bbox is None:
        logger.log("No latitude/longitude/bbox configured -- skipping automatic historical-image download.")
        return []

    target = resolve_target_datetime(config)
    start = (target - timedelta(days=config.search_window_days)).strftime("%Y-%m-%d")
    end = (target + timedelta(days=config.search_window_days)).strftime("%Y-%m-%d")

    try:
        catalog = pystac_client.Client.open(config.stac_catalog_url)
        search = catalog.search(
            collections=[config.stac_collection],
            bbox=bbox,
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": config.cloud_cover_threshold}},
        )
        items = list(search.items())
    except Exception as exc:
        logger.log(f"\u26a0 STAC search failed ({exc}); continuing without historical imagery.")
        return []

    def date_distance(item) -> float:
        item_dt = item.datetime.replace(tzinfo=None) if item.datetime.tzinfo else item.datetime
        return abs((item_dt - target).total_seconds())

    items = sorted(items, key=lambda it: (date_distance(it), it.properties.get("eo:cloud_cover", 100.0)))
    selected = items[: config.max_historical_images]
    logger.log(f"STAC search found {len(items)} candidate scene(s); using the best {len(selected)}.")
    for it in selected:
        logger.record_downloaded_image(
            it.id, it.datetime, it.properties.get("eo:cloud_cover"), registration_error=None
        )
    return selected
