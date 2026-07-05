"""
temporal_stack.py
==================
Assembles every clear-ish observation (uploaded reference + registered
historical scenes) into a weighted temporal stack, plus a quicklook
preview image.
"""
from .common import np, plt, Optional, Dict, Any, List
from .normalization import normalize_image
from .stac_search import resolve_target_datetime


def build_temporal_stack(cloudy_meta: Dict[str, Any], reference_meta: Optional[Dict[str, Any]],
                          historical_results: List[Dict[str, Any]], config: "Config"
                          ) -> List[Dict[str, Any]]:
    """Assemble every available clear-ish observation into one list of
    "layers", each carrying its own array, clear-sky mask, and a base
    temporal weight (closer acquisition date to the target = higher weight;
    the uploaded reference -- if any -- gets its own configurable boost since
    it was presumably hand-picked as a good match).

    Historical layers only carry `config.stack_band_count` bands
    (Red/Green/Blue/NIR by convention); the uploaded reference (if present)
    contributes its full band set.
    """
    height, width = cloudy_meta["array"].shape[-2:]
    layers: List[Dict[str, Any]] = []

    if reference_meta is not None:
        layers.append({
            "id": "uploaded_reference",
            "array": normalize_image(reference_meta["array"]),
            "clear_mask": np.ones((height, width), dtype=bool),
            "weight": config.uploaded_reference_weight,
            "date_distance_days": 0.0,
        })

    target_dt = resolve_target_datetime(config)
    for res in historical_results:
        item_dt = res["datetime"].replace(tzinfo=None) if res["datetime"].tzinfo else res["datetime"]
        days = abs((item_dt - target_dt).total_seconds()) / 86400.0
        weight = 1.0 / (1.0 + days)
        clear_mask = res["clear_mask"] if res["clear_mask"] is not None else np.ones((height, width), dtype=bool)
        layers.append({
            "id": res["id"],
            "array": res["array"],
            "clear_mask": clear_mask,
            "weight": weight,
            "date_distance_days": days,
        })

    return layers


def save_temporal_stack_preview(layers: List[Dict[str, Any]], output_mgr: "OutputManager") -> None:
    """Save a simple side-by-side quicklook of every layer in the stack."""
    if not layers:
        return
    n = len(layers)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, layer in zip(axes, layers):
        rgb = layer["array"][:3]
        vmax = np.nanpercentile(rgb, 99) or 1.0
        disp = np.clip(np.moveaxis(rgb, 0, -1) / vmax, 0, 1)
        ax.imshow(disp)
        ax.set_title(f"{layer['id']}\nw={layer['weight']:.2f}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(layers):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_mgr.path("10_temporal_stack_preview.png"), dpi=100)
    plt.close(fig)
