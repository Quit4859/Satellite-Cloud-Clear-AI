# Satellite Cloud Clear AI

Automatic **multi-temporal Sentinel-2 cloud removal** system. Removes clouds,
thin clouds/cirrus, haze and cloud shadows from a Sentinel-2 GeoTIFF by
reconstructing the affected pixels from a **temporal stack**: an optional
clear reference image you provide, plus historical Sentinel-2 scenes
**automatically downloaded** for the same coordinates from Microsoft
Planetary Computer's STAC catalog.

The result is designed to look like *the same image, the same date*, just
without the clouds — not a patch visibly copied from a different acquisition.

This started as a Google Colab notebook and has been split into a normal,
runnable-on-your-PC Python package.

## Pipeline overview

| # | Stage |
|---|-------|
| 1 | Search & download historical Sentinel-2 imagery (STAC) |
| 2 | Register every image to the cloudy image (reproject + ORB/RANSAC) |
| 3 | Build the temporal stack (reference + historical + weights) |
| 4 | Detect clouds (SCL + s2cloudless core/thin/adaptive + borders + halo) |
| 5 | Detect cloud shadows (SCL + dark-object + sun geometry) |
| 6 | Combine and adaptively expand the final mask |
| 7 | Select replacement pixels from the temporal stack (weighted median) |
| 8 | Local color normalization of the selected pixels |
| 9 | Edge blending (Laplacian pyramid + optional Poisson) |
| 10 | Texture refinement (keep buildings/roads/runways sharp) |
| 11 | Export the final cloud-free GeoTIFF + PNG (metadata preserved) |
| 12 | Write the processing log |

Every intermediate result is written to disk inside a unique, timestamped
run folder under `outputs/`, so nothing is ever overwritten.

## Project structure

```
Satellite-Cloud-Clear-AI/
├── main.py                    # CLI entry point — run this
├── install_dependencies.py    # one-time dependency installer (optional helper)
├── requirements.txt
└── cloud_removal/             # the actual pipeline package
    ├── common.py               # shared third-party imports + optional-dep flags
    ├── config.py                # Config dataclass (every tunable parameter)
    ├── logging_utils.py         # run logger + processing-log writer
    ├── io_manager.py             # OutputManager (timestamped run folder, PNG export)
    ├── cli_input.py              # resolves cloudy/reference/SCL input paths
    ├── geotiff_io.py             # GeoTIFF reading
    ├── alignment.py              # grid-alignment check + reprojection
    ├── normalization.py          # reflectance <-> raw DN conversion
    ├── stac_search.py            # historical Sentinel-2 scene search (STAC)
    ├── registration.py           # download + ORB/RANSAC registration
    ├── temporal_stack.py         # assembles the weighted temporal stack
    ├── cloud_detection.py        # SCL + s2cloudless cloud masks, borders, halo
    ├── mask_cleanup.py           # denoise + adaptive mask expansion
    ├── shadow_detection.py       # cloud shadow detection
    ├── pixel_selection.py        # weighted-median temporal pixel fusion
    ├── color_matching.py         # local color/histogram matching
    ├── ai_inpainting.py          # optional LaMa AI-inpainting fallback
    ├── blending.py               # Laplacian pyramid + Poisson edge blending
    ├── texture.py                # texture/detail refinement
    ├── export.py                 # final GeoTIFF + PNG export
    └── pipeline.py                # CloudRemovalPipeline — orchestrates it all
```

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/Quit4859/Satellite-Cloud-Clear-AI.git
cd Satellite-Cloud-Clear-AI

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

python install_dependencies.py
```

`install_dependencies.py` installs everything in `requirements.txt`, and
installs `torch` / `simple-lama-inpainting` with `--no-deps` (they're only
used for an *optional* fallback step, and installing them normally can pull
in a NumPy/SciPy version that conflicts with the one this project needs).
It's idempotent — safe to re-run any time.

If you'd rather install manually:

```bash
pip install -r requirements.txt
pip install --no-deps torch simple-lama-inpainting   # optional AI fallback
```

## Usage

Minimal run (you'll be prompted for the cloudy image path if you don't pass
`--cloudy`):

```bash
python main.py --cloudy path/to/cloudy_image.tif
```

With automatic historical-image download enabled (recommended — this is
what actually fills in the clouds if you don't have a reference image):

```bash
python main.py \
  --cloudy path/to/cloudy_image.tif \
  --lat 48.8566 --lon 2.3522 \
  --date 2026-03-15
```

With an optional clear reference image and/or SCL classification raster:

```bash
python main.py \
  --cloudy path/to/cloudy_image.tif \
  --reference path/to/clear_reference.tif \
  --scl path/to/scl_classification.tif \
  --lat 48.8566 --lon 2.3522
```

See every available flag:

```bash
python main.py --help
```

Or use it as a library:

```python
from cloud_removal import Config, CloudRemovalPipeline
from cloud_removal.cli_input import get_input_paths

config = Config(latitude=48.8566, longitude=2.3522, target_date="2026-03-15")
paths = get_input_paths(cloudy="path/to/cloudy_image.tif")

pipeline = CloudRemovalPipeline(config)
results = pipeline.run(paths)
print(results)
```

## Output

Each run creates `outputs/run_<timestamp>/` containing:
- Every intermediate step as a PNG (input, masks, temporal stack preview,
  blending stages, etc.) for visual debugging.
- `15_final.tif` — the final cloud-free GeoTIFF, with original CRS/transform/
  metadata preserved.
- `14_final.png` — a quick-look PNG of the final result.
- `processing_log.txt` — full configuration, metrics, downloaded-image list,
  library versions, and the timestamped event log for that run.

## Notes

- Automatic historical-image download requires `--lat`/`--lon` (or `--bbox`)
  and needs `pystac-client` + `planetary-computer` (installed by default).
  If unavailable/unconfigured, the pipeline simply relies on whatever
  reference/SCL image you supplied instead.
- AI inpainting (LaMa) only ever touches the handful of pixels that had
  *no* usable observation anywhere in the temporal stack — it runs on GPU
  automatically if `torch.cuda.is_available()`, otherwise CPU.
