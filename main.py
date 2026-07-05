#!/usr/bin/env python3
"""
main.py
========
Command-line entry point for the Sentinel-2 multi-temporal cloud removal
pipeline. Run this on your own PC (no Colab required).

First-time setup:
    python install_dependencies.py
    python main.py --cloudy path/to/cloudy.tif

All input coordinates / thresholds can be overridden with CLI flags; run
`python main.py --help` to see every option. Anything not overridden uses
the defaults defined in cloud_removal/config.py.
"""
import argparse
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatic Multi-Temporal Sentinel-2 Cloud Removal System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Inputs ----
    parser.add_argument("--cloudy", type=str, default=None,
                         help="Path to the CLOUDY Sentinel-2 GeoTIFF (required). "
                              "If omitted, you will be prompted for it interactively.")
    parser.add_argument("--reference", type=str, default=None,
                         help="Optional path to a CLEAR REFERENCE GeoTIFF.")
    parser.add_argument("--scl", type=str, default=None,
                         help="Optional path to an SCL classification GeoTIFF for the cloudy image.")

    # ---- Location / date (for automatic historical-image download) ----
    parser.add_argument("--lat", type=float, default=None, help="Target latitude.")
    parser.add_argument("--lon", type=float, default=None, help="Target longitude.")
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                         metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                         help="Explicit bounding box instead of --lat/--lon.")
    parser.add_argument("--date", type=str, default=None,
                         help="Target acquisition date 'YYYY-MM-DD' of the cloudy image "
                              "(defaults to today).")

    # ---- Common tunables ----
    parser.add_argument("--output-root", type=str, default="outputs",
                         help="Root folder under which a new timestamped run folder is created.")
    parser.add_argument("--max-historical-images", type=int, default=15,
                         help="Max number of historical scenes to download.")
    parser.add_argument("--search-window-days", type=int, default=365,
                         help="How many days before/after --date to search for historical scenes.")
    parser.add_argument("--cloud-cover-threshold", type=float, default=20.0,
                         help="Max acceptable cloud cover (%%) for candidate historical scenes.")
    parser.add_argument("--cloud-prob-threshold", type=float, default=0.22,
                         help="s2cloudless probability threshold for a 'core' cloud pixel.")
    parser.add_argument("--no-ai-inpainting", action="store_true",
                         help="Disable the optional LaMa AI-inpainting fallback.")
    parser.add_argument("--no-poisson-blending", action="store_true",
                         help="Disable OpenCV Poisson blending (keep Laplacian-pyramid blend only).")
    parser.add_argument("--quiet-debug", action="store_true",
                         help="Disable extra debug PNG exports (faster, fewer files on disk).")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # Import the package lazily so `--help` works even before dependencies
    # are installed.
    try:
        from cloud_removal import Config, CloudRemovalPipeline, print_environment_summary
        from cloud_removal.cli_input import get_input_paths
    except ImportError as exc:
        print(f"\u2718 Missing dependency: {exc}")
        print("  Run 'python install_dependencies.py' first, then try again.")
        sys.exit(1)

    print_environment_summary()

    bbox = tuple(args.bbox) if args.bbox else None

    config = Config(
        latitude=args.lat,
        longitude=args.lon,
        bbox=bbox,
        target_date=args.date,
        output_root=args.output_root,
        max_historical_images=args.max_historical_images,
        search_window_days=args.search_window_days,
        cloud_cover_threshold=args.cloud_cover_threshold,
        cloud_prob_threshold=args.cloud_prob_threshold,
        use_ai_inpainting=not args.no_ai_inpainting,
        use_poisson_blending=not args.no_poisson_blending,
        debug_mode=not args.quiet_debug,
    )
    print(config)

    paths = get_input_paths(cloudy=args.cloudy, reference=args.reference, scl=args.scl)

    pipeline = CloudRemovalPipeline(config)
    pipeline.run(paths)


if __name__ == "__main__":
    main()
