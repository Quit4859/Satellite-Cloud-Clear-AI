"""
pipeline.py
============
CloudRemovalPipeline -- orchestrates the full 12-step multi-temporal
Sentinel-2 cloud removal pipeline end to end.
"""
from .common import traceback, tqdm, Optional, Dict, Any
from .config import Config
from .logging_utils import ProcessingLogger, get_library_versions
from .io_manager import OutputManager
from .geotiff_io import read_geotiff
from .alignment import ensure_alignment
from .normalization import normalize_image, denormalize_image
from .stac_search import search_historical_images
from .registration import download_and_register_all
from .temporal_stack import build_temporal_stack, save_temporal_stack_preview
from .cloud_detection import generate_scl_mask, generate_cloud_masks, merge_all_cloud_masks
from .mask_cleanup import clean_mask, expand_mask_adaptive
from .shadow_detection import detect_shadows
from .pixel_selection import extract_cloud_pixels, select_temporal_replacement
from .ai_inpainting import ai_inpaint_residual
from .color_matching import histogram_match_replaced, local_brightness_correction
from .blending import blend_edges, poisson_blend
from .texture import refine_texture
from .export import save_geotiff, save_final_png


class CloudRemovalPipeline:
    """End-to-end multi-temporal Sentinel-2 cloud removal pipeline."""

    TOTAL_STEPS = 12

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = ProcessingLogger()
        self.output_mgr: Optional[OutputManager] = None
        self._pbar: Optional[tqdm] = None

    def _step(self, n: int, description: str) -> None:
        self.logger.log(f"Step {n}/{self.TOTAL_STEPS}: {description}")
        if self._pbar is not None:
            self._pbar.set_description(f"Step {n}/{self.TOTAL_STEPS}: {description}")
            self._pbar.update(1)

    def run(self, paths: Dict[str, Optional[str]]) -> Dict[str, Any]:
        """Execute the full 12-step pipeline. Returns a dict of result paths."""
        with tqdm(total=self.TOTAL_STEPS, desc="Starting", unit="step") as pbar:
            self._pbar = pbar
            try:
                return self._run(paths)
            except Exception as exc:
                self.logger.log(f"\u2718 PIPELINE FAILED: {exc}")
                print("=" * 60)
                print("A friendly summary of what went wrong:")
                print(f"  {type(exc).__name__}: {exc}")
                traceback.print_exc()
                print("=" * 60)
                if self.output_mgr is not None:
                    versions = get_library_versions()
                    self.logger.write(self.output_mgr.path("processing_log.txt"), self.config, versions)
                raise
            finally:
                self._pbar = None

    def _run(self, paths: Dict[str, Optional[str]]) -> Dict[str, Any]:
        cfg = self.config
        self.output_mgr = OutputManager(cfg.output_root)
        self.logger.log(f"Output folder: {self.output_mgr.run_dir}")

        # ---- Read inputs -----------------------------------------------------------------
        cloudy_meta = read_geotiff(paths["cloudy"])
        reference_meta = read_geotiff(paths["reference"]) if paths.get("reference") else None
        scl_meta = read_geotiff(paths["scl"]) if paths.get("scl") else None
        self.output_mgr.save_png("01_input.png", cloudy_meta["array"])
        if reference_meta is not None:
            reference_meta = ensure_alignment(cloudy_meta, reference_meta, "Reference image", self.logger)
            self.output_mgr.save_png("02_reference.png", reference_meta["array"])
        if scl_meta is not None:
            scl_meta = ensure_alignment(cloudy_meta, scl_meta, "SCL image", self.logger)
        scl_array = scl_meta["array"] if scl_meta is not None else None

        # ---- Step 1: search + register historical imagery ------------------------------
        items = search_historical_images(cfg, self.logger)
        historical_results = download_and_register_all(items, cloudy_meta, cfg, self.output_mgr, self.logger) if items else []
        self._step(1, "Historical Sentinel-2 imagery downloaded")

        # ---- Step 2: registration already performed inside download_and_register_all ---
        self._step(2, "Images registered to the cloudy image's grid")

        # ---- Step 3: temporal stack ------------------------------------------------------
        layers = build_temporal_stack(cloudy_meta, reference_meta, historical_results, cfg)
        save_temporal_stack_preview(layers, self.output_mgr)
        self.logger.set_metric("temporal_stack_layers", len(layers))
        self._step(3, f"Temporal stack built ({len(layers)} layer(s))")

        # ---- Step 4: cloud detection -----------------------------------------------------
        cloudy_norm = normalize_image(cloudy_meta["array"])
        scl_mask = generate_scl_mask(scl_array, cfg.scl_cloud_classes)
        s2_masks = generate_cloud_masks(cloudy_norm, cfg, scl_mask=scl_mask)
        self.output_mgr.save_png("05_cloud_probability.png", s2_masks["probability"], cmap="viridis")
        merged = merge_all_cloud_masks(scl_mask, s2_masks, cloudy_norm, cfg)
        self.output_mgr.save_png("06_cloud_mask.png", merged["merged_mask"])
        self.logger.set_metric("s2cloudless_threshold", cfg.cloud_prob_threshold)
        self._step(4, "Clouds detected (SCL + s2cloudless + borders + halo)")

        # ---- Step 5: shadow detection -----------------------------------------------------
        shadow_mask = detect_shadows(cloudy_norm, scl_array, merged["merged_mask"], cfg)
        self.output_mgr.save_png("07_shadow_mask.png", shadow_mask)
        self._step(5, "Cloud shadows detected")

        # ---- Step 6: combine + clean + expand ---------------------------------------------
        combined_mask = merged["merged_mask"] | shadow_mask
        self.output_mgr.save_png("08_combined_mask.png", combined_mask)
        cleaned_mask = clean_mask(combined_mask, cfg.morph_kernel_size, cfg.min_object_size,
                                    cfg.min_hole_size, debug=cfg.debug_mode, output_mgr=self.output_mgr)
        final_mask = expand_mask_adaptive(cleaned_mask, cfg, debug=cfg.debug_mode, output_mgr=self.output_mgr)
        self.output_mgr.save_png("09_expanded_mask.png", final_mask)
        cloud_pct = 100.0 * final_mask.mean()
        self.logger.set_metric("cloud_percentage", f"{cloud_pct:.2f}%")
        self.logger.set_metric("cloud_pixels", int(final_mask.sum()))
        self._step(6, "Mask combined, cleaned, and adaptively expanded")

        # ---- Step 7: temporal pixel selection ----------------------------------------------
        # NOTE: everything from here through Step 10 operates in NORMALIZED
        # reflectance space ([0,1], via `cloudy_norm`) -- the SAME domain as
        # every temporal-stack layer -- and gets denormalized back to the
        # source dtype/scale exactly once, in Step 11. Mixing raw (0-10000)
        # values in here previously crushed every replaced pixel to ~0.
        composite, sel_stats, unresolved_mask = select_temporal_replacement(cloudy_norm, layers, final_mask, cfg)
        if cfg.use_ai_inpainting and unresolved_mask.any():
            composite, ai_used = ai_inpaint_residual(composite, unresolved_mask, self.logger)
            sel_stats["ai_inpainted_pixels"] = int(unresolved_mask.sum()) if ai_used else 0
        self.output_mgr.save_png("11_selected_pixels.png", extract_cloud_pixels(composite, final_mask))
        self.logger.set_metric("temporal_selection_stats", sel_stats)
        self._step(7, f"Replacement pixels selected from temporal stack ({sel_stats['layers_used']} layer(s)), "
                      f"AI-filled {sel_stats.get('ai_inpainted_pixels', 0)} residual px")

        # ---- Step 8: local color matching -------------------------------------------------
        matched = histogram_match_replaced(composite, cloudy_norm, final_mask, cfg.local_match_margin)
        matched = local_brightness_correction(matched, cloudy_norm, final_mask, cfg.local_color_window)
        self.output_mgr.save_png("12_color_normalized.png", matched)
        self._step(8, "Local color normalization applied")

        # ---- Step 9: edge blending ----------------------------------------------------------
        blended = blend_edges(matched, cloudy_norm, final_mask, cfg.feather_radius, cfg.pyramid_levels)
        if cfg.use_poisson_blending:
            blended = poisson_blend(blended, cloudy_norm, final_mask)
        self.output_mgr.save_png("13_blended.png", blended)
        self._step(9, "Edges blended (Laplacian pyramid + Poisson)")

        # ---- Step 10: texture refinement -----------------------------------------------------
        refined = refine_texture(blended, composite, final_mask, cfg.texture_refinement_strength, cfg.texture_blur_sigma)
        self.output_mgr.save_png("14_final.png", refined)
        self._step(10, "Texture refinement applied")

        # ---- Step 11: export -------------------------------------------------------------------
        # Denormalize back to the ORIGINAL dtype/scale exactly once, here.
        final_array = denormalize_image(refined, cloudy_meta["array"].dtype, dst_max=10000.0)
        tif_path = self.output_mgr.path("15_final.tif")
        save_geotiff(tif_path, final_array, cloudy_meta["profile"], cfg.export_compression)
        png_path = self.output_mgr.path("14_final.png")
        save_final_png(png_path, final_array)
        self._step(11, "Final GeoTIFF and PNG exported (metadata preserved)")

        # ---- Step 12: log -----------------------------------------------------------------------
        versions = get_library_versions()
        self.logger.set_metric("output_folder", self.output_mgr.run_dir)
        log_path = self.output_mgr.path("processing_log.txt")
        self.logger.write(log_path, cfg, versions)
        self._step(12, "Export complete")

        print("\n\u2714 Pipeline finished successfully.")
        print(f"  Output folder : {self.output_mgr.run_dir}")
        print(f"  Final GeoTIFF : {tif_path}")
        print(f"  Final PNG     : {png_path}")
        print(f"  Log file      : {log_path}")

        return {
            "run_dir": self.output_mgr.run_dir, "geotiff": tif_path, "png": png_path,
            "log": log_path, "cloud_percentage": cloud_pct, "temporal_layers": len(layers),
        }
