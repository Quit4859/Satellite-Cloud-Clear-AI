"""Complete inference pipeline for satellite cloud removal.

Chains every stage into a single configurable call:

  load image → preprocess → cloud detection → registration →
  temporal reconstruction → AI refinement → evaluation → save outputs

All stages are optional and the pipeline degrades gracefully when
temporal images are unavailable (skips registration / temporal steps).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from models.base_model import ModelConfig, ModelOutput
from models.model_factory import get_model
from pipeline.config import PipelineConfig
from utils.cloud_detection import ThresholdCloudDetector, CloudDetectionConfig, CloudDetectionResult
from utils.evaluation import EvaluationReport, evaluate, save_json_report, save_csv_report
from utils.registration import ImageRegistration, RegistrationConfig
from models.temporal_reconstruction import (
    TemporalReconstruction,
    ReconstructionConfig,
    ReconstructionResult,
)
from utils.visualization import (
    create_rgb_preview,
    create_cloud_mask_overlay,
    create_before_after,
    create_difference_image,
    export_geotiff,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete output of the inference pipeline for one scene."""

    input_path: str
    output_dir: str

    original_image: Optional[np.ndarray] = None
    cloud_mask: Optional[np.ndarray] = None
    temporal_result: Optional[ReconstructionResult] = None
    ai_output: Optional[ModelOutput] = None
    evaluation: Optional[EvaluationReport] = None

    stage_timings: Dict[str, float] = field(default_factory=dict)
    stage_status: Dict[str, str] = field(default_factory=dict)
    output_files: Dict[str, str] = field(default_factory=dict)

    @property
    def final_image(self) -> Optional[np.ndarray]:
        """Best available reconstructed image."""
        if self.ai_output is not None:
            return self.ai_output.reconstructed
        if self.temporal_result is not None:
            return self.temporal_result.reconstructed_image
        return self.original_image

    def summary(self) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "stages": self.stage_status,
            "timings": {k: round(v, 3) for k, v in self.stage_timings.items()},
            "output_files": self.output_files,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Configurable end-to-end cloud removal pipeline.

    Usage::

        pipeline = InferencePipeline(PipelineConfig(model_name="prithvi"))
        result = pipeline.run("scene.tif", previous="prev.tif", next_image="next.tif")
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Stage: load
    # ------------------------------------------------------------------

    def _stage_load(
        self,
        input_path: Union[str, Path],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        import rasterio

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with rasterio.open(input_path, "r") as src:
            data = src.read().astype(np.float32)
            metadata = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "file_path": str(input_path),
            }

        logger.info(
            f"Loaded {input_path.name}: {data.shape[0]} bands, "
            f"{data.shape[1]}x{data.shape[2]}"
        )
        return data, metadata

    # ------------------------------------------------------------------
    # Stage: preprocess
    # ------------------------------------------------------------------

    def _stage_preprocess(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        nan_mask = np.isnan(data)
        if nan_mask.any():
            data[nan_mask] = 0.0
            logger.warning(f"Filled {int(nan_mask.sum())} NaN pixels with 0")
        return data

    # ------------------------------------------------------------------
    # Stage: cloud detection
    # ------------------------------------------------------------------

    def _stage_cloud_detection(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
    ) -> CloudDetectionResult:
        detector = ThresholdCloudDetector(
            CloudDetectionConfig(
                use_otsu=self.config.use_otsu,
                brightness_threshold=self.config.brightness_threshold,
                min_component_area=self.config.min_component_area,
                morph_open_kernel=self.config.morph_open_kernel,
                morph_close_kernel=self.config.morph_close_kernel,
            )
        )
        result = detector.detect(data, metadata=metadata)
        logger.info(f"Cloud coverage: {result.coverage:.2%}")
        return result

    # ------------------------------------------------------------------
    # Stage: registration
    # ------------------------------------------------------------------

    def _stage_registration(
        self,
        current_path: Union[str, Path],
        previous_path: Optional[Union[str, Path]],
        next_path: Optional[Union[str, Path]],
        output_dir: Path,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if previous_path is None and next_path is None:
            return None, None

        reg = ImageRegistration(
            RegistrationConfig(
                ecc_iterations=self.config.ecc_iterations,
                ecc_motion_type=self.config.ecc_motion_type,
                orb_features=self.config.orb_features,
                band_selection=self.config.band_selection,
            )
        )

        reg_dir = output_dir / "registered"
        reg_dir.mkdir(parents=True, exist_ok=True)

        prev_data = None
        next_data = None

        if previous_path is not None and Path(previous_path).exists():
            prev_result = reg.register_files(
                reference_path=current_path,
                moving_path=previous_path,
                output_path=reg_dir / "previous.tif",
            )
            prev_data = prev_result.registered_data
            logger.info(f"Registered previous: method={prev_result.method_used}")

        if next_path is not None and Path(next_path).exists():
            next_result = reg.register_files(
                reference_path=current_path,
                moving_path=next_path,
                output_path=reg_dir / "next.tif",
            )
            next_data = next_result.registered_data
            logger.info(f"Registered next: method={next_result.method_used}")

        return prev_data, next_data

    # ------------------------------------------------------------------
    # Stage: temporal reconstruction
    # ------------------------------------------------------------------

    def _stage_temporal(
        self,
        current: np.ndarray,
        previous: Optional[np.ndarray],
        next_image: Optional[np.ndarray],
        cloud_mask: np.ndarray,
    ) -> Optional[ReconstructionResult]:
        if previous is None and next_image is None:
            return None

        engine = TemporalReconstruction(
            ReconstructionConfig(
                invalid_data_value=self.config.invalid_data_value,
                max_cloud_fraction=self.config.max_cloud_fraction,
            )
        )

        if previous is None:
            previous = current.copy()
        if next_image is None:
            next_image = current.copy()

        result = engine.reconstruct(
            current=current,
            previous=previous,
            next_image=next_image,
            cloud_mask=cloud_mask,
        )
        logger.info(
            f"Temporal reconstruction: replacement_rate={result.overall_replacement_rate:.3f}"
        )
        return result

    # ------------------------------------------------------------------
    # Stage: AI refinement
    # ------------------------------------------------------------------

    def _stage_ai(
        self,
        cloudy: np.ndarray,
        cloud_mask: np.ndarray,
        previous: Optional[np.ndarray] = None,
        next_image: Optional[np.ndarray] = None,
    ) -> ModelOutput:
        model_config = ModelConfig(
            model_name=self.config.model_name,
            input_channels=cloudy.shape[0],
            output_channels=cloudy.shape[0],
            image_size=self.config.model_image_size,
            device=self.config.model_device,
            custom_params=self.config.model_custom_params,
        )

        model = get_model(
            self.config.model_name,
            model_config,
            load=True,
            checkpoint_path=self.config.checkpoint_path,
        )

        output = model.predict(
            cloudy=cloudy,
            previous=previous,
            next_image=next_image,
            cloud_mask=cloud_mask,
        )

        model.unload()
        logger.info(f"AI refinement complete: model={self.config.model_name}")
        return output

    # ------------------------------------------------------------------
    # Stage: evaluation
    # ------------------------------------------------------------------

    def _stage_evaluate(
        self,
        reference: np.ndarray,
        reconstructed: np.ndarray,
        cloud_mask: Optional[np.ndarray],
        unresolved_mask: Optional[np.ndarray],
        image_name: str,
    ) -> EvaluationReport:
        return evaluate(
            reference=reference,
            reconstructed=reconstructed,
            cloud_mask=cloud_mask,
            unresolved_mask=unresolved_mask,
            image_name=image_name,
        )

    # ------------------------------------------------------------------
    # Stage: save outputs
    # ------------------------------------------------------------------

    def _stage_save(
        self,
        result: PipelineResult,
        data: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, str]:
        import rasterio
        from rasterio.transform import Affine
        from rasterio.crs import CRS

        files: Dict[str, str] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        transform = metadata.get("transform")
        crs = metadata.get("crs")
        n_bands, height, width = data.shape

        profile = {
            "driver": "GTiff",
            "dtype": self.config.output_dtype,
            "width": width,
            "height": height,
            "count": n_bands,
            "crs": crs,
            "transform": transform,
            "compress": self.config.output_compression,
        }

        if self.config.save_reconstructed:
            out_path = output_dir / "reconstructed.tif"
            with rasterio.open(out_path, "w", **profile) as dst:
                for i in range(n_bands):
                    dst.write(data[i].astype(self.config.output_dtype), i + 1)
            files["reconstructed"] = str(out_path)

        if self.config.save_cloud_mask and result.cloud_mask is not None:
            mask_path = output_dir / "cloud_mask.tif"
            mask_profile = profile.copy()
            mask_profile.update(count=1, dtype="uint8", nodata=None)
            with rasterio.open(mask_path, "w", **mask_profile) as dst:
                dst.write(result.cloud_mask.astype(np.uint8), 1)
            files["cloud_mask"] = str(mask_path)

        if self.config.save_evaluation and result.evaluation is not None:
            json_path = output_dir / "evaluation.json"
            save_json_report(result.evaluation, json_path)
            files["evaluation_json"] = str(json_path)

        if self.config.save_visualizations:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

            if result.original_image is not None:
                rgb_path = vis_dir / "rgb_preview.png"
                create_rgb_preview(data, output_path=rgb_path, title="Reconstructed RGB")
                files["vis_rgb"] = str(rgb_path)

                ba_path = vis_dir / "before_after.png"
                create_before_after(result.original_image, data, output_path=ba_path)
                files["vis_before_after"] = str(ba_path)

                diff_path = vis_dir / "difference.png"
                create_difference_image(result.original_image, data, output_path=diff_path)
                files["vis_difference"] = str(diff_path)

            if result.cloud_mask is not None:
                overlay_path = vis_dir / "cloud_overlay.png"
                create_cloud_mask_overlay(data, result.cloud_mask, output_path=overlay_path)
                files["vis_cloud_overlay"] = str(overlay_path)

        logger.info(f"Saved {len(files)} outputs to {output_dir}")
        return files

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: Union[str, Path],
        previous_path: Optional[Union[str, Path]] = None,
        next_path: Optional[Union[str, Path]] = None,
        reference_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PipelineResult:
        """Execute the full pipeline on a single scene.

        Args:
            input_path: Path to the cloudy GeoTIFF.
            previous_path: Optional previous temporal image.
            next_path: Optional next temporal image.
            reference_path: Optional ground-truth reference for evaluation.
            output_dir: Where to write outputs. Defaults to ``*_processed``.

        Returns:
            PipelineResult with all intermediate outputs and metadata.
        """
        import time

        input_path = Path(input_path)
        if output_dir is None:
            output_dir = self.config.output_dir_for(input_path)
        output_dir = Path(output_dir)

        result = PipelineResult(
            input_path=str(input_path),
            output_dir=str(output_dir),
        )

        # Stage 1: Load
        t0 = time.time()
        data, metadata = self._stage_load(input_path)
        result.original_image = data.copy()
        result.stage_timings["load"] = time.time() - t0
        result.stage_status["load"] = "ok"

        # Stage 2: Preprocess
        t0 = time.time()
        data = self._stage_preprocess(data)
        result.stage_timings["preprocess"] = time.time() - t0
        result.stage_status["preprocess"] = "ok"

        # Stage 3: Cloud detection
        t0 = time.time()
        cloud_result = self._stage_cloud_detection(data, metadata)
        result.cloud_mask = cloud_result.cloud_mask
        result.stage_timings["cloud_detection"] = time.time() - t0
        result.stage_status["cloud_detection"] = "ok"

        # Stage 4: Registration (optional)
        t0 = time.time()
        prev_data, next_data = self._stage_registration(
            input_path, previous_path, next_path, output_dir,
        )
        result.stage_timings["registration"] = time.time() - t0
        result.stage_status["registration"] = "skipped" if prev_data is None and next_data is None else "ok"

        # Stage 5: Temporal reconstruction (optional)
        t0 = time.time()
        temporal = self._stage_temporal(data, prev_data, next_data, cloud_result.cloud_mask)
        result.temporal_result = temporal
        result.stage_timings["temporal_reconstruction"] = time.time() - t0
        result.stage_status["temporal_reconstruction"] = "skipped" if temporal is None else "ok"

        # Stage 6: AI refinement
        t0 = time.time()
        ai_output = self._stage_ai(
            cloudy=data,
            cloud_mask=cloud_result.cloud_mask,
            previous=prev_data,
            next_image=next_data,
        )
        result.ai_output = ai_output
        result.stage_timings["ai_refinement"] = time.time() - t0
        result.stage_status["ai_refinement"] = "ok"

        # Determine best reconstructed image
        final = ai_output.reconstructed

        # Stage 7: Evaluation (optional)
        if self.config.compute_metrics and reference_path is not None:
            t0 = time.time()
            ref_data, _ = self._stage_load(reference_path)
            ref_data = self._stage_preprocess(ref_data)
            unresolved = temporal.unresolved_mask if temporal is not None else None
            result.evaluation = self._stage_evaluate(
                reference=ref_data,
                reconstructed=final,
                cloud_mask=cloud_result.cloud_mask,
                unresolved_mask=unresolved,
                image_name=input_path.name,
            )
            result.stage_timings["evaluation"] = time.time() - t0
            result.stage_status["evaluation"] = "ok"
        else:
            result.stage_status["evaluation"] = "skipped"

        # Stage 8: Save
        t0 = time.time()
        result.output_files = self._stage_save(result, final, metadata, output_dir)
        result.stage_timings["save"] = time.time() - t0
        result.stage_status["save"] = "ok"

        total = sum(result.stage_timings.values())
        result.stage_timings["total"] = total
        logger.info(f"Pipeline complete in {total:.2f}s: {result.stage_status}")

        return result


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def run_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[PipelineConfig] = None,
    pattern: str = "*.tif",
    previous_suffix: str = "_prev",
    next_suffix: str = "_next",
    reference_suffix: str = "_ref",
) -> List[PipelineResult]:
    """Process every matching file in a directory.

    Naming convention::

        scene.tif           – cloudy image
        scene_prev.tif      – previous temporal (optional)
        scene_next.tif      – next temporal (optional)
        scene_ref.tif       – reference for evaluation (optional)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = InferencePipeline(config)

    files = sorted(input_dir.glob(pattern))
    if not files:
        logger.warning(f"No files matching '{pattern}' in {input_dir}")
        return []

    logger.info(f"Batch processing {len(files)} files")
    results: List[PipelineResult] = []

    for f in files:
        stem = f.stem
        prev = input_dir / f"{stem}{previous_suffix}.tif"
        nxt = input_dir / f"{stem}{next_suffix}.tif"
        ref = input_dir / f"{stem}{reference_suffix}.tif"

        out = output_dir / stem

        try:
            result = pipeline.run(
                input_path=f,
                previous_path=prev if prev.exists() else None,
                next_path=nxt if nxt.exists() else None,
                reference_path=ref if ref.exists() else None,
                output_dir=out,
            )
            results.append(result)
            logger.info(f"Processed {stem}: {result.stage_status}")
        except Exception as e:
            logger.error(f"Failed on {stem}: {e}")

    logger.info(f"Batch complete: {len(results)}/{len(files)} scenes processed")
    return results
