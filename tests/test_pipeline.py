"""Tests for the complete inference pipeline (Phase 13)."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds

from pipeline.config import PipelineConfig
from pipeline.engine import InferencePipeline, PipelineResult, run_batch
from models.model_factory import ModelRegistry, PlaceholderModel


def _make_geotiff(path, data, crs="EPSG:4326", bounds=(0, 0, 1, 1)):
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    n, h, w = data.shape
    transform = from_bounds(*bounds, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=n, dtype="float32", crs=crs, transform=transform) as dst:
        dst.write(data.astype(np.float32))
    return path


@pytest.fixture(autouse=True)
def _register_placeholder():
    ModelRegistry.register("placeholder", PlaceholderModel)
    yield
    ModelRegistry.clear()


@pytest.fixture
def config():
    return PipelineConfig(model_name="placeholder")


@pytest.fixture
def pipeline(config):
    return InferencePipeline(config)


@pytest.fixture
def scene_files(tmp_path):
    np.random.seed(42)
    cloudy = np.random.rand(3, 32, 32).astype(np.float32)
    prev = np.random.rand(3, 32, 32).astype(np.float32)
    nxt = np.random.rand(3, 32, 32).astype(np.float32)
    ref = np.random.rand(3, 32, 32).astype(np.float32)

    cloud_mask = np.zeros((32, 32), dtype=bool)
    cloud_mask[5:15, 5:15] = True

    cloudy[:, 5:15, 5:15] = 0.9

    paths = {
        "cloudy": _make_geotiff(tmp_path / "scene.tif", cloudy),
        "prev": _make_geotiff(tmp_path / "scene_prev.tif", prev),
        "next": _make_geotiff(tmp_path / "scene_next.tif", nxt),
        "ref": _make_geotiff(tmp_path / "scene_ref.tif", ref),
    }
    return paths


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.model_name == "placeholder"
        assert cfg.target_bands == 3
        assert cfg.compute_metrics is True

    def test_output_dir_for(self, tmp_path):
        cfg = PipelineConfig()
        out = cfg.output_dir_for(tmp_path / "scene.tif")
        assert out.name == "scene_processed"

    def test_to_dict(self):
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert "model_name" in d
        assert "compute_metrics" in d


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_summary(self):
        result = PipelineResult(input_path="/a.tif", output_dir="/out")
        s = result.summary()
        assert s["input_path"] == "/a.tif"

    def test_final_image_none(self):
        result = PipelineResult(input_path="/a.tif", output_dir="/out")
        assert result.final_image is None


# ---------------------------------------------------------------------------
# Full pipeline run
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_basic_run(self, pipeline, scene_files, tmp_path):
        out = tmp_path / "output"
        result = pipeline.run(
            input_path=scene_files["cloudy"],
            output_dir=out,
        )
        assert isinstance(result, PipelineResult)
        assert result.stage_status["load"] == "ok"
        assert result.stage_status["preprocess"] == "ok"
        assert result.stage_status["cloud_detection"] == "ok"
        assert result.final_image is not None

    def test_with_temporal(self, pipeline, scene_files, tmp_path):
        out = tmp_path / "output"
        result = pipeline.run(
            input_path=scene_files["cloudy"],
            previous_path=scene_files["prev"],
            next_path=scene_files["next"],
            output_dir=out,
        )
        assert result.stage_status["temporal_reconstruction"] == "ok"
        assert result.temporal_result is not None

    def test_with_reference(self, pipeline, scene_files, tmp_path):
        out = tmp_path / "output"
        result = pipeline.run(
            input_path=scene_files["cloudy"],
            reference_path=scene_files["ref"],
            output_dir=out,
        )
        assert result.evaluation is not None
        assert result.evaluation.image_metrics.psnr > 0

    def test_saves_outputs(self, pipeline, scene_files, tmp_path):
        out = tmp_path / "output"
        result = pipeline.run(
            input_path=scene_files["cloudy"],
            output_dir=out,
        )
        assert "reconstructed" in result.output_files

    def test_file_not_found(self, pipeline, tmp_path):
        with pytest.raises(FileNotFoundError):
            pipeline.run(input_path=tmp_path / "nonexistent.tif")

    def test_default_output_dir(self, pipeline, scene_files):
        result = pipeline.run(input_path=scene_files["cloudy"])
        assert Path(result.output_dir).exists()

    def test_timings_recorded(self, pipeline, scene_files, tmp_path):
        result = pipeline.run(
            input_path=scene_files["cloudy"],
            output_dir=tmp_path / "out",
        )
        assert "total" in result.stage_timings
        assert result.stage_timings["total"] > 0


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestBatchRun:
    def test_batch(self, scene_files, tmp_path):
        scene_dir = tmp_path / "scenes"
        scene_dir.mkdir()
        import shutil
        shutil.copy(scene_files["cloudy"], scene_dir / "a.tif")

        out_dir = tmp_path / "batch_out"
        results = run_batch(
            input_dir=scene_dir,
            output_dir=out_dir,
            config=PipelineConfig(model_name="placeholder"),
        )
        assert len(results) == 1
        assert results[0].stage_status["load"] == "ok"

    def test_batch_with_temporal(self, scene_files, tmp_path):
        scene_dir = tmp_path / "scenes"
        scene_dir.mkdir()
        import shutil
        shutil.copy(scene_files["cloudy"], scene_dir / "a.tif")
        shutil.copy(scene_files["prev"], scene_dir / "a_prev.tif")

        out_dir = tmp_path / "batch_out"
        results = run_batch(
            input_dir=scene_dir,
            output_dir=out_dir,
            config=PipelineConfig(model_name="placeholder"),
        )
        assert len(results) == 2
        # The scene with a_prev.tif should have temporal reconstruction
        prev_result = [r for r in results if "a_prev" in r.input_path][0]
        assert prev_result.stage_status["load"] == "ok"

    def test_batch_empty_dir(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        results = run_batch(empty, tmp_path / "out")
        assert len(results) == 0
