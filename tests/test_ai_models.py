"""Tests for the AI reconstruction framework (Phase 9)."""

import numpy as np
import pytest

from models.base_model import BaseReconstructionModel, ModelConfig, ModelOutput
from models.model_factory import ModelRegistry, PlaceholderModel, get_model
from models.prithvi import PrithviModel
from models.satmae import SatMAEModel
from models.diffusion import DiffusionModel
from models.controlnet import ControlNetModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a clean registry for each test."""
    yield
    ModelRegistry.clear()


@pytest.fixture
def cfg():
    return ModelConfig(model_name="test", input_channels=3, output_channels=3)


@pytest.fixture
def sample_rgb():
    return np.random.rand(3, 64, 64).astype(np.float32)


@pytest.fixture
def sample_mask():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True
    return mask


@pytest.fixture
def sample_prev():
    return np.random.rand(3, 64, 64).astype(np.float32)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.model_name == "base"
        assert cfg.input_channels == 3
        assert cfg.patch_size == 16
        assert cfg.device == "cpu"

    def test_custom(self):
        cfg = ModelConfig(model_name="x", input_channels=6, custom_params={"lr": 0.001})
        assert cfg.model_name == "x"
        assert cfg.input_channels == 6
        assert cfg.custom_params["lr"] == 0.001


# ---------------------------------------------------------------------------
# ModelOutput
# ---------------------------------------------------------------------------

class TestModelOutput:
    def test_basic(self):
        out = ModelOutput(reconstructed=np.zeros((3, 16, 16), dtype=np.float32))
        assert out.reconstructed.shape == (3, 16, 16)
        assert out.confidence is None
        assert out.metadata == {}

    def test_with_confidence(self):
        conf = np.ones((16, 16), dtype=np.float32)
        out = ModelOutput(reconstructed=np.zeros((1, 16, 16), dtype=np.float32), confidence=conf)
        assert out.confidence.shape == (16, 16)


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_and_create(self):
        ModelRegistry.register("ph", PlaceholderModel)
        model = ModelRegistry.create("ph")
        assert isinstance(model, PlaceholderModel)

    def test_case_insensitive(self):
        ModelRegistry.register("PH", PlaceholderModel)
        assert "ph" in ModelRegistry.list_models()

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            ModelRegistry.create("nonexistent")

    def test_list_models(self):
        ModelRegistry.register("a", PlaceholderModel)
        ModelRegistry.register("b", PlaceholderModel)
        assert ModelRegistry.list_models() == ["a", "b"]

    def test_clear(self):
        ModelRegistry.register("tmp", PlaceholderModel)
        ModelRegistry.clear()
        assert ModelRegistry.list_models() == []

    def test_register_non_subclass_raises(self):
        with pytest.raises(TypeError, match="subclass"):
            ModelRegistry.register("bad", str)


# ---------------------------------------------------------------------------
# PlaceholderModel
# ---------------------------------------------------------------------------

class TestPlaceholderModel:
    def test_load_and_predict(self, cfg, sample_rgb, sample_mask):
        ModelRegistry.register("placeholder", PlaceholderModel)
        model = get_model("placeholder", cfg)
        assert model.is_loaded
        out = model.predict(sample_rgb, cloud_mask=sample_mask)
        assert out.reconstructed.shape == sample_rgb.shape
        assert out.confidence is not None

    def test_predict_single_band(self, cfg):
        ModelRegistry.register("ph", PlaceholderModel)
        model = get_model("ph", cfg)
        single = np.random.rand(64, 64).astype(np.float32)
        out = model.predict(single)
        assert out.reconstructed.ndim == 3

    def test_unload(self, cfg):
        ModelRegistry.register("ph", PlaceholderModel)
        model = get_model("ph", cfg)
        assert model.is_loaded
        model.unload()
        assert not model.is_loaded

    def test_get_info(self, cfg):
        ModelRegistry.register("ph", PlaceholderModel)
        model = get_model("ph", cfg)
        info = model.get_info()
        assert "model_name" in info
        assert info["loaded"] is True

    def test_predict_with_previous(self, cfg, sample_rgb, sample_mask):
        ModelRegistry.register("ph", PlaceholderModel)
        model = get_model("ph", cfg)
        prev = np.random.rand(3, 64, 64).astype(np.float32)
        out = model.predict(sample_rgb, previous=prev, cloud_mask=sample_mask)
        # Cloudy pixels should come from previous
        assert out.reconstructed.shape == sample_rgb.shape


# ---------------------------------------------------------------------------
# PrithviModel
# ---------------------------------------------------------------------------

class TestPrithviModel:
    def test_load_and_predict(self, sample_rgb, sample_mask):
        cfg = ModelConfig(model_name="prithvi", input_channels=3, output_channels=3, patch_size=16)
        model = PrithviModel(cfg)
        model.load_weights()
        assert model.is_loaded
        out = model.predict(sample_rgb, cloud_mask=sample_mask)
        assert out.reconstructed.shape == sample_rgb.shape
        assert out.confidence is not None

    def test_unloaded_raises(self, sample_rgb):
        cfg = ModelConfig(model_name="prithvi", input_channels=3, output_channels=3, patch_size=16)
        model = PrithviModel(cfg)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(sample_rgb)

    def test_single_band_input(self):
        cfg = ModelConfig(model_name="prithvi", input_channels=1, output_channels=1, patch_size=16)
        model = PrithviModel(cfg)
        model.load_weights()
        single = np.random.rand(64, 64).astype(np.float32)
        out = model.predict(single)
        assert out.reconstructed.ndim == 3

    def test_custom_params(self):
        cfg = ModelConfig(
            model_name="prithvi", input_channels=6, output_channels=6,
            patch_size=8, custom_params={"embed_dim": 256},
        )
        model = PrithviModel(cfg)
        model.load_weights()
        img = np.random.rand(6, 32, 32).astype(np.float32)
        out = model.predict(img)
        assert out.reconstructed.shape[0] == 6

    def test_unload(self):
        model = PrithviModel()
        model.load_weights()
        model.unload()
        assert not model.is_loaded

    def test_default_config(self):
        model = PrithviModel()
        assert model.config.model_name == "prithvi"
        assert model.config.input_channels == 6


# ---------------------------------------------------------------------------
# SatMAEModel
# ---------------------------------------------------------------------------

class TestSatMAEModel:
    def test_load_and_predict(self, sample_rgb, sample_mask):
        cfg = ModelConfig(model_name="satmae", input_channels=3, output_channels=3, patch_size=16)
        model = SatMAEModel(cfg)
        model.load_weights()
        out = model.predict(sample_rgb, cloud_mask=sample_mask)
        assert out.reconstructed.shape == sample_rgb.shape

    def test_unloaded_raises(self, sample_rgb):
        model = SatMAEModel(ModelConfig(input_channels=3, output_channels=3, patch_size=16))
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(sample_rgb)

    def test_single_band(self):
        cfg = ModelConfig(model_name="satmae", input_channels=1, output_channels=1, patch_size=8)
        model = SatMAEModel(cfg)
        model.load_weights()
        out = model.predict(np.random.rand(32, 32).astype(np.float32))
        assert out.reconstructed.ndim == 3

    def test_default_config(self):
        model = SatMAEModel()
        assert model.config.input_channels == 13

    def test_unload(self):
        model = SatMAEModel()
        model.load_weights()
        model.unload()
        assert not model.is_loaded


# ---------------------------------------------------------------------------
# DiffusionModel
# ---------------------------------------------------------------------------

class TestDiffusionModel:
    def test_load_and_predict(self, sample_rgb, sample_mask):
        cfg = ModelConfig(
            model_name="diffusion", input_channels=3, output_channels=3,
            custom_params={"n_steps": 10, "skip_steps": 3},
        )
        model = DiffusionModel(cfg)
        model.load_weights()
        out = model.predict(sample_rgb, cloud_mask=sample_mask)
        assert out.reconstructed.shape == sample_rgb.shape
        assert out.confidence is not None

    def test_unloaded_raises(self, sample_rgb):
        model = DiffusionModel(ModelConfig(input_channels=3, output_channels=3))
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(sample_rgb)

    def test_single_band(self):
        cfg = ModelConfig(
            model_name="diffusion", input_channels=1, output_channels=1,
            custom_params={"n_steps": 5, "skip_steps": 2},
        )
        model = DiffusionModel(cfg)
        model.load_weights()
        out = model.predict(np.random.rand(32, 32).astype(np.float32))
        assert out.reconstructed.ndim == 3

    def test_metadata(self, sample_rgb):
        cfg = ModelConfig(
            model_name="diffusion", input_channels=3, output_channels=3,
            custom_params={"n_steps": 20, "guidance_scale": 10.0},
        )
        model = DiffusionModel(cfg)
        model.load_weights()
        out = model.predict(sample_rgb)
        assert out.metadata["n_steps"] == 20
        assert out.metadata["guidance_scale"] == 10.0

    def test_unload(self):
        model = DiffusionModel()
        model.load_weights()
        model.unload()
        assert not model.is_loaded


# ---------------------------------------------------------------------------
# ControlNetModel
# ---------------------------------------------------------------------------

class TestControlNetModel:
    def test_load_and_predict(self, sample_rgb, sample_mask):
        cfg = ModelConfig(model_name="controlnet", input_channels=3, output_channels=3)
        model = ControlNetModel(cfg)
        model.load_weights()
        out = model.predict(sample_rgb, cloud_mask=sample_mask)
        assert out.reconstructed.shape == sample_rgb.shape

    def test_no_mask(self, sample_rgb):
        cfg = ModelConfig(model_name="controlnet", input_channels=3, output_channels=3)
        model = ControlNetModel(cfg)
        model.load_weights()
        out = model.predict(sample_rgb)
        assert out.reconstructed.shape == sample_rgb.shape

    def test_with_previous(self, sample_rgb, sample_prev, sample_mask):
        cfg = ModelConfig(model_name="controlnet", input_channels=3, output_channels=3)
        model = ControlNetModel(cfg)
        model.load_weights()
        out = model.predict(sample_rgb, previous=sample_prev, cloud_mask=sample_mask)
        assert out.reconstructed.shape == sample_rgb.shape

    def test_unloaded_raises(self, sample_rgb):
        model = ControlNetModel(ModelConfig(input_channels=3, output_channels=3))
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(sample_rgb)

    def test_single_band(self):
        cfg = ModelConfig(model_name="controlnet", input_channels=1, output_channels=1)
        model = ControlNetModel(cfg)
        model.load_weights()
        out = model.predict(np.random.rand(32, 32).astype(np.float32))
        assert out.reconstructed.ndim == 3

    def test_unload(self):
        model = ControlNetModel()
        model.load_weights()
        model.unload()
        assert not model.is_loaded


# ---------------------------------------------------------------------------
# get_model convenience
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_get_without_load(self):
        ModelRegistry.register("ph", PlaceholderModel)
        model = get_model("ph", load=False)
        assert not model.is_loaded

    def test_get_with_checkpoint(self):
        ModelRegistry.register("ph", PlaceholderModel)
        model = get_model("ph", checkpoint_path="/tmp/fake.ckpt")
        assert model.is_loaded
