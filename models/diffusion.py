"""Diffusion-based reconstruction model wrapper.

Architecture-only placeholder for a denoising diffusion model
tailored to satellite cloud removal.  The expected pipeline is:
noisy cloudy image → denoise conditioned on temporal context → clean output.

This module defines the architecture shape (forward diffusion,
reverse denoising, noise schedule) so the inference pipeline can
run end-to-end without actual trained weights.
"""

from typing import Optional

import numpy as np

from models.base_model import BaseReconstructionModel, ModelConfig, ModelOutput


class NoiseSchedule:
    """Linear beta schedule for diffusion timesteps."""

    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.n_steps = n_steps
        self.betas = np.linspace(beta_start, beta_end, n_steps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod)

    def add_noise(self, x: np.ndarray, t: int, rng: np.random.RandomState) -> tuple:
        """Add noise to x at timestep t. Returns (noised, noise)."""
        noise = rng.randn(*x.shape).astype(np.float32)
        sqrt_a = self.sqrt_alpha_cumprod[t]
        sqrt_b = self.sqrt_one_minus_alpha_cumprod[t]
        return sqrt_a * x + sqrt_b * noise, noise


class DenoiseUNet:
    """Minimal U-Net skeleton for denoising (no torch dependency)."""

    def __init__(self, in_channels: int, mid_channels: int = 64) -> None:
        self.in_channels = in_channels
        self.mid_channels = mid_channels

    def forward(self, x: np.ndarray, t: int) -> np.ndarray:
        """Identity forward pass with slight noise modulation (placeholder)."""
        scale = 1.0 - (t / 1000.0) * 0.01
        return x * scale


class DiffusionModel(BaseReconstructionModel):
    """Diffusion-based cloud removal model.

    Custom params:
        n_steps (int): Number of diffusion steps. Default 100.
        guidance_scale (float): Classifier-free guidance weight. Default 7.5.
        skip_steps (int): Denoise every Nth step for speed. Default 1.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        if config is None:
            config = ModelConfig(
                model_name="diffusion",
                input_channels=3,
                output_channels=3,
                image_size=256,
            )
        super().__init__(config)
        n_steps = config.custom_params.get("n_steps", 100)
        self._schedule = NoiseSchedule(n_steps=n_steps)
        self._unet = DenoiseUNet(config.input_channels)
        self._n_steps = n_steps
        self._guidance_scale = config.custom_params.get("guidance_scale", 7.5)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path is not None:
            pass  # Would load diffusion checkpoint
        self._loaded = True

    def _denoise_loop(self, image: np.ndarray) -> np.ndarray:
        """Reverse diffusion process: start from noisy, denoise to clean."""
        rng = np.random.RandomState(123)
        x = rng.randn(*image.shape).astype(np.float32) * 0.1 + image.mean()

        skip = self.config.custom_params.get("skip_steps", 1)
        for t in reversed(range(0, self._n_steps, skip)):
            x = self._unet.forward(x, t)

        return x.astype(np.float32)

    def predict(
        self,
        cloudy: np.ndarray,
        previous: Optional[np.ndarray] = None,
        next_image: Optional[np.ndarray] = None,
        cloud_mask: Optional[np.ndarray] = None,
    ) -> ModelOutput:
        if not self._loaded:
            raise RuntimeError("Model weights not loaded. Call load_weights() first.")

        if cloudy.ndim == 2:
            cloudy = cloudy[np.newaxis, ...]

        h, w = cloudy.shape[1], cloudy.shape[2]
        reconstructed = self._denoise_loop(cloudy)

        if reconstructed.shape[1] < h or reconstructed.shape[2] < w:
            padded = np.zeros((reconstructed.shape[0], h, w), dtype=np.float32)
            padded[:, :reconstructed.shape[1], :reconstructed.shape[2]] = reconstructed
            reconstructed = padded
        reconstructed = reconstructed[:, :h, :w]

        confidence = np.ones((h, w), dtype=np.float32) * 0.75
        if cloud_mask is not None:
            confidence[cloud_mask.astype(bool)] = 0.6

        return ModelOutput(
            reconstructed=reconstructed,
            confidence=confidence,
            metadata={
                "n_steps": self._n_steps,
                "guidance_scale": self._guidance_scale,
            },
        )

    def unload(self) -> None:
        self._loaded = False
