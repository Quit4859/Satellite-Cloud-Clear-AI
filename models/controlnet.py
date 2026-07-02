"""ControlNet-conditioned reconstruction model wrapper.

Architecture-only placeholder for a ControlNet-style model that uses
the cloud mask as a spatial conditioning signal.  The architecture
expects:

  Encoder cloud-free features ← (from temporal images)
  ControlNet conditioning     ← (cloud mask + noisy cloudy)
  Decoder                    → (reconstructed output)

This enables mask-aware inpainting without retraining the backbone.
"""

from typing import Optional

import numpy as np

from models.base_model import BaseReconstructionModel, ModelConfig, ModelOutput


class ControlNetConditioner:
    """Processes cloud mask into a conditioning feature map."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_weight = np.random.RandomState(99).randn(
            out_channels, in_channels, 3, 3
        ).astype(np.float32) * 0.01

    def condition(self, cloudy: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
        """Produce a conditioning tensor from cloudy image + mask.

        Args:
            cloudy: (C, H, W) float32.
            cloud_mask: (H, W) bool.

        Returns:
            Conditioning tensor (out_channels, H, W).
        """
        c, h, w = cloudy.shape
        mask_3d = cloud_mask[np.newaxis, ...].astype(np.float32)
        cond_input = np.concatenate([cloudy, mask_3d], axis=0)

        # Simplified: mean-pool over spatial dims and broadcast
        pooled = cond_input.mean(axis=(1, 2))
        out = np.zeros((self.out_channels, h, w), dtype=np.float32)
        for ch in range(self.out_channels):
            out[ch] = pooled[:c].mean() if c > 0 else 0.0
        return out


class ControlNetDecoder:
    """Decoder that fuses backbone features with ControlNet conditioning."""

    def __init__(self, n_bands: int) -> None:
        self.n_bands = n_bands

    def decode(self, features: np.ndarray, conditioning: np.ndarray, h: int, w: int) -> np.ndarray:
        """Merge backbone features with conditioning signal."""
        if features.shape[1] < h or features.shape[2] < w:
            padded = np.zeros((features.shape[0], h, w), dtype=np.float32)
            padded[:, :features.shape[1], :features.shape[2]] = features
            features = padded

        if conditioning.shape[1] < h or conditioning.shape[2] < w:
            pad_c = np.zeros((conditioning.shape[0], h, w), dtype=np.float32)
            pad_c[:, :conditioning.shape[1], :conditioning.shape[2]] = conditioning
            conditioning = pad_c

        # Weighted blend: features dominate, conditioning modulates
        alpha = 0.85
        out = alpha * features[:, :h, :w] + (1 - alpha) * conditioning[:, :h, :w]
        return out[: self.n_bands]


class ControlNetModel(BaseReconstructionModel):
    """ControlNet-style cloud removal with mask conditioning.

    Custom params:
        conditioning_channels (int): Extra input channels for mask. Default 1.
        guidance_scale (float): Conditioning strength. Default 1.0.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        if config is None:
            config = ModelConfig(
                model_name="controlnet",
                input_channels=3,
                output_channels=3,
                image_size=256,
            )
        super().__init__(config)
        cond_ch = config.custom_params.get("conditioning_channels", 1)
        self._conditioner = ControlNetConditioner(
            in_channels=config.input_channels + cond_ch,
            out_channels=config.input_channels,
        )
        self._decoder = ControlNetDecoder(config.output_channels)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path is not None:
            pass
        self._loaded = True

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

        if cloud_mask is None:
            cloud_mask = np.zeros((h, w), dtype=bool)

        conditioning = self._conditioner.condition(cloudy, cloud_mask)
        backbone_features = cloudy.copy()

        if previous is not None and previous.shape[1:] == (h, w):
            backbone_features = 0.5 * backbone_features + 0.5 * previous[:, :h, :w]

        reconstructed = self._decoder.decode(backbone_features, conditioning, h, w)
        reconstructed = reconstructed[:, :h, :w]

        confidence = np.ones((h, w), dtype=np.float32)
        confidence[cloud_mask.astype(bool)] = 0.65

        return ModelOutput(
            reconstructed=reconstructed,
            confidence=confidence,
            metadata={"conditioning": "controlnet_mask"},
        )

    def unload(self) -> None:
        self._loaded = False
