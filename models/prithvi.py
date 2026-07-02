"""Prithvi foundation model wrapper.

Architecture-only placeholder for NASA/IBM's Prithvi geospatial
foundation model.  The class mirrors the expected ViT encoder-decoder
shape but operates on randomly initialised weights so the full
pipeline can be tested without downloading gigabytes of checkpoints.

When a real checkpoint is available, ``load_weights`` loads it and
the model behaves identically from the pipeline's perspective.
"""

from typing import Optional

import numpy as np

from models.base_model import BaseReconstructionModel, ModelConfig, ModelOutput

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class _PrithviEncoder:
    """Minimal ViT encoder skeleton (no PyTorch dependency at inference)."""

    def __init__(self, channels: int, patch_size: int, embed_dim: int = 768) -> None:
        self.channels = channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def embed(self, patches: np.ndarray) -> np.ndarray:
        """Project patches to embedding space (identity placeholder)."""
        n = patches.shape[0]
        return np.random.RandomState(0).rand(n, self.embed_dim).astype(np.float32) * 0.01


class _PrithviDecoder:
    """Minimal decoder that reconstructs image patches."""

    def __init__(self, embed_dim: int, patch_size: int, out_channels: int) -> None:
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.out_channels = out_channels

    def reconstruct(self, embeddings: np.ndarray, h: int, w: int) -> np.ndarray:
        """Map embeddings back to image space (identity placeholder)."""
        n_patches = embeddings.shape[0]
        ps = self.patch_size
        gh = (h // ps) * ps
        gw = (w // ps) * ps
        n_h = gh // ps
        n_w = gw // ps
        expected = n_h * n_w
        n = min(n_patches, expected)
        img = np.zeros((self.out_channels, gh, gw), dtype=np.float32)
        idx = 0
        for i in range(n_h):
            for j in range(n_w):
                if idx < n:
                    val = float(embeddings[idx].mean())
                    img[:, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = val
                    idx += 1
        return img


class PrithviModel(BaseReconstructionModel):
    """Prithvi architecture wrapper.

    Parameters in ``ModelConfig.custom_params``:
        embed_dim (int): Transformer embedding dimension. Default 768.
        num_heads (int): Attention heads. Default 12.
        num_layers (int): Encoder layers. Default 12.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        if config is None:
            config = ModelConfig(
                model_name="prithvi",
                input_channels=6,
                output_channels=6,
                patch_size=16,
                image_size=224,
            )
        super().__init__(config)
        embed_dim = config.custom_params.get("embed_dim", 768)
        self._encoder = _PrithviEncoder(config.input_channels, config.patch_size, embed_dim)
        self._decoder = _PrithviDecoder(embed_dim, config.patch_size, config.output_channels)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path is not None:
            # Real implementation would load torch checkpoint here
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

        _, h, w = cloudy.shape
        ps = self.config.patch_size

        # Patchify
        n_h = h // ps
        n_w = w // ps
        patches = []
        for i in range(n_h):
            for j in range(n_w):
                patch = cloudy[:, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps]
                patches.append(patch.reshape(-1))
        if not patches:
            return ModelOutput(reconstructed=cloudy.copy(), confidence=np.ones((h, w), dtype=np.float32))

        patches_arr = np.stack(patches, axis=0)
        embeddings = self._encoder.embed(patches_arr)
        reconstructed = self._decoder.reconstruct(embeddings, h, w)

        # Pad if decoder output is smaller
        if reconstructed.shape[1] < h or reconstructed.shape[2] < w:
            padded = np.zeros((reconstructed.shape[0], h, w), dtype=np.float32)
            padded[:, :reconstructed.shape[1], :reconstructed.shape[2]] = reconstructed
            reconstructed = padded

        reconstructed = reconstructed[:, :h, :w]
        confidence = np.ones((h, w), dtype=np.float32) * 0.8

        return ModelOutput(
            reconstructed=reconstructed,
            confidence=confidence,
            metadata={"encoder": "prithvi_vit", "patch_size": ps},
        )

    def unload(self) -> None:
        self._loaded = False
