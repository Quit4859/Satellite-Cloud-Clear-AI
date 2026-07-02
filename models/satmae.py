"""SatMAE foundation model wrapper.

Architecture-only placeholder for the SatMAE (Satellite Masked
Autoencoder) model.  Implements the expected patch-embed → MAE
encoder → decoder shape for pipeline integration without requiring
actual pretrained weights.

Differences from Prithvi:
- Uses separate per-band positional embeddings.
- Decoder includes a band-reconstruction head.
"""

from typing import Optional

import numpy as np

from models.base_model import BaseReconstructionModel, ModelConfig, ModelOutput


class SatMAEEncoder:
    """Satellite-specific MAE encoder with per-band positional encoding."""

    def __init__(self, n_bands: int, patch_size: int, embed_dim: int = 512) -> None:
        self.n_bands = n_bands
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.band_embeddings = np.random.RandomState(42).randn(
            n_bands, embed_dim
        ).astype(np.float32) * 0.02

    def encode(self, patches: np.ndarray, band_idx: int = 0) -> np.ndarray:
        """Encode patches with band-specific positional embedding."""
        pos = self.band_embeddings[band_idx % self.n_bands]
        return patches + pos[: patches.shape[1]]


class SatMAEDecoder:
    """MAE-style decoder with band reconstruction head."""

    def __init__(self, embed_dim: int, n_bands: int, patch_size: int) -> None:
        self.embed_dim = embed_dim
        self.n_bands = n_bands
        self.patch_size = patch_size
        self.band_heads = [
            np.random.RandomState(i).randn(embed_dim, patch_size * patch_size).astype(np.float32) * 0.01
            for i in range(n_bands)
        ]

    def decode(self, embeddings: np.ndarray, h: int, w: int) -> np.ndarray:
        """Decode embeddings to per-band image patches."""
        ps = self.patch_size
        n_h = h // ps
        n_w = w // ps
        n_patches = min(embeddings.shape[0], n_h * n_w)

        img = np.zeros((self.n_bands, h, w), dtype=np.float32)
        idx = 0
        for i in range(n_h):
            for j in range(n_w):
                if idx >= n_patches:
                    break
                for b in range(self.n_bands):
                    vals = embeddings[idx] @ self.band_heads[b]
                    patch = vals.reshape(ps, ps)
                    img[b, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = patch.mean()
                idx += 1
        return img


class SatMAEModel(BaseReconstructionModel):
    """SatMAE architecture wrapper.

    Custom params:
        embed_dim (int): Transformer hidden size. Default 512.
        mask_ratio (float): Fraction of patches masked during MAE pretraining. Default 0.75.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        if config is None:
            config = ModelConfig(
                model_name="satmae",
                input_channels=13,
                output_channels=13,
                patch_size=16,
                image_size=224,
            )
        super().__init__(config)
        embed_dim = config.custom_params.get("embed_dim", 512)
        self._encoder = SatMAEEncoder(config.input_channels, config.patch_size, embed_dim)
        self._decoder = SatMAEDecoder(embed_dim, config.output_channels, config.patch_size)

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path is not None:
            pass  # Would load satmae checkpoint
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
        n_h = h // ps
        n_w = w // ps

        patches = []
        for i in range(n_h):
            for j in range(n_w):
                patch = cloudy[:, i * ps:(i + 1) * ps, j * ps:(j + 1) * ps]
                patches.append(patch.reshape(cloudy.shape[0], -1).mean(axis=1))

        if not patches:
            return ModelOutput(reconstructed=cloudy.copy(), confidence=np.ones((h, w), dtype=np.float32))

        patches_arr = np.stack(patches, axis=0)
        embeddings = self._encoder.encode(patches_arr)
        reconstructed = self._decoder.decode(embeddings, h, w)

        if reconstructed.shape[1] < h or reconstructed.shape[2] < w:
            padded = np.zeros((reconstructed.shape[0], h, w), dtype=np.float32)
            padded[:, :reconstructed.shape[1], :reconstructed.shape[2]] = reconstructed
            reconstructed = padded

        reconstructed = reconstructed[:, :h, :w]
        confidence = np.ones((h, w), dtype=np.float32) * 0.7

        return ModelOutput(
            reconstructed=reconstructed,
            confidence=confidence,
            metadata={"encoder": "satmae", "mask_ratio": self.config.custom_params.get("mask_ratio", 0.75)},
        )

    def unload(self) -> None:
        self._loaded = False
