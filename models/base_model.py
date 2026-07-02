"""Abstract base class for AI reconstruction models.

Defines the interface that all foundation-model wrappers must implement.
The pipeline consumes only this interface, enabling plug-and-play swapping
between Prithvi, SatMAE, diffusion-based, or any future architecture
without modifying downstream code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ModelConfig:
    """Universal configuration for any reconstruction model."""

    model_name: str = "base"
    """Identifier for the model variant."""

    input_channels: int = 3
    """Number of input spectral bands."""

    output_channels: int = 3
    """Number of output spectral bands."""

    patch_size: int = 16
    """Patch size for ViT-based models."""

    image_size: int = 256
    """Expected spatial dimension (assumed square)."""

    device: str = "cpu"
    """Compute device: 'cpu', 'cuda', or 'cuda:0'."""

    half_precision: bool = False
    """Use float16 inference when True."""

    custom_params: Dict[str, Any] = field(default_factory=dict)
    """Model-specific parameters injected at runtime."""


@dataclass
class ModelOutput:
    """Standardised output from any reconstruction model."""

    reconstructed: np.ndarray
    """Cloud-free image (bands, H, W) float32."""

    confidence: Optional[np.ndarray] = None
    """Per-pixel confidence map (H, W) in [0, 1]. None when unavailable."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary model-specific metadata (latency, loss, etc.)."""


class BaseReconstructionModel(ABC):
    """Abstract interface for AI-based cloud removal models.

    Subclasses must implement ``load_weights``, ``predict``, and ``unload``.
    The rest of the pipeline interacts only with this ABC, so swapping models
    requires zero changes outside the factory.

    Lifecycle
    ---------
    1. ``__init__`` receives a ``ModelConfig`` (no I/O, no weights).
    2. ``load_weights`` loads checkpoint from disk or initialises random.
    3. ``predict`` runs a forward pass on numpy arrays.
    4. ``unload`` frees GPU memory.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether weights have been loaded and the model is ready."""
        return self._loaded

    @abstractmethod
    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        """Load model weights from a checkpoint or initialise randomly.

        Args:
            checkpoint_path: File-system path to a saved checkpoint.
                             When None the model initialises with random
                             weights (useful for architecture testing).
        """

    @abstractmethod
    def predict(
        self,
        cloudy: np.ndarray,
        previous: Optional[np.ndarray] = None,
        next_image: Optional[np.ndarray] = None,
        cloud_mask: Optional[np.ndarray] = None,
    ) -> ModelOutput:
        """Run a single inference pass.

        Args:
            cloudy: Current cloudy image (bands, H, W) float32.
            previous: Optional previous temporal image.
            next_image: Optional next temporal image.
            cloud_mask: Optional boolean (H, W) cloud mask.

        Returns:
            ModelOutput with the reconstructed cloud-free image.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release GPU / CPU resources held by the model."""

    def get_info(self) -> Dict[str, Any]:
        """Return a dictionary describing the model configuration."""
        return {
            "model_name": self.config.model_name,
            "input_channels": self.config.input_channels,
            "output_channels": self.config.output_channels,
            "patch_size": self.config.patch_size,
            "image_size": self.config.image_size,
            "device": self.config.device,
            "loaded": self._loaded,
        }
