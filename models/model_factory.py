"""Model factory for creating reconstruction model instances.

Implements the Factory Pattern with Dependency Injection so that
new model backends can be registered at runtime and discovered
by name.  The pipeline never imports concrete model classes directly —
it always goes through this factory.
"""

from typing import Dict, Optional, Type

import numpy as np

from models.base_model import BaseReconstructionModel, ModelConfig, ModelOutput


class ModelRegistry:
    """Singleton registry mapping model names to their classes.

    Usage::

        ModelRegistry.register("prithvi", PrithviModel)
        model = ModelRegistry.create("prithvi", config)
    """

    _registry: Dict[str, Type[BaseReconstructionModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseReconstructionModel]) -> None:
        """Register a model class under a given name.

        Args:
            name: Lookup key (case-insensitive).
            model_class: A concrete subclass of BaseReconstructionModel.

        Raises:
            TypeError: If model_class is not a BaseReconstructionModel subclass.
        """
        if not (isinstance(model_class, type) and issubclass(model_class, BaseReconstructionModel)):
            raise TypeError(
                f"{model_class} must be a subclass of BaseReconstructionModel"
            )
        cls._registry[name.lower()] = model_class

    @classmethod
    def create(cls, name: str, config: Optional[ModelConfig] = None) -> BaseReconstructionModel:
        """Instantiate a registered model.

        Args:
            name: Registered model name.
            config: Configuration to pass to the model constructor.
                    If None, a default ``ModelConfig`` is used.

        Returns:
            An instantiated (but not yet loaded) model.

        Raises:
            KeyError: If the name is not registered.
        """
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise KeyError(
                f"Model '{name}' is not registered. Available: {available}"
            )
        if config is None:
            config = ModelConfig(model_name=name)
        return cls._registry[key](config)

    @classmethod
    def list_models(cls) -> list:
        """Return sorted list of registered model names."""
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations.  Primarily used in tests."""
        cls._registry.clear()


class PlaceholderModel(BaseReconstructionModel):
    """Deterministic pass-through for architecture testing.

    Returns the input image unchanged (or blends with a synthetic
    inpainting when a cloud mask is provided).  Useful for validating
    the pipeline without real weights.
    """

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        self._loaded = True

    def predict(
        self,
        cloudy: np.ndarray,
        previous: Optional[np.ndarray] = None,
        next_image: Optional[np.ndarray] = None,
        cloud_mask: Optional[np.ndarray] = None,
    ) -> ModelOutput:
        if cloudy.ndim == 2:
            cloudy = cloudy[np.newaxis, ...]

        result = cloudy.copy()

        if cloud_mask is not None:
            mask = cloud_mask.astype(bool)
            for b in range(result.shape[0]):
                if previous is not None and previous.shape == result.shape:
                    prev_band = previous[b] if previous.ndim == 3 else previous
                    result[b][mask] = prev_band[mask]
                elif next_image is not None and next_image.shape == result.shape:
                    nxt_band = next_image[b] if next_image.ndim == 3 else next_image
                    result[b][mask] = nxt_band[mask]

        confidence = np.ones(result.shape[1:], dtype=np.float32)
        if cloud_mask is not None:
            confidence[cloud_mask.astype(bool)] = 0.5

        return ModelOutput(reconstructed=result, confidence=confidence)

    def unload(self) -> None:
        self._loaded = False


def get_model(
    name: str,
    config: Optional[ModelConfig] = None,
    load: bool = True,
    checkpoint_path: Optional[str] = None,
) -> BaseReconstructionModel:
    """Convenience function: create, optionally load, and return a model.

    Args:
        name: Registered model name (e.g. 'prithvi', 'placeholder').
        config: Model configuration override.
        load: Whether to call ``load_weights`` immediately.
        checkpoint_path: Path passed to ``load_weights``.

    Returns:
        A ready-to-use model instance.
    """
    model = ModelRegistry.create(name, config)
    if load:
        model.load_weights(checkpoint_path)
    return model
