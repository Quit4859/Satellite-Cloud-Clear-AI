"""Pipeline configuration.

Defines every tunable knob for the complete inference pipeline:
preprocessing, cloud detection, registration, temporal reconstruction,
AI refinement, evaluation, and output.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PipelineConfig:
    """Top-level configuration for the satellite cloud-clear pipeline."""

    # --- Preprocessing ---
    target_bands: int = 3
    output_dtype: str = "float32"

    # --- Cloud detection ---
    use_otsu: bool = True
    brightness_threshold: Optional[int] = None
    min_component_area: int = 50
    morph_open_kernel: int = 3
    morph_close_kernel: int = 7

    # --- Registration ---
    ecc_iterations: int = 200
    ecc_motion_type: str = "euclidean"
    orb_features: int = 5000
    band_selection: str = "first"

    # --- Temporal reconstruction ---
    invalid_data_value: Optional[float] = None
    max_cloud_fraction: float = 1.0

    # --- AI model ---
    model_name: str = "placeholder"
    """Name of the registered model to use for AI refinement."""
    checkpoint_path: Optional[str] = None
    """Path to model checkpoint. None = random weights."""
    model_device: str = "cpu"
    model_image_size: int = 256
    model_custom_params: Dict[str, Any] = field(default_factory=dict)

    # --- Evaluation ---
    compute_metrics: bool = True

    # --- Output ---
    save_reconstructed: bool = True
    save_cloud_mask: bool = True
    save_evaluation: bool = True
    save_visualizations: bool = True
    output_compression: str = "lzw"

    # --- Directories ---
    temp_dir: Optional[str] = None
    """Scratch directory for intermediate files. None = system temp."""

    def output_dir_for(self, input_path: Any) -> Path:
        """Return a sibling ``*_processed`` directory for *input_path*."""
        p = Path(input_path)
        return p.parent / f"{p.stem}_processed"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
