"""Backend configuration.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BackendConfig:
    """Application settings loaded from environment."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    temp_dir: str = "./temp"
    max_upload_size_mb: int = 500
    allowed_extensions: List[str] = field(default_factory=lambda: [".tif", ".tiff"])

    model_name: str = "placeholder"
    checkpoint_path: str = ""
    model_device: str = "cpu"

    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "BackendConfig":
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            upload_dir=os.getenv("UPLOAD_DIR", "./uploads"),
            output_dir=os.getenv("OUTPUT_DIR", "./outputs"),
            temp_dir=os.getenv("TEMP_DIR", "./temp"),
            max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "500")),
            model_name=os.getenv("MODEL_NAME", "placeholder"),
            checkpoint_path=os.getenv("CHECKPOINT_PATH", ""),
            model_device=os.getenv("MODEL_DEVICE", "cpu"),
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        )

    def ensure_dirs(self) -> None:
        for d in [self.upload_dir, self.output_dir, self.temp_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
