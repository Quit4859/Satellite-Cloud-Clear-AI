"""Business logic services.

Manages file uploads, processing jobs, and output retrieval.
Thread-safe for concurrent request handling.
"""

import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from backend.config import BackendConfig
from backend.schemas import (
    JobStatus,
    MetricsResponse,
    ProcessingStatus,
)
from models.model_factory import ModelRegistry
from models.prithvi import PrithviModel
from models.satmae import SatMAEModel
from models.diffusion import DiffusionModel
from models.controlnet import ControlNetModel
from pipeline.config import PipelineConfig
from pipeline.engine import InferencePipeline


def _register_default_models() -> None:
    for name, cls in [
        ("prithvi", PrithviModel),
        ("satmae", SatMAEModel),
        ("diffusion", DiffusionModel),
        ("controlnet", ControlNetModel),
    ]:
        if name not in ModelRegistry.list_models():
            ModelRegistry.register(name, cls)


_register_default_models()


class ProcessingService:
    """Manages upload processing lifecycle."""

    def __init__(self, config: Optional[BackendConfig] = None) -> None:
        self.config = config or BackendConfig.from_env()
        self.config.ensure_dirs()
        self._jobs: Dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def save_upload(self, filename: str, content: bytes) -> Dict[str, Any]:
        file_id = uuid.uuid4().hex[:12]
        ext = Path(filename).suffix.lower()
        safe_name = f"{file_id}{ext}"
        dest = Path(self.config.upload_dir) / safe_name
        dest.write_bytes(content)

        logger.info(f"Saved upload: {safe_name} ({len(content)} bytes)")
        return {
            "file_id": file_id,
            "filename": filename,
            "stored_name": safe_name,
            "size_bytes": len(content),
            "path": str(dest),
        }

    def get_upload_path(self, file_id: str) -> Optional[Path]:
        upload_dir = Path(self.config.upload_dir)
        for f in upload_dir.iterdir():
            if f.stem == file_id:
                return f
        return None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def start_processing(
        self,
        file_id: str,
        model_name: str = "placeholder",
        previous_file_id: Optional[str] = None,
        next_file_id: Optional[str] = None,
        reference_file_id: Optional[str] = None,
        compute_metrics: bool = True,
        save_visualizations: bool = True,
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        now = datetime.utcnow()

        job = JobStatus(
            job_id=job_id,
            status=ProcessingStatus.PENDING,
            created_at=now,
        )

        with self._lock:
            self._jobs[job_id] = job

        input_path = self.get_upload_path(file_id)
        if input_path is None:
            job.status = ProcessingStatus.FAILED
            job.message = f"Upload file not found: {file_id}"
            return job_id

        prev_path = self.get_upload_path(previous_file_id) if previous_file_id else None
        next_path = self.get_upload_path(next_file_id) if next_file_id else None
        ref_path = self.get_upload_path(reference_file_id) if reference_file_id else None

        output_dir = Path(self.config.output_dir) / job_id

        thread = threading.Thread(
            target=self._run_pipeline,
            args=(job_id, input_path, model_name, prev_path, next_path, ref_path, output_dir, compute_metrics, save_visualizations),
            daemon=True,
        )
        thread.start()

        logger.info(f"Started job {job_id}")
        return job_id

    def _run_pipeline(
        self,
        job_id: str,
        input_path: Path,
        model_name: str,
        prev_path: Optional[Path],
        next_path: Optional[Path],
        ref_path: Optional[Path],
        output_dir: Path,
        compute_metrics: bool,
        save_visualizations: bool,
    ) -> None:
        job = self._jobs[job_id]
        job.status = ProcessingStatus.RUNNING

        try:
            pipeline_config = PipelineConfig(
                model_name=model_name,
                model_device=self.config.model_device,
                checkpoint_path=self.config.checkpoint_path or None,
                compute_metrics=compute_metrics and ref_path is not None,
                save_visualizations=save_visualizations,
            )
            pipeline = InferencePipeline(pipeline_config)

            result = pipeline.run(
                input_path=input_path,
                previous_path=prev_path,
                next_path=next_path,
                reference_path=ref_path,
                output_dir=output_dir,
            )

            job.status = ProcessingStatus.COMPLETED
            job.output_files = result.output_files
            job.stage_status = result.stage_status
            job.timings = {k: round(v, 3) for k, v in result.stage_timings.items()}
            job.completed_at = datetime.utcnow()

            if result.evaluation is not None:
                em = result.evaluation.image_metrics
                cm = result.evaluation.cloud_metrics
                job.metrics = MetricsResponse(
                    psnr=em.psnr,
                    ssim=em.ssim,
                    mae=em.mae,
                    rmse=em.rmse,
                    cloud_coverage=cm.original_cloud_coverage if cm else 0.0,
                    replacement_percentage=cm.replacement_percentage if cm else 0.0,
                    unresolved_percentage=cm.unresolved_percentage if cm else 0.0,
                )

            logger.info(f"Job {job_id} completed")

        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.message = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Job {job_id} failed: {e}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        return self._jobs.get(job_id)

    def get_output_path(self, job_id: str, filename: str) -> Optional[Path]:
        return Path(self.config.output_dir) / job_id / filename

    def list_outputs(self, job_id: str) -> List[str]:
        job = self._jobs.get(job_id)
        if job is None:
            return []
        return list(job.output_files.values())

    def list_available_models(self) -> List[str]:
        return ModelRegistry.list_models()
