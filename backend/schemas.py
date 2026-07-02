"""Pydantic schemas for request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    upload_time: datetime = Field(default_factory=datetime.utcnow)


class ProcessRequest(BaseModel):
    file_id: str
    model_name: str = "placeholder"
    previous_file_id: Optional[str] = None
    next_file_id: Optional[str] = None
    reference_file_id: Optional[str] = None
    compute_metrics: bool = True
    save_visualizations: bool = True


class ProcessResponse(BaseModel):
    job_id: str
    status: ProcessingStatus
    message: str


class MetricsResponse(BaseModel):
    psnr: float
    ssim: float
    mae: float
    rmse: float
    cloud_coverage: float = 0.0
    replacement_percentage: float = 0.0
    unresolved_percentage: float = 0.0


class JobStatus(BaseModel):
    job_id: str
    status: ProcessingStatus
    progress: float = 0.0
    message: str = ""
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_files: Dict[str, str] = Field(default_factory=dict)
    metrics: Optional[MetricsResponse] = None
    stage_status: Dict[str, str] = Field(default_factory=dict)
    timings: Dict[str, float] = Field(default_factory=dict)


class DownloadInfo(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    download_url: str


class HealthResponse(BaseModel):
    status: str
    version: str
    models_available: List[str]


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
