"""Processing and file-management endpoints."""

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from backend.schemas import (
    DownloadInfo,
    ErrorResponse,
    JobStatus,
    ProcessRequest,
    ProcessResponse,
    ProcessingStatus,
    UploadResponse,
)
from backend.services import ProcessingService

router = APIRouter(prefix="/api", tags=["processing"])

_service: Optional[ProcessingService] = None


def get_service() -> ProcessingService:
    global _service
    if _service is None:
        _service = ProcessingService()
    return _service


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse, responses={400: {"model": ErrorResponse}})
async def upload_file(file: UploadFile = File(...)):
    """Upload a GeoTIFF for processing."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".tif", ".tiff"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    content = await file.read()
    svc = get_service()
    info = svc.save_upload(file.filename, content)

    return UploadResponse(
        file_id=info["file_id"],
        filename=info["filename"],
        size_bytes=info["size_bytes"],
    )


# ------------------------------------------------------------------
# Process
# ------------------------------------------------------------------

@router.post("/process", response_model=ProcessResponse, responses={404: {"model": ErrorResponse}})
async def process_file(req: ProcessRequest):
    """Start processing a previously uploaded file."""
    svc = get_service()

    if svc.get_upload_path(req.file_id) is None:
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_id}")

    job_id = svc.start_processing(
        file_id=req.file_id,
        model_name=req.model_name,
        previous_file_id=req.previous_file_id,
        next_file_id=req.next_file_id,
        reference_file_id=req.reference_file_id,
        compute_metrics=req.compute_metrics,
        save_visualizations=req.save_visualizations,
    )

    return ProcessResponse(
        job_id=job_id,
        status=ProcessingStatus.PENDING,
        message="Processing started",
    )


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------

@router.get("/status/{job_id}", response_model=JobStatus, responses={404: {"model": ErrorResponse}})
async def get_status(job_id: str):
    """Get the status of a processing job."""
    svc = get_service()
    job = svc.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

@router.get("/metrics/{job_id}", responses={404: {"model": ErrorResponse}})
async def get_metrics(job_id: str):
    """Get evaluation metrics for a completed job."""
    svc = get_service()
    job = svc.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.metrics is None:
        raise HTTPException(status_code=404, detail="Metrics not available (no reference provided)")
    return job.metrics


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------

@router.get("/download/{job_id}/{filename}", responses={404: {"model": ErrorResponse}})
async def download_file(job_id: str, filename: str):
    """Download an output file from a completed job."""
    svc = get_service()
    path = svc.get_output_path(job_id, filename)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=path, filename=filename, media_type="application/octet-stream")


@router.get("/download/{job_id}", response_model=list, responses={404: {"model": ErrorResponse}})
async def list_downloads(job_id: str):
    """List all downloadable files for a job."""
    svc = get_service()
    outputs = svc.list_outputs(job_id)
    if not outputs:
        raise HTTPException(status_code=404, detail=f"No outputs for job: {job_id}")
    files = []
    for p in outputs:
        path = Path(p)
        if path.exists():
            files.append(DownloadInfo(
                file_id=job_id,
                filename=path.name,
                size_bytes=path.stat().st_size,
                download_url=f"/api/download/{job_id}/{path.name}",
            ))
    return files
