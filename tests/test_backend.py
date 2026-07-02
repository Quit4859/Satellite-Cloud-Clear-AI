"""Tests for the FastAPI backend (Phase 14)."""

import io
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_bounds
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import BackendConfig
from backend.schemas import (
    ErrorResponse,
    JobStatus,
    MetricsResponse,
    ProcessRequest,
    ProcessResponse,
    ProcessingStatus,
    UploadResponse,
)
from backend.services import ProcessingService
from models.model_factory import ModelRegistry, PlaceholderModel


def _make_geotiff_bytes(shape=(3, 32, 32)):
    data = np.random.rand(*shape).astype(np.float32)
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff",
        height=shape[1], width=shape[2], count=shape[0],
        dtype="float32", crs="EPSG:4326", transform=Affine.identity(),
    ) as dst:
        dst.write(data)
    buf.seek(0)
    return buf.read(), data


@pytest.fixture(autouse=True)
def _register():
    ModelRegistry.register("placeholder", PlaceholderModel)
    yield
    ModelRegistry.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def service(tmp_path):
    cfg = BackendConfig(
        upload_dir=str(tmp_path / "uploads"),
        output_dir=str(tmp_path / "outputs"),
        temp_dir=str(tmp_path / "temp"),
    )
    return ProcessingService(cfg)


@pytest.fixture
def uploaded_file(client):
    content, _ = _make_geotiff_bytes()
    resp = client.post("/api/upload", files={"file": ("test.tif", content, "image/tiff")})
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestBackendConfig:
    def test_from_env(self):
        cfg = BackendConfig.from_env()
        assert cfg.host == "0.0.0.0"

    def test_ensure_dirs(self, tmp_path):
        cfg = BackendConfig(
            upload_dir=str(tmp_path / "u"),
            output_dir=str(tmp_path / "o"),
            temp_dir=str(tmp_path / "t"),
        )
        cfg.ensure_dirs()
        assert (tmp_path / "u").exists()
        assert (tmp_path / "o").exists()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_upload_response(self):
        r = UploadResponse(file_id="abc", filename="a.tif", size_bytes=100)
        assert r.file_id == "abc"

    def test_process_request_defaults(self):
        r = ProcessRequest(file_id="x")
        assert r.model_name == "placeholder"
        assert r.compute_metrics is True

    def test_job_status(self):
        j = JobStatus(job_id="j1", status=ProcessingStatus.PENDING, created_at="2024-01-01T00:00:00")
        assert j.status == ProcessingStatus.PENDING

    def test_metrics_response(self):
        m = MetricsResponse(psnr=30.0, ssim=0.9, mae=0.01, rmse=0.02)
        assert m.psnr == 30.0

    def test_health_response(self):
        from backend.schemas import HealthResponse
        h = HealthResponse(status="ok", version="1.0", models_available=["a"])
        assert h.status == "ok"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TestProcessingService:
    def test_save_upload(self, service):
        content, _ = _make_geotiff_bytes()
        info = service.save_upload("test.tif", content)
        assert "file_id" in info
        assert info["size_bytes"] == len(content)

    def test_get_upload_path(self, service):
        content, _ = _make_geotiff_bytes()
        info = service.save_upload("test.tif", content)
        path = service.get_upload_path(info["file_id"])
        assert path is not None
        assert path.exists()

    def test_get_upload_path_missing(self, service):
        assert service.get_upload_path("nonexistent") is None

    def test_start_processing(self, service):
        content, _ = _make_geotiff_bytes()
        info = service.save_upload("test.tif", content)
        job_id = service.start_processing(file_id=info["file_id"])
        assert isinstance(job_id, str)

    def test_get_job(self, service):
        content, _ = _make_geotiff_bytes()
        info = service.save_upload("test.tif", content)
        job_id = service.start_processing(file_id=info["file_id"])
        import time
        time.sleep(3)
        job = service.get_job(job_id)
        assert job is not None
        assert job.status in [ProcessingStatus.RUNNING, ProcessingStatus.COMPLETED]

    def test_list_available_models(self, service):
        models = service.list_available_models()
        assert "placeholder" in models


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "placeholder" in data["models_available"]

    def test_upload(self, client):
        content, _ = _make_geotiff_bytes()
        resp = client.post("/api/upload", files={"file": ("test.tif", content, "image/tiff")})
        assert resp.status_code == 200
        data = resp.json()
        assert "file_id" in data
        assert data["filename"] == "test.tif"

    def test_upload_invalid_extension(self, client):
        resp = client.post("/api/upload", files={"file": ("test.txt", b"data", "text/plain")})
        assert resp.status_code == 400

    def test_process(self, client, uploaded_file):
        resp = client.post("/api/process", json={"file_id": uploaded_file["file_id"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_process_unknown_file(self, client):
        resp = client.post("/api/process", json={"file_id": "nonexistent"})
        assert resp.status_code == 404

    def test_status(self, client, uploaded_file):
        proc = client.post("/api/process", json={"file_id": uploaded_file["file_id"]})
        job_id = proc.json()["job_id"]
        import time
        time.sleep(3)
        resp = client.get(f"/api/status/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] in ["running", "completed"]

    def test_status_not_found(self, client):
        resp = client.get("/api/status/nonexistent")
        assert resp.status_code == 404

    def test_download_not_found(self, client):
        resp = client.get("/api/download/nonexistent/file.tif")
        assert resp.status_code == 404

    def test_list_downloads_not_found(self, client):
        resp = client.get("/api/download/nonexistent")
        assert resp.status_code == 404
