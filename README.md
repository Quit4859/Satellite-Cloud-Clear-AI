# Satellite Cloud Clear AI

A production-ready AI system for cloud removal and surface reconstruction from satellite imagery using multi-temporal observations, foundation models, and deep learning.

## Overview

Satellite Cloud Clear AI removes clouds from satellite images and reconstructs the underlying surface. The system chains cloud detection, temporal registration, multi-frame reconstruction, and AI refinement into a single configurable pipeline.

**Supported sensors:** LISS-IV, Sentinel-1 SAR, Sentinel-2, DEM, and temporal imagery.

**Supported models:** Prithvi (NASA/IBM), SatMAE, Diffusion, ControlNet, and a Placeholder pass-through for testing.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Frontend   │────▶│    Backend    │────▶│    Pipeline      │
│  (Streamlit) │     │   (FastAPI)   │     │   (Inference)    │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    ▼                              ▼                              ▼
              ┌──────────┐                 ┌──────────────┐              ┌──────────────┐
              │   Cloud   │                 │  Temporal     │              │   AI Model    │
              │ Detection │                 │ Reconstruction│              │  (Prithvi,    │
              └──────────┘                 └──────────────┘              │   SatMAE...)  │
                                                                         └──────────────┘
```

### Pipeline Stages

1. **Load** — Read GeoTIFF with rasterio, preserve CRS/transform metadata
2. **Preprocess** — Normalize bands, fill NaN values
3. **Cloud Detection** — Otsu thresholding + morphological filtering + connected-component analysis
4. **Registration** — ECC alignment with ORB feature-based fallback
5. **Temporal Reconstruction** — Replace cloudy pixels from previous/next acquisitions (priority: previous > next)
6. **AI Refinement** — Foundation model enhancement via plug-and-play architecture
7. **Evaluation** — PSNR, SSIM, MAE, RMSE, cloud coverage, replacement percentage
8. **Visualization** — RGB preview, false color, cloud mask overlay, before/after, difference image, histograms
9. **Save** — GeoTIFF output with metadata, JSON evaluation report, PNG visualizations

## Project Structure

```
Satellite-Cloud-Clear-AI/
├── models/                    # AI reconstruction framework (Phase 9)
│   ├── base_model.py          # Abstract base class + ModelConfig/ModelOutput
│   ├── model_factory.py       # Factory pattern with registry
│   ├── prithvi.py             # Prithvi ViT wrapper
│   ├── satmae.py              # SatMAE wrapper
│   ├── diffusion.py           # Diffusion model wrapper
│   ├── controlnet.py          # ControlNet wrapper
│   └── temporal_reconstruction.py  # Multi-frame reconstruction engine
├── utils/                     # Core utilities
│   ├── cloud_detection.py     # Threshold-based cloud detector
│   ├── registration.py        # ECC/ORB image registration
│   ├── fusion.py              # Multi-sensor data fusion (Phase 10)
│   ├── evaluation.py          # Metrics + report export (Phase 11)
│   ├── visualization.py       # Publication-quality plots (Phase 12)
│   └── preprocessing.py       # Data preprocessing
├── pipeline/                  # Complete inference pipeline (Phase 13)
│   ├── config.py              # Pipeline configuration
│   └── engine.py              # Stage-based pipeline engine
├── backend/                   # FastAPI backend (Phase 14)
│   ├── main.py                # App entry point
│   ├── config.py              # Environment-based settings
│   ├── schemas.py             # Pydantic request/response models
│   ├── services.py            # Business logic
│   └── routers/
│       └── process.py         # API endpoints
├── frontend/                  # Streamlit dashboard (Phase 15)
│   └── app.py                 # Interactive UI
├── tests/                     # pytest test suite
│   ├── test_ai_models.py
│   ├── test_cloud_detection.py
│   ├── test_registration.py
│   ├── test_temporal_reconstruction.py
│   ├── test_fusion.py
│   ├── test_evaluation.py
│   ├── test_visualization.py
│   ├── test_data_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_pipeline.py
│   ├── test_backend.py
│   └── test_frontend.py
├── .github/workflows/ci.yml   # GitHub Actions CI (Phase 16)
├── Dockerfile                 # API container
├── Dockerfile.frontend        # Frontend container
├── docker-compose.yml         # Multi-service orchestration
├── .env.example               # Environment template
└── requirements.txt           # Python dependencies
```

## Installation

### Local Development

```bash
git clone https://github.com/your-org/Satellite-Cloud-Clear-AI.git
cd Satellite-Cloud-Clear-AI

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000` and the frontend at `http://localhost:8501`.

## Running

### Backend (FastAPI)

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

### Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

### Pipeline (CLI)

```python
from pipeline.config import PipelineConfig
from pipeline.engine import InferencePipeline

config = PipelineConfig(model_name="prithvi")
pipeline = InferencePipeline(config)

result = pipeline.run(
    input_path="scene.tif",
    previous_path="scene_prev.tif",
    next_path="scene_next.tif",
    reference_path="scene_ref.tif",
)
print(result.summary())
```

### Batch Processing

```python
from pipeline.engine import run_batch

results = run_batch(
    input_dir="./data/scenes",
    output_dir="./data/processed",
    pattern="*.tif",
)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload a GeoTIFF file |
| `POST` | `/api/process` | Start reconstruction processing |
| `GET` | `/api/status/{job_id}` | Check job status |
| `GET` | `/api/metrics/{job_id}` | Get evaluation metrics |
| `GET` | `/api/download/{job_id}/{filename}` | Download output file |
| `GET` | `/api/download/{job_id}` | List all output files |
| `GET` | `/health` | Health check |

## Configuration

All settings can be overridden via environment variables. See `.env.example` for the full list.

Key settings:
- `MODEL_NAME` — Which AI model to use (placeholder, prithvi, satmae, diffusion, controlnet)
- `MODEL_DEVICE` — Compute device (cpu, cuda)
- `CHECKPOINT_PATH` — Path to model weights (empty = random initialization)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# With coverage
pytest tests/ --cov=models --cov=utils --cov=pipeline --cov=backend --cov-report=html
```

## Deployment

### Hugging Face Spaces

1. Create a new Space with Docker SDK
2. Set `Dockerfile` as the build file
3. Configure environment variables in Space settings
4. Push to deploy

### Docker Compose

```bash
docker compose up -d
```

### GitHub Actions

CI runs automatically on push to main/develop:
- Unit tests across Python 3.10
- Linting with ruff and mypy
- Docker build verification

## Design Principles

- **SOLID** — Single responsibility, open/closed, dependency injection
- **Factory Pattern** — Models are registered by name and created via factory
- **Abstract Base Classes** — All models implement the same interface
- **Config-Driven** — Every behavior is tunable via dataclass configs
- **Metadata Preservation** — CRS, transform, nodata flow through every stage
- **Graceful Degradation** — Pipeline works without temporal images or reference

## License

MIT License. See [LICENSE](LICENSE) for details.
