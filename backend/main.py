"""FastAPI application entry point.

Run with::

    uvicorn backend.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import BackendConfig
from backend.routers import process
from backend.schemas import HealthResponse
from backend.services import ProcessingService

config = BackendConfig.from_env()

app = FastAPI(
    title="Satellite Cloud Clear AI",
    description="API for satellite imagery cloud removal and reconstruction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(process.router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    svc = ProcessingService(config)
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_available=svc.list_available_models(),
    )


@app.on_event("startup")
async def startup():
    config.ensure_dirs()
