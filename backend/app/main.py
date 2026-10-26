from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from . import config
from .api.routes import health, diagnosis, batch, datasets, experiments
from .ml.feature_engineering import FeatureError
from .ml.model_registry import ArtifactError
from .services import experiment_service

log = logging.getLogger("dga")


@asynccontextmanager
async def lifespan(app: FastAPI):
    n = experiment_service.sync_index()
    log.info("indexed %d experiments", n)
    yield


app = FastAPI(title="Transformer DGA Fault Diagnosis API", version="1.0.0", lifespan=lifespan,
              description="Academic research prototype. Outputs are decision support, not confirmed diagnoses.")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
for r in (health, diagnosis, batch, datasets, experiments):
    app.include_router(r.router, prefix="/api")


@app.exception_handler(FeatureError)
async def feature_error(_: Request, e: FeatureError):
    return JSONResponse(status_code=422, content={"detail": str(e), "rows": e.rows})


@app.exception_handler(ArtifactError)
async def artifact_error(_: Request, e: ArtifactError):
    return JSONResponse(status_code=503, content={"detail": str(e)})
