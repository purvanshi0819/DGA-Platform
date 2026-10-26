from fastapi import APIRouter
from ... import config
from ...ml import model_registry as reg
from ...schemas.api import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    ex = list(reg.EXPERIMENTS_DIR.glob("*.json")) if reg.EXPERIMENTS_DIR.exists() else []
    return HealthResponse(status="ok", models_available=len(reg.list_models()),
                          experiments_available=len(ex), dataset_available=config.DATASET_PATH.exists())
