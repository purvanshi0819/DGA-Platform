from __future__ import annotations
import math
from typing import Optional
from pydantic import BaseModel, Field, field_validator

GAS_FIELDS = ("H2", "CH4", "C2H6", "C2H4", "C2H2")


class GasReading(BaseModel):
    """Dissolved-gas concentrations. The five gases are required; CO is optional and unused by every model."""
    H2: float = Field(..., ge=0, description="Hydrogen")
    CH4: float = Field(..., ge=0, description="Methane")
    C2H6: float = Field(..., ge=0, description="Ethane")
    C2H4: float = Field(..., ge=0, description="Ethylene")
    C2H2: float = Field(..., ge=0, description="Acetylene")
    CO: Optional[float] = Field(None, ge=0, description="Carbon monoxide (recorded, not used by any model)")

    @field_validator("*")
    @classmethod
    def _finite(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("value must be a finite number")
        return v


class DiagnosisRequest(BaseModel):
    reading: GasReading
    model_ids: list[str] = Field(..., min_length=1, max_length=40)


class ModelResult(BaseModel):
    model_id: str
    implementation: str
    model_name: str
    feature_set: str
    status: str                      # "ok" | "not_computable"
    predicted_class: Optional[str] = None
    probabilities: Optional[dict[str, float]] = None
    probability_note: Optional[str] = None
    features: Optional[dict[str, float]] = None
    message: Optional[str] = None


class DiagnosisResponse(BaseModel):
    input: dict[str, Optional[float]]
    results: list[ModelResult]
    agreement: dict
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    models_available: int
    experiments_available: int
    dataset_available: bool
