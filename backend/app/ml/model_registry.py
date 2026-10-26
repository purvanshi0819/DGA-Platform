"""Artifact naming, metadata and loading. Inference refuses to run without metadata."""
from __future__ import annotations
import json, re
from functools import lru_cache
from pathlib import Path
import joblib
from .. import config

MODELS_DIR = config.ARTIFACT_DIR / "models"
EXPERIMENTS_DIR = config.ARTIFACT_DIR / "experiments"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def model_id(implementation: str, feature_set: str, model: str) -> str:
    return f"{implementation}__{feature_set}__{slug(model)}"


class ArtifactError(RuntimeError):
    pass


def save_model(mid: str, estimator, meta: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, MODELS_DIR / f"{mid}.joblib")
    (MODELS_DIR / f"{mid}.json").write_text(json.dumps(meta, indent=2))


def list_models() -> list[dict]:
    if not MODELS_DIR.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(MODELS_DIR.glob("*.json"))]


@lru_cache(maxsize=128)
def load_model(mid: str):
    """Load (estimator, metadata). Cached: models are loaded once per process."""
    mp, jp = MODELS_DIR / f"{mid}.joblib", MODELS_DIR / f"{mid}.json"
    if not mp.exists() or not jp.exists():
        raise ArtifactError(f"Model '{mid}' is not available. Run scripts/train_all.py first.")
    return joblib.load(mp), json.loads(jp.read_text())
