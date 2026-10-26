"""Central configuration. Everything overridable through environment variables."""
from __future__ import annotations
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DGA_DATA_DIR", BACKEND_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"            # immutable originals (never overwritten)
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned copies / reports
UPLOAD_DIR = Path(os.getenv("DGA_UPLOAD_DIR", DATA_DIR / "uploads"))
ARTIFACT_DIR = Path(os.getenv("DGA_ARTIFACT_DIR", BACKEND_DIR / "app" / "artifacts"))
DATASET_PATH = Path(os.getenv("DGA_DATASET_PATH", RAW_DIR / "DuVal.csv"))
DATASET_URL = os.getenv(
    "DGA_DATASET_URL",
    "https://raw.githubusercontent.com/ahmedtariq71/Dataset1/main/DuVal.csv",
)
DB_PATH = Path(os.getenv("DGA_DB_PATH", DATA_DIR / "dga.sqlite3"))
MAX_UPLOAD_BYTES = int(os.getenv("DGA_MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
MAX_UPLOAD_ROWS = int(os.getenv("DGA_MAX_UPLOAD_ROWS", 20000))
CORS_ORIGINS = [o.strip() for o in os.getenv("DGA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
SEED = int(os.getenv("DGA_SEED", 42))
