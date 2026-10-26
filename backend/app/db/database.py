"""SQLite persistence: experiment index + prediction activity. Source of truth for metrics stays in the JSON artifacts."""
from __future__ import annotations
import json, sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from .. import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments(
  id TEXT PRIMARY KEY, implementation TEXT, title TEXT, kind TEXT, status TEXT,
  created_utc TEXT, seed INTEGER, dataset_sha256 TEXT, n_rows INTEGER, seconds REAL);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, mode TEXT, model_ids TEXT,
  n_records INTEGER, n_valid INTEGER, summary TEXT);
"""


@contextmanager
def conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def upsert_experiment(e: dict) -> None:
    p = e.get("provenance", {})
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (e["id"], e.get("implementation"), e.get("title"), e.get("kind"), e.get("status"),
                   p.get("created_utc"), p.get("seed"), p.get("dataset_sha256"), p.get("n_rows"), e.get("seconds")))


def list_experiments() -> list[dict]:
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM experiments ORDER BY created_utc DESC")]


def log_prediction(mode: str, model_ids: list[str], n_records: int, n_valid: int, summary: dict) -> None:
    with conn() as c:
        c.execute("INSERT INTO predictions(ts,mode,model_ids,n_records,n_valid,summary) VALUES (?,?,?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(timespec="seconds"), mode, json.dumps(model_ids),
                   n_records, n_valid, json.dumps(summary)))


def recent_predictions(limit: int = 10) -> list[dict]:
    with conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))]
    for r in rows:
        r["model_ids"] = json.loads(r["model_ids"]); r["summary"] = json.loads(r["summary"])
    return rows


def prediction_count() -> int:
    with conn() as c:
        return c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
