from __future__ import annotations
import csv, io, json
from pathlib import Path
from fastapi import HTTPException
from ..ml import model_registry as reg
from ..db import database as db

_cache: dict[str, tuple[float, dict]] = {}


def load(eid: str) -> dict:
    p = reg.EXPERIMENTS_DIR / f"{eid}.json"
    if not p.exists():
        raise HTTPException(404, f"Experiment '{eid}' not found. Run scripts/train_all.py.")
    m = p.stat().st_mtime
    if eid not in _cache or _cache[eid][0] != m:
        _cache[eid] = (m, json.loads(p.read_text()))
    return _cache[eid][1]


def sync_index() -> int:
    n = 0
    if reg.EXPERIMENTS_DIR.exists():
        for p in sorted(reg.EXPERIMENTS_DIR.glob("*.json")):
            db.upsert_experiment(load(p.stem)); n += 1
    return n


def registry() -> list[dict]:
    """Experiment registry: how each result was produced."""
    rows = []
    for e in db.list_experiments():
        full = load(e["id"])
        rows.append({**e, "split": full.get("split"), "provenance": full.get("provenance"),
                     "warnings": full.get("warnings", []), "n_records": len(full.get("records", []))})
    return rows


def metric_rows(eid: str) -> list[dict]:
    """Flat metric table for the comparison page; protocol attributes are explicit on every row."""
    e = load(eid); out = []
    impl = e["implementation"]
    avg_note = {"A": "macro (notebook A headline)", "B": "weighted (notebook B headline)", "C": "accuracy only"}[impl]
    if impl == "C":
        f = e["final"]
        out.append({"experiment": eid, "implementation": "C", "feature_set": "pca15",
                    "model": f"PCA + ANN (k={f['k']})", "stage": "test", "n": f["n_test"], "headline_average": avg_note, **flatten(f["metrics"])})
        by = {}
        for r in e["comparison_10splits"]:
            by.setdefault(r["model"], []).append(r["test_acc_pct"])
        for m, v in by.items():
            import numpy as np
            out.append({"experiment": eid, "implementation": "C", "feature_set": "see model name", "model": m,
                        "stage": "mean_10_random_splits", "n": len(v), "headline_average": "accuracy only",
                        "accuracy": float(np.mean(v)) / 100, "accuracy_std": float(np.std(v, ddof=1)) / 100})
        return out
    for r in e["records"]:
        out.append({"experiment": eid, "implementation": impl, "feature_set": r["feature_set"], "model": r["model"],
                    "stage": r["stage"], "n": r["n"], "headline_average": avg_note, **flatten(r["metrics"])})
    return out


def flatten(m: dict) -> dict:
    f = {"accuracy": m.get("accuracy")}
    for avg in ("macro", "weighted"):
        for k, v in (m.get(avg) or {}).items():
            f[f"{avg}_{k}"] = v
    for k in ("specificity_weighted", "mcc"):
        if k in m: f[k] = m[k]
    return f


def export_csv(eid: str) -> str:
    rows = metric_rows(eid)
    keys = sorted({k for r in rows for k in r})
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return buf.getvalue()
