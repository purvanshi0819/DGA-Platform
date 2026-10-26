"""Train/evaluate the three research implementations and export artifacts.

Usage:  python scripts/train_all.py [--only rfrep|dga|pcaann] [--skip-corrected]
Each exported model is the notebook's own fitted estimator (trained on the 80% training split),
saved together with a metadata JSON. Nothing here is mocked; all metrics come from these runs.
"""
from __future__ import annotations
import argparse, json, platform, sys, time, datetime as dt
from pathlib import Path
import numpy as np, sklearn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config
from app.ml import dataset, dga_methods, rf_comparison, pca_ann, model_registry as reg
from app.ml.feature_engineering import FEATURE_SETS


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o


def provenance(sha, protocol, n_rows):
    return {"origin": "newly_run", "note": "The source notebooks contain no executed outputs; every value was computed by this pipeline.",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "seed": config.SEED, "dataset_file": config.DATASET_PATH.name, "dataset_sha256": sha,
            "cleaning_protocol": protocol, "n_rows": n_rows,
            "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__}}


def write_experiment(eid, payload):
    reg.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in payload.items() if k != "fitted"}
    (reg.EXPERIMENTS_DIR / f"{eid}.json").write_text(json.dumps(jsonable(out), indent=1))
    print(f"  wrote experiment {eid}")


def export(mid, est, meta, base):
    reg.save_model(mid, est, jsonable({"id": mid, **base, **meta}))


def do_rfrep(raw, sha):
    t = time.time(); print("[A] RF replication ...")
    df = dataset.clean_conflict(raw)
    r = rf_comparison.run_experiment(df)
    classes = r["labels"]
    prov = provenance(sha, "dedup + conflicting-label removal (6-gas key)", len(df))
    write_experiment("rfrep_main", {"id": "rfrep_main", "implementation": "A", "title": "Random Forest comparative ML (replication)",
                                    "status": "completed", "kind": "replication", "provenance": prov, "seconds": round(time.time()-t, 1), **r})
    for name, gs in r["fitted"].items():
        base = {"implementation": "A", "implementation_label": "RF comparative ML", "feature_set": "rf15",
                "model_name": name, "classes": classes, "label_encoding": "index_into_classes",
                "supports_proba": hasattr(gs.best_estimator_, "predict_proba") and name != "SVM (RBF)",
                "provenance": prov, "best_params": gs.best_params_, "trained_on": "80% stratified training split",
                "metrics_ref": {"experiment": "rfrep_main", "model": name}}
        export(reg.model_id("rfrep", "rf15", name), gs.best_estimator_, {}, base)


def do_dga(raw, sha):
    t = time.time(); print("[B] DGA cases x classifiers ...")
    df = dataset.clean_dedup(raw)
    r = dga_methods.run_experiment(df)
    prov = provenance(sha, "dedup only", len(df))
    write_experiment("dga_main", {"id": "dga_main", "implementation": "B", "title": "Duval / Doernenburg / Rogers feature cases x 6 classifiers",
                                  "status": "completed", "kind": "replication", "provenance": prov, "seconds": round(time.time()-t, 1), **r})
    for (case, name), m in r["fitted"].items():
        est = m
        has_proba = hasattr(est, "predict_proba")
        base = {"implementation": "B", "implementation_label": "DGA feature cases", "feature_set": case,
                "model_name": name, "classes": r["labels"], "label_encoding": "string_labels",
                "supports_proba": has_proba, "provenance": prov, "trained_on": "80% stratified training split",
                "metrics_ref": {"experiment": "dga_main", "model": f"{case}|{name}"}}
        export(reg.model_id("dga", case, name), est, {}, base)


def do_pcaann(raw, sha, corrected=True):
    t = time.time(); print("[C] PCA + ANN ...")
    df = dataset.clean_dedup(raw)
    r = pca_ann.run_experiment(df, include_corrected=corrected)
    prov = provenance(sha, "dedup only", len(df))
    write_experiment("pcaann_main", {"id": "pcaann_main", "implementation": "C", "title": "PCA + ANN",
                                     "status": "completed", "kind": "replication", "provenance": prov, "seconds": round(time.time()-t, 1), **r})
    base = {"implementation": "C", "implementation_label": "PCA + ANN", "feature_set": "pca15",
            "model_name": f"PCA + ANN (k={r['k_best_cv']})", "classes": r["labels"], "label_encoding": "string_labels",
            "supports_proba": True, "provenance": prov, "trained_on": "80% stratified training split",
            "n_components": r["k_best_cv"], "metrics_ref": {"experiment": "pcaann_main", "model": "PCA + ANN"}}
    export(reg.model_id("pcaann", "pca15", "pca_ann"), r["fitted"]["PCA + ANN"], {}, base)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["rfrep", "dga", "pcaann"])
    ap.add_argument("--skip-corrected", action="store_true")
    a = ap.parse_args()
    raw = dataset.load_raw(config.DATASET_PATH); sha = dataset.sha256_of(config.DATASET_PATH)
    print("dataset", config.DATASET_PATH.name, sha[:12], raw.shape)
    qpath = config.PROCESSED_DIR; qpath.mkdir(parents=True, exist_ok=True)
    (qpath / "quality_report.json").write_text(json.dumps(jsonable({**dataset.quality_report(raw), "sha256": sha}), indent=1))
    if a.only in (None, "dga"): do_dga(raw, sha)
    if a.only in (None, "pcaann"): do_pcaann(raw, sha, not a.skip_corrected)
    if a.only in (None, "rfrep"): do_rfrep(raw, sha)
    print("done")
