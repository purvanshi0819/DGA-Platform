from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from ...services import experiment_service as es
from ...db import database as db
from ...ml import model_registry as reg

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
def experiments():
    return es.registry()


@router.get("/experiments/{eid}")
def experiment(eid: str):
    return es.load(eid)


@router.get("/experiments/{eid}/metrics")
def metrics(eid: str):
    return es.metric_rows(eid)


@router.get("/experiments/{eid}/confusion")
def confusion(eid: str, model: str = Query(..., description="model key, e.g. 'duval3|RF' or 'Random Forest'")):
    e = es.load(eid)
    cm = e.get("confusion", {}).get(model) or (e.get("final", {}).get("confusion") if model.startswith("PCA") else None)
    if cm is None:
        raise HTTPException(404, f"No confusion matrix for '{model}' in {eid}.")
    pc = (e.get("per_class") or {}).get(model) or (e.get("final", {}).get("per_class") if model.startswith("PCA") else None)
    return {"experiment": eid, "model": model, "confusion": cm, "per_class": pc}


@router.get("/experiments/{eid}/feature-importance")
def feature_importance(eid: str):
    e = es.load(eid)
    if "feature_importance" not in e:
        raise HTTPException(404, "Feature importance was only computed for the RF replication (notebook A).")
    return {"method": "permutation importance on the held-out test set (accuracy, 20 repeats)", "items": e["feature_importance"], "oob_curve": e.get("oob_curve")}


@router.get("/experiments/{eid}/export", response_class=PlainTextResponse)
def export(eid: str):
    return PlainTextResponse(es.export_csv(eid), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={eid}_metrics.csv"})


@router.get("/pca/summary")
def pca_summary():
    e = es.load("pcaann_main")
    return {k: e[k] for k in ("pca", "sweep", "k_best_test_paper_rule", "k_best_cv", "final", "ann_config", "split",
                              "comparison_10splits", "provenance") if k in e} | {"comparison_corrected": e.get("comparison_corrected"),
            "rerun_supported": False,
            "note": "Component-count selection shows precomputed results from the training run; re-running k interactively is not implemented."}


@router.get("/overview")
def overview():
    ds_rows = es.registry()
    models = reg.list_models()
    return {"n_experiments": len(ds_rows), "n_models": len(models), "n_implementations": len({m["implementation"] for m in models}),
            "latest_experiment": ds_rows[0] if ds_rows else None, "n_predictions_logged": db.prediction_count(),
            "recent_predictions": db.recent_predictions(8)}
