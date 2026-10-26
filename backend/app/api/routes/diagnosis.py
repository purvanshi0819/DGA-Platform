from __future__ import annotations
import pandas as pd
from fastapi import APIRouter, HTTPException
from ...ml import model_registry as reg
from ...ml.feature_engineering import FEATURE_SETS
from ...schemas.api import DiagnosisRequest, DiagnosisResponse, ModelResult
from ...services import inference_service as inf
from ...db import database as db

router = APIRouter(tags=["models & diagnosis"])


@router.get("/models")
def models():
    """Available trained models with their feature set, class order and provenance."""
    out = []
    for m in reg.list_models():
        out.append({k: m[k] for k in ("id", "implementation", "implementation_label", "model_name", "feature_set",
                                      "classes", "supports_proba", "trained_on", "metrics_ref") if k in m}
                   | {"provenance": m.get("provenance")})
    return out


@router.get("/feature-sets")
def feature_sets():
    return [{"id": f.id, "label": f.label, "notebook": f.notebook, "gases": list(f.gases),
             "positive_required": list(f.positive), "uses_epsilon": f.uses_epsilon,
             "formulas": f.formulas, "note": f.note, "feature_names": list(f.fn(_probe()).columns)}
            for f in FEATURE_SETS.values()]


def _probe():
    return pd.DataFrame([dict(H2=1.0, CH4=1.0, C2H6=1.0, C2H4=1.0, C2H2=1.0)])


@router.post("/diagnosis", response_model=DiagnosisResponse)
def diagnose(req: DiagnosisRequest):
    reading = req.reading.model_dump()
    df = pd.DataFrame([reading])
    results = []
    for mid in dict.fromkeys(req.model_ids):
        try:
            _, meta = reg.load_model(mid)
        except reg.ArtifactError as e:
            raise HTTPException(404, str(e))
        r = inf.predict_frame(mid, df)
        base = dict(model_id=mid, implementation=meta["implementation"], model_name=meta["model_name"],
                    feature_set=meta["feature_set"])
        if not bool(r["ok"].iloc[0]):
            results.append(ModelResult(**base, status="not_computable", message=inf.message_for_invalid(meta["feature_set"])))
            continue
        pr = r["proba"][0]
        results.append(ModelResult(**base, status="ok", predicted_class=r["pred"][0], probabilities=pr,
                                   probability_note=inf.PROBA_NOTE if pr else "This classifier does not provide probabilities.",
                                   features={k: float(v) for k, v in r["features"].iloc[0].items()}))
    ok = [x for x in results if x.status == "ok"]
    votes: dict[str, int] = {}
    for x in ok: votes[x.predicted_class] = votes.get(x.predicted_class, 0) + 1
    agreement = {"n_models": len(ok), "votes": votes, "unanimous": len(votes) == 1 and len(ok) > 1}
    if ok:
        db.log_prediction("single", [x.model_id for x in results], 1, 1, {"votes": votes})
    return DiagnosisResponse(input=reading, results=results, agreement=agreement, disclaimer=inf.DISCLAIMER)


from ...schemas.api import GasReading
from ...ml.feature_engineering import computable_mask, compute_features


@router.post("/features")
def compute_all_features(reading: GasReading):
    """Feature values for every feature set for one reading; zero denominators are reported, never replaced."""
    df = pd.DataFrame([reading.model_dump()])
    out = []
    for f in FEATURE_SETS.values():
        if bool(computable_mask(f.id, df).iloc[0]):
            vals = compute_features(f.id, df).iloc[0]
            out.append({"id": f.id, "status": "ok", "values": {k: float(v) for k, v in vals.items()}})
        else:
            out.append({"id": f.id, "status": "not_computable", "message": inf.message_for_invalid(f.id)})
    return out
