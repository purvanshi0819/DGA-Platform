from __future__ import annotations
import json
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from ...ml import model_registry as reg
from ...services import dataset_service as ds, inference_service as inf
from ...db import database as db

router = APIRouter(prefix="/batch", tags=["batch"])


@router.get("/template", response_class=PlainTextResponse)
def template():
    return PlainTextResponse(ds.TEMPLATE_CSV, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=dga_template.csv"})


async def _read(file: UploadFile) -> bytes:
    data = await file.read(ds.config.MAX_UPLOAD_BYTES + 1)
    return data


@router.post("/validate")
async def validate(file: UploadFile = File(...)):
    df, issues = ds.parse_gas_csv(await _read(file))
    bad = ds.invalid_rows(issues)
    return {"n_rows": len(df), "n_valid_rows": len(df) - len(bad), "n_invalid_rows": len(bad),
            "issues": issues[:500], "issues_truncated": len(issues) > 500,
            "columns": [c for c in df.columns], "preview": json.loads(df.head(10).to_json(orient="records"))}


@router.post("/predict")
async def predict(file: UploadFile = File(...), model_ids: str = Form(..., description="comma-separated model ids")):
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    if not ids:
        raise HTTPException(422, "Select at least one model.")
    if len(ids) > 10:
        raise HTTPException(422, "At most 10 models per batch request.")
    df, issues = ds.parse_gas_csv(await _read(file))
    bad = ds.invalid_rows(issues)
    valid_input = ~(df.index + 1).isin(bad)
    rows = [{"row": int(i) + 1, **{k: (None if v != v else v) for k, v in df.loc[i, [c for c in df.columns if c != "Fault"]].items()},
             "status": "ok" if valid_input[i] else "invalid_input", "predictions": {}, "notes": []} for i in df.index]
    summaries = {}
    work = df[valid_input]
    for mid in ids:
        try:
            _, meta = reg.load_model(mid)
        except reg.ArtifactError as e:
            raise HTTPException(404, str(e))
        counts: dict[str, int] = {}
        if len(work):
            r = inf.predict_frame(mid, work)
            for pos, idx in enumerate(work.index):
                p = r["pred"][pos]
                if p is None:
                    rows[idx]["notes"].append(f"{meta['model_name']} ({meta['feature_set']}): not computable (zero denominator)")
                else:
                    rows[idx]["predictions"][mid] = p; counts[p] = counts.get(p, 0) + 1
        summaries[mid] = {"model_name": meta["model_name"], "feature_set": meta["feature_set"],
                          "implementation": meta["implementation"], "class_counts": counts,
                          "n_not_computable": int(len(work) - sum(counts.values()))}
    for r in rows:
        if r["status"] == "ok" and not r["predictions"]:
            r["status"] = "not_computable"
    db.log_prediction("batch", ids, len(df), int(valid_input.sum()), {k: v["class_counts"] for k, v in summaries.items()})
    return {"n_rows": len(df), "n_valid_rows": int(valid_input.sum()), "n_invalid_rows": len(bad),
            "issues": issues[:500], "models": summaries, "rows": rows, "disclaimer": inf.DISCLAIMER}
