"""Inference: loads the saved pipeline once, computes features exactly like training, predicts."""
from __future__ import annotations
import numpy as np, pandas as pd
from ..ml import model_registry as reg
from ..ml.feature_engineering import compute_features, computable_mask, FeatureError, FEATURE_SETS

DISCLAIMER = ("Research decision support only. Predictions are model outputs, not confirmed engineering "
              "diagnoses, and must be validated by qualified transformer engineers.")
PROBA_NOTE = ("Classifier output score, not a calibrated real-world probability (calibration has not been tested).")


def _decode(meta, raw_pred):
    if meta.get("label_encoding") == "index_into_classes":
        return [meta["classes"][int(i)] for i in raw_pred]
    return [str(p) for p in raw_pred]


def _input_matrix(meta, feats: pd.DataFrame):
    # notebook A fitted on a DataFrame (named columns); B and C fitted on numpy arrays
    return feats if meta["implementation"] == "A" else feats.values


def predict_frame(mid: str, df: pd.DataFrame) -> dict:
    """df has gas columns. Returns per-row status + predictions; rows that cannot be computed are flagged, never imputed."""
    est, meta = reg.load_model(mid)
    fs = meta["feature_set"]
    ok = computable_mask(fs, df)
    out = {"meta": meta, "ok": ok, "pred": [None] * len(df), "proba": [None] * len(df), "features": None}
    if ok.any():
        feats = compute_features(fs, df[ok], strict=True)
        X = _input_matrix(meta, feats)
        raw = est.predict(X)
        labels = _decode(meta, raw)
        pos = list(np.flatnonzero(ok.values))
        for p, lab in zip(pos, labels):
            out["pred"][p] = lab
        if meta.get("supports_proba") and hasattr(est, "predict_proba"):
            pr = est.predict_proba(X)
            cls = [meta["classes"][int(c)] for c in est.classes_] if meta.get("label_encoding") == "index_into_classes" \
                else [str(c) for c in est.classes_]
            for p, row in zip(pos, pr):
                out["proba"][p] = {c: float(v) for c, v in zip(cls, row)}
        out["features"] = feats
    return out


def message_for_invalid(fs: str) -> str:
    f = FEATURE_SETS[fs]
    return ("Cannot compute this feature set: " + ", ".join(f.positive) + " must be > 0 (the source notebook uses no epsilon, so zero values are not replaced)."
            if f.positive else "Input values must be present, numeric and non-negative.")
