from __future__ import annotations
import io, json
import numpy as np, pandas as pd
from fastapi import HTTPException
from .. import config
from ..ml import dataset
from ..ml.feature_engineering import GASES5

REQUIRED = GASES5
TEMPLATE_CSV = "H2,CH4,C2H6,C2H4,C2H2,CO\n127,24,5,32,81,60\n54,8,3,4,1,106\n"


def parse_gas_csv(content: bytes) -> tuple[pd.DataFrame, list[dict]]:
    """Validate and parse an uploaded CSV. Returns (df with numeric gas columns incl. NaN for bad cells, issues)."""
    if len(content) == 0:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds the {config.MAX_UPLOAD_BYTES // (1024*1024)} MB limit.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "File is not valid UTF-8 text; please upload a CSV file.")
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    df.columns = [str(c).strip() for c in df.columns]
    canon = {c.lower(): c for c in REQUIRED + ["CO", "Fault"]}
    df = df.rename(columns={c: canon[c.lower()] for c in df.columns if c.lower() in canon})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Missing required column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED)}.")
    if len(df) == 0:
        raise HTTPException(422, "The CSV has a header but no data rows.")
    if len(df) > config.MAX_UPLOAD_ROWS:
        raise HTTPException(413, f"Too many rows ({len(df)}); limit is {config.MAX_UPLOAD_ROWS}.")
    issues = []
    cols = REQUIRED + (["CO"] if "CO" in df.columns else [])
    num = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c]
        n = pd.to_numeric(s, errors="coerce")
        for i in df.index[s.isna() | (s.astype(str).str.strip() == "")]:
            issues.append({"row": int(i) + 1, "column": c, "issue": "missing value"})
        for i in df.index[n.isna() & s.notna() & (s.astype(str).str.strip() != "")]:
            issues.append({"row": int(i) + 1, "column": c, "issue": f"non-numeric value '{s[i]}'"})
        for i in df.index[np.isinf(n)]:
            issues.append({"row": int(i) + 1, "column": c, "issue": "non-finite value"})
        for i in df.index[n < 0]:
            issues.append({"row": int(i) + 1, "column": c, "issue": f"negative concentration ({n[i]})"})
        num[c] = n
    if "Fault" in df.columns:
        num["Fault"] = df["Fault"]
    return num, issues


def invalid_rows(issues: list[dict]) -> set[int]:
    return {i["row"] for i in issues}


def dataset_summary() -> dict:
    raw = dataset.load_raw(config.DATASET_PATH)
    q = dataset.quality_report(raw)
    q["sha256"] = dataset.sha256_of(config.DATASET_PATH)
    hist = {}
    for c in dataset.GASES6:
        v = np.log10(raw[c].clip(lower=1e-3))
        counts, edges = np.histogram(v, bins=12)
        hist[c] = [{"bin": f"{10**edges[i]:.3g}–{10**edges[i+1]:.3g}", "count": int(counts[i])} for i in range(len(counts))]
    q["gas_histograms_log10"] = hist
    q["gas_stats"] = {c: {k: float(v) for k, v in raw[c].describe().items()} for c in dataset.GASES6}
    q["provenance"] = {"file": config.DATASET_PATH.name, "source_url": config.DATASET_URL,
                       "note": "Raw file is stored unchanged; cleaned views are derived in memory per notebook protocol.",
                       "cleaning_actions": [
                           "Label whitespace stripped (no effect on this file)",
                           f"{q['duplicate_rows']} exact duplicate rows removed (notebooks A, B, C)",
                           f"{q['conflicting_label_rows_6gas_key']} conflicting-label rows removed (notebook A only)"]}
    return q
