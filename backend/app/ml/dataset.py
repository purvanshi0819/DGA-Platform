"""Dataset loading, hashing, and the two cleaning protocols used by the notebooks.

Raw data is never modified. Cleaned views are derived in memory.

* ``clean_dedup``   : notebooks B and C  -> strip labels, drop exact duplicates.
* ``clean_conflict``: notebook A         -> drop duplicates, then drop every row whose
                      (H2, CH4, C2H6, C2H4, C2H2, CO) vector appears with >1 label.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

GASES5 = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
GASES6 = GASES5 + ["CO"]
LABEL = "Fault"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in GASES6 + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df[LABEL] = df[LABEL].astype(str).str.strip()
    return df


def clean_dedup(raw: pd.DataFrame) -> pd.DataFrame:
    """Protocol of notebooks B and C."""
    return raw.drop_duplicates().reset_index(drop=True)


def clean_conflict(raw: pd.DataFrame) -> pd.DataFrame:
    """Protocol of notebook A (dedup + conflicting-label removal, CO included in the key)."""
    d = raw.drop_duplicates()
    conflict = d.groupby(GASES6)[LABEL].transform("nunique") > 1
    return d[~conflict].reset_index(drop=True)


def quality_report(raw: pd.DataFrame) -> dict:
    d = raw.drop_duplicates()
    c6 = d.groupby(GASES6)[LABEL].transform("nunique") > 1
    c5 = d.groupby(GASES5)[LABEL].transform("nunique") > 1
    a = clean_conflict(raw)
    return {
        "raw_rows": int(len(raw)),
        "columns": list(raw.columns),
        "missing_values": {c: int(v) for c, v in raw.isna().sum().items()},
        "duplicate_rows": int(raw.duplicated().sum()),
        "conflicting_label_rows_6gas_key": int(c6.sum()),
        "conflicting_label_rows_5gas_key": int(c5.sum()),
        "zero_values": {c: int((raw[c] == 0).sum()) for c in GASES6},
        "min_values": {c: float(raw[c].min()) for c in GASES6},
        "class_counts_raw": raw[LABEL].value_counts().to_dict(),
        "protocols": {
            "dedup_only (notebooks B, C)": {
                "rows": int(len(d)), "class_counts": d[LABEL].value_counts().to_dict()},
            "dedup_and_conflict_removal (notebook A)": {
                "rows": int(len(a)), "class_counts": a[LABEL].value_counts().to_dict()},
        },
    }
