"""Feature engineering ported from the three notebooks.

Formulas and epsilon policy are copied verbatim. Parity with the notebook code is
enforced by ``tests/test_notebook_parity.py`` (which executes the notebook cells).

Feature-set ids
---------------
rf15        notebook A  : 15 features (log1p gases, 6 ratios, TCG incl. H2, 3 Duval %), eps=1e-6
duval3      notebook B  : Case A  - Duval percentages of CH4/C2H4/C2H2 (no epsilon)
doern4      notebook B  : Case B  - 4 ratios used as ML inputs (no epsilon)
rogers3     notebook B  : Case C  - 3 ratios used as ML inputs (no epsilon)
pca15       notebook C  : 15 features (5 gases, TCH w/o H2, 9 ratios), eps=1e-6
raw5        notebook C  : 5 raw gases
iec3        notebook C  : 3 IEC ratios, eps=1e-6

These are ML input representations. They are NOT the rule-based Duval / Doernenburg /
Rogers diagnostic procedures.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd

EPS = 1e-6
GASES5 = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]


class FeatureError(ValueError):
    """Raised when a record cannot be turned into features without inventing values."""

    def __init__(self, message: str, rows: list[int] | None = None):
        super().__init__(message)
        self.rows = rows or []


# ---------------------------------------------------------------- notebook A
def features_rf15(d: pd.DataFrame) -> pd.DataFrame:
    e = 1e-6
    g = d[GASES5].astype(float)
    X = pd.DataFrame(index=d.index)
    for c in GASES5:
        X[f"log_{c}"] = np.log1p(g[c])
    X["R_CH4_H2"] = g.CH4 / (g.H2 + e)
    X["R_C2H2_C2H4"] = g.C2H2 / (g.C2H4 + e)
    X["R_C2H4_C2H6"] = g.C2H4 / (g.C2H6 + e)
    X["R_C2H2_CH4"] = g.C2H2 / (g.CH4 + e)
    X["R_C2H2_H2"] = g.C2H2 / (g.H2 + e)
    X["R_C2H6_C2H2"] = g.C2H6 / (g.C2H2 + e)
    X["TCG"] = g.sum(axis=1)
    s = g.CH4 + g.C2H4 + g.C2H2 + e
    X["Duval_CH4pct"] = 100 * g.CH4 / s
    X["Duval_C2H4pct"] = 100 * g.C2H4 / s
    X["Duval_C2H2pct"] = 100 * g.C2H2 / s
    return X


# ---------------------------------------------------------------- notebook B
def _g(d):
    return d[["H2", "CH4", "C2H6", "C2H4", "C2H2"]].astype(float)


def features_duval3(d: pd.DataFrame) -> pd.DataFrame:
    g = _g(d)
    s = g["CH4"] + g["C2H4"] + g["C2H2"]
    return pd.DataFrame({"pCH4": 100 * g["CH4"] / s, "pC2H4": 100 * g["C2H4"] / s,
                         "pC2H2": 100 * g["C2H2"] / s})


def features_doern4(d: pd.DataFrame) -> pd.DataFrame:
    g = _g(d)
    return pd.DataFrame({"C2H6/C2H2": g["C2H6"] / g["C2H2"], "C2H2/CH4": g["C2H2"] / g["CH4"],
                         "C2H2/C2H4": g["C2H2"] / g["C2H4"], "CH4/H2": g["CH4"] / g["H2"]})


def features_rogers3(d: pd.DataFrame) -> pd.DataFrame:
    g = _g(d)
    return pd.DataFrame({"C2H2/C2H4": g["C2H2"] / g["C2H4"], "CH4/H2": g["CH4"] / g["H2"],
                         "C2H4/C2H6": g["C2H4"] / g["C2H6"]})


# ---------------------------------------------------------------- notebook C
def features_pca15(d: pd.DataFrame) -> pd.DataFrame:
    g = d[GASES5].astype(float).copy()
    g["TCH"] = g["CH4"] + g["C2H6"] + g["C2H4"] + g["C2H2"]
    F = g.copy()
    F["CH4/H2"] = g["CH4"] / (g["H2"] + EPS)
    F["C2H2/C2H4"] = g["C2H2"] / (g["C2H4"] + EPS)
    F["C2H4/C2H6"] = g["C2H4"] / (g["C2H6"] + EPS)
    F["C2H2/CH4"] = g["C2H2"] / (g["CH4"] + EPS)
    F["C2H6/CH4"] = g["C2H6"] / (g["CH4"] + EPS)
    F["C2H4/TCH"] = g["C2H4"] / (g["TCH"] + EPS)
    F["C2H6/TCH"] = g["C2H6"] / (g["TCH"] + EPS)
    F["CH4/TCH"] = g["CH4"] / (g["TCH"] + EPS)
    F["C2H2/TCH"] = g["C2H2"] / (g["TCH"] + EPS)
    return F


def features_raw5(d: pd.DataFrame) -> pd.DataFrame:
    return features_pca15(d)[GASES5]


def features_iec3(d: pd.DataFrame) -> pd.DataFrame:
    return features_pca15(d)[["C2H2/C2H4", "CH4/H2", "C2H4/C2H6"]]


@dataclass(frozen=True)
class FeatureSet:
    id: str
    label: str
    notebook: str
    fn: Callable[[pd.DataFrame], pd.DataFrame]
    gases: tuple[str, ...]            # gases that must be present
    positive: tuple[str, ...]          # gases that must be > 0 (denominators) -> no epsilon in source
    uses_epsilon: bool
    formulas: dict[str, str]
    note: str = ""


_RATIO_NOTE = ("ML input representation taken from the notebook; not the rule-based "
               "diagnostic procedure of the same name.")

FEATURE_SETS: dict[str, FeatureSet] = {
    "rf15": FeatureSet("rf15", "Engineered 15 (RF replication)", "A", features_rf15, tuple(GASES5), (), True, {
        "log_<gas>": "log(1 + gas) for H2, CH4, C2H6, C2H4, C2H2",
        "R_CH4_H2": "CH4 / (H2 + 1e-6)", "R_C2H2_C2H4": "C2H2 / (C2H4 + 1e-6)",
        "R_C2H4_C2H6": "C2H4 / (C2H6 + 1e-6)", "R_C2H2_CH4": "C2H2 / (CH4 + 1e-6)",
        "R_C2H2_H2": "C2H2 / (H2 + 1e-6)", "R_C2H6_C2H2": "C2H6 / (C2H2 + 1e-6)",
        "TCG": "H2 + CH4 + C2H6 + C2H4 + C2H2 (includes H2)",
        "Duval_<gas>pct": "100 * gas / (CH4 + C2H4 + C2H2 + 1e-6) for CH4, C2H4, C2H2"}),
    "duval3": FeatureSet("duval3", "Duval triangle percentages", "B", features_duval3, tuple(GASES5),
                         ("CH4+C2H4+C2H2",), False,
                         {"pCH4": "100*CH4/(CH4+C2H4+C2H2)", "pC2H4": "100*C2H4/(CH4+C2H4+C2H2)",
                          "pC2H2": "100*C2H2/(CH4+C2H4+C2H2)"},
                         "Percentages used as classifier inputs; the triangle's zone polygons are not implemented."),
    "doern4": FeatureSet("doern4", "Doernenburg ratios", "B", features_doern4, tuple(GASES5),
                         ("C2H2", "CH4", "C2H4", "H2"), False,
                         {"C2H6/C2H2": "C2H6/C2H2", "C2H2/CH4": "C2H2/CH4", "C2H2/C2H4": "C2H2/C2H4",
                          "CH4/H2": "CH4/H2"}, _RATIO_NOTE),
    "rogers3": FeatureSet("rogers3", "Rogers ratios", "B", features_rogers3, tuple(GASES5),
                          ("C2H4", "H2", "C2H6"), False,
                          {"C2H2/C2H4": "C2H2/C2H4", "CH4/H2": "CH4/H2", "C2H4/C2H6": "C2H4/C2H6"}, _RATIO_NOTE),
    "pca15": FeatureSet("pca15", "Engineered 15 (PCA+ANN)", "C", features_pca15, tuple(GASES5), (), True, {
        "H2..C2H2": "raw concentrations", "TCH": "CH4 + C2H6 + C2H4 + C2H2 (excludes H2)",
        "9 ratios": "numerator / (denominator + 1e-6): CH4/H2, C2H2/C2H4, C2H4/C2H6, C2H2/CH4, C2H6/CH4, "
                    "C2H4/TCH, C2H6/TCH, CH4/TCH, C2H2/TCH"}),
    "raw5": FeatureSet("raw5", "Raw 5 gases", "C", features_raw5, tuple(GASES5), (), False,
                       {g: "raw concentration" for g in GASES5}),
    "iec3": FeatureSet("iec3", "IEC three ratios", "C", features_iec3, tuple(GASES5), (), True,
                       {"C2H2/C2H4": "C2H2/(C2H4+1e-6)", "CH4/H2": "CH4/(H2+1e-6)",
                        "C2H4/C2H6": "C2H4/(C2H6+1e-6)"}),
}


def input_validity(d: pd.DataFrame, required: tuple[str, ...] = tuple(GASES5)) -> pd.Series:
    """Rows with all required gases present, numeric, finite and non-negative."""
    missing_cols = [c for c in required if c not in d.columns]
    if missing_cols:
        raise FeatureError(f"Missing required gas columns: {missing_cols}")
    num = d[list(required)].apply(pd.to_numeric, errors="coerce")
    return num.notna().all(axis=1) & np.isfinite(num).all(axis=1) & (num >= 0).all(axis=1)


def computable_mask(set_id: str, d: pd.DataFrame) -> pd.Series:
    """True where the feature set can be computed WITHOUT inventing values.

    Sets whose source has no epsilon (duval3, doern4, rogers3) need strictly positive
    denominators; a zero is reported as invalid instead of being silently replaced.
    """
    fs = FEATURE_SETS[set_id]
    ok = input_validity(d)
    g = d[GASES5].apply(pd.to_numeric, errors="coerce")
    if set_id == "duval3":
        ok &= (g["CH4"] + g["C2H4"] + g["C2H2"]) > 0
    else:
        for c in fs.positive:
            ok &= g[c] > 0
    return ok


def compute_features(set_id: str, d: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    if set_id not in FEATURE_SETS:
        raise FeatureError(f"Unknown feature set '{set_id}'")
    fs = FEATURE_SETS[set_id]
    ok = computable_mask(set_id, d)
    if strict and not ok.all():
        bad = [int(i) for i in d.index[~ok]]
        raise FeatureError(
            f"Feature set '{set_id}' cannot be computed for {len(bad)} record(s): values must be "
            "present, numeric, non-negative" + (", and denominators must be > 0" if fs.positive else "") + ".",
            rows=bad)
    sub = d[ok] if not strict else d
    sub = sub.copy()
    sub[GASES5] = sub[GASES5].apply(pd.to_numeric)
    return fs.fn(sub)
