"""Executes the ORIGINAL notebook cells and checks the ported modules reproduce them exactly."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, pytest
from pandas.testing import assert_frame_equal
from app import config
from app.ml import dataset, feature_engineering as fe

NB = Path(__file__).resolve().parents[1] / "notebooks"


def cells(name):
    nb = json.loads((NB / name).read_text())
    out = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = c["source"]
            out.append("".join(s) if isinstance(s, list) else s)
    return out


def run(name, pick, ns=None, cwd=None):
    ns = ns or {}
    old = os.getcwd()
    if cwd: os.chdir(cwd)
    try:
        for src in cells(name):
            if pick(src):
                exec(compile(src, name, "exec"), ns)
    finally:
        os.chdir(old)
    return ns


def test_cleaning_notebook_A(raw):
    ns = run("DGA_RandomForest_Replication.ipynb",
             lambda s: "conflict = " in s or "def make_features" in s,
             {"df": raw.copy(), "np": np, "pd": pd})
    mine = dataset.clean_conflict(raw)
    assert_frame_equal(ns["df"], mine)
    assert len(mine) == 442
    assert_frame_equal(ns["X"], fe.features_rf15(mine))
    assert ns["classes"] == sorted(mine["Fault"].unique())


def test_features_notebook_B(raw):
    ns = run("DGA_transformer_fault_colab_clean.ipynb",
             lambda s: "Cell 1" in s or "Cell 2" in s, cwd=config.DATASET_PATH.parent)
    d = dataset.clean_dedup(raw)
    assert_frame_equal(ns["df"], d)
    assert len(d) == 452
    assert_frame_equal(ns["A"], fe.features_duval3(d))
    assert_frame_equal(ns["B"], fe.features_doern4(d))
    assert_frame_equal(ns["C"], fe.features_rogers3(d))


def test_features_notebook_C(raw):
    ns = run("PCA_ANN_paper3_colab.ipynb", lambda s: "Cell 1" in s or "Cell 2" in s,
             cwd=config.DATASET_PATH.parent)
    d = dataset.clean_dedup(raw)
    assert_frame_equal(ns["df"], d)
    assert_frame_equal(ns["F"], fe.features_pca15(d))
    assert_frame_equal(ns["RAW5"], fe.features_raw5(d))
    assert_frame_equal(ns["IEC3"], fe.features_iec3(d))


def test_model_parameters_notebook_B():
    from app.ml import dga_methods as m
    ns = run("DGA_transformer_fault_colab_clean.ipynb", lambda s: "Cell 1" in s or "Cell 3" in s,
             cwd=config.DATASET_PATH.parent)
    a, b = ns["models"](), m.make_models()
    assert a.keys() == b.keys()
    for k in a:
        assert repr(a[k]) == repr(b[k]), k
