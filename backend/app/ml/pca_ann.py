"""Notebook C: PCA + ANN (Nkwambe & Thango, Energies 2026).

Verbatim protocol: dedup only, 15 features, stratified 80/20 (seed 42), StandardScaler+PCA fitted on
training data inside the pipeline, ANN = 1 hidden layer of 20 tanh neurons, L-BFGS, alpha=0.01.
k = 1..15 swept; k chosen by best 5-fold CV on the training set (as the notebook's final model).

``run_comparison`` is the notebook's 10-split comparison with K fixed from the seed-42 train split
(which leaks mildly across the other splits). ``run_comparison_corrected`` re-selects k inside each
split's training data - reported separately as a corrected experiment.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from .evaluation import full_metrics, confusion, per_class
from .feature_engineering import compute_features

SEED = 42
N_SEEDS = 10


def ann(seed: int = 42):
    return MLPClassifier(hidden_layer_sizes=(20,), activation="tanh", solver="lbfgs",
                         alpha=0.01, max_iter=2000, random_state=seed)


def pca_ann_pipeline(k: int, seed: int = 42):
    return make_pipeline(StandardScaler(), PCA(n_components=k), ann(seed))


def prepare(df: pd.DataFrame):
    F = compute_features("pca15", df)
    y = df["Fault"].values
    Xtr, Xte, ytr, yte = train_test_split(F.values, y, test_size=0.2, random_state=SEED, stratify=y)
    return F, Xtr, Xte, ytr, yte


def pca_summary(F, Xtr) -> dict:
    scaler = StandardScaler().fit(Xtr)
    pca = PCA().fit(scaler.transform(Xtr))
    cum = np.cumsum(pca.explained_variance_ratio_) * 100
    return {
        "features": list(F.columns),
        "explained_variance_pct": (pca.explained_variance_ratio_ * 100).tolist(),
        "cumulative_variance_pct": cum.tolist(),
        "k_for_95": int(np.argmax(cum >= 95)) + 1, "k_for_99": int(np.argmax(cum >= 99)) + 1,
        "loadings": {f"PC{i+1}": dict(zip(F.columns, pca.components_[i].tolist())) for i in range(len(F.columns))},
        "correlation": {"features": list(F.columns), "matrix": F.corr().values.tolist(),
                        "note": "computed on the full cleaned dataset before the split (visualisation only)"},
    }


def sweep(F, Xtr, Xte, ytr, yte, cum) -> list[dict]:
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rows = []
    for k in range(1, 16):
        pipe = pca_ann_pipeline(k).fit(Xtr, ytr)
        cv = cross_val_score(pca_ann_pipeline(k), Xtr, ytr, cv=skf).mean()
        rows.append({"k": k, "cum_var_pct": float(cum[k - 1]),
                     "train_acc_pct": accuracy_score(ytr, pipe.predict(Xtr)) * 100,
                     "test_acc_pct": accuracy_score(yte, pipe.predict(Xte)) * 100, "cv5_acc_pct": cv * 100})
    return rows


def _comparison_models(F, K, seed):
    raw5 = list(F.columns[:5]); iec = ["C2H2/C2H4", "CH4/H2", "C2H4/C2H6"]
    cols = list(F.columns)
    sc = StandardScaler
    return {
        "ANN, raw 5 gases": (raw5, lambda s: make_pipeline(sc(), ann(s))),
        "ANN, IEC 3 ratios": (iec, lambda s: make_pipeline(sc(), ann(s))),
        "ANN, all 15 (no PCA)": (cols, lambda s: make_pipeline(sc(), ann(s))),
        f"PCA + ANN (k={K})": (cols, lambda s: pca_ann_pipeline(K, s)),
        "GNB (15)": (cols, lambda s: make_pipeline(sc(), GaussianNB())),
        "KNN k=3 (15)": (cols, lambda s: make_pipeline(sc(), KNeighborsClassifier(3))),
        "KNN k=5 (15)": (cols, lambda s: make_pipeline(sc(), KNeighborsClassifier(5))),
        "SVM RBF (15)": (cols, lambda s: make_pipeline(sc(), SVC(kernel="rbf"))),
    }


def run_comparison(df, F, y, K, n_seeds=N_SEEDS) -> list[dict]:
    """Verbatim notebook Cell 8 (k fixed from seed-42 train split)."""
    res = []
    for seed in range(n_seeds):
        a, b, ya, yb = train_test_split(np.arange(len(df)), y, test_size=0.2, random_state=seed, stratify=y)
        for name, (cols, mk) in _comparison_models(F, K, seed).items():
            X = F[cols].values
            m = mk(seed).fit(X[a], ya)
            res.append({"model": name, "seed": seed, "train_acc_pct": accuracy_score(ya, m.predict(X[a])) * 100,
                        "test_acc_pct": accuracy_score(yb, m.predict(X[b])) * 100})
    return res


def run_comparison_corrected(df, F, y, n_seeds=N_SEEDS) -> list[dict]:
    """PCA+ANN only: k selected by 5-fold CV inside each split's training data (no test leakage)."""
    out = []
    for seed in range(n_seeds):
        a, b, ya, yb = train_test_split(np.arange(len(df)), y, test_size=0.2, random_state=seed, stratify=y)
        X = F.values
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        cvs = [cross_val_score(pca_ann_pipeline(k, seed), X[a], ya, cv=skf).mean() for k in range(1, 16)]
        k = int(np.argmax(cvs)) + 1
        m = pca_ann_pipeline(k, seed).fit(X[a], ya)
        out.append({"seed": seed, "k_selected": k, "train_acc_pct": accuracy_score(ya, m.predict(X[a])) * 100,
                    "test_acc_pct": accuracy_score(yb, m.predict(X[b])) * 100})
    return out


def run_experiment(df: pd.DataFrame, include_corrected: bool = True) -> dict:
    F, Xtr, Xte, ytr, yte = prepare(df)
    labels = sorted(set(df["Fault"]))
    summ = pca_summary(F, Xtr)
    sw = sweep(F, Xtr, Xte, ytr, yte, np.array(summ["cumulative_variance_pct"]))
    k_paper = int(max(sw, key=lambda r: r["test_acc_pct"])["k"])
    k_cv = int(max(sw, key=lambda r: r["cv5_acc_pct"])["k"])
    final = pca_ann_pipeline(k_cv).fit(Xtr, ytr)
    yp = final.predict(Xte)
    y = df["Fault"].values
    cmp_rows = run_comparison(df, F, y, k_cv)
    out = {
        "pca": summ, "sweep": sw, "k_best_test_paper_rule": k_paper, "k_best_cv": k_cv,
        "final": {"k": k_cv, "metrics": full_metrics(yte, yp), "confusion": confusion(yte, yp, labels),
                  "per_class": per_class(yte, yp, labels), "n_train": int(len(ytr)), "n_test": int(len(yte))},
        "comparison_10splits": cmp_rows, "labels": labels, "fitted": {"PCA + ANN": final},
        "ann_config": {"hidden_layer_sizes": [20], "activation": "tanh", "solver": "lbfgs", "alpha": 0.01,
                       "max_iter": 2000, "random_state": 42},
        "split": {"test_size": 0.2, "random_state": SEED, "stratified": True, "n_rows": int(len(df)),
                  "cv": "StratifiedKFold(5, shuffle=True, random_state=42) on training set"},
    }
    if include_corrected:
        out["comparison_corrected"] = run_comparison_corrected(df, F, y)
    return out
