"""Notebook A: RF / KNN / SVM(RBF) / ANN with SMOTE inside the pipeline + GridSearchCV.

Protocol (verbatim): dedup + conflicting-label removal, 15 engineered features, stratified 80/20
(seed 42), RepeatedStratifiedKFold(3x3) grid search on the training part only, test on held-out 20%.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from .evaluation import full_metrics, confusion, per_class
from .feature_engineering import compute_features

SEED = 42
MODEL_IDS = ["Random Forest", "KNN", "SVM (RBF)", "ANN"]


def make_specs(seed: int = SEED):
    sm = lambda: SMOTE(k_neighbors=1, random_state=seed)
    return {
        "Random Forest": (Pipeline([("smote", sm()), ("m", RandomForestClassifier(random_state=seed, n_jobs=-1))]),
                          {"m__n_estimators": [50, 100, 200, 300, 500], "m__min_samples_leaf": [1, 2, 5],
                           "m__max_features": ["sqrt", None]}),
        "KNN": (Pipeline([("sc", StandardScaler()), ("smote", sm()), ("m", KNeighborsClassifier())]),
                {"m__n_neighbors": list(range(1, 26, 2))}),
        "SVM (RBF)": (Pipeline([("sc", StandardScaler()), ("smote", sm()),
                                ("m", SVC(kernel="rbf", decision_function_shape="ovo"))]),
                      {"m__C": [0.1, 1, 10, 100], "m__gamma": ["scale", 0.01, 0.1, 1]}),
        "ANN": (Pipeline([("sc", StandardScaler()), ("smote", sm()),
                          ("m", MLPClassifier(hidden_layer_sizes=(20, 10), early_stopping=True,
                                              max_iter=2000, random_state=seed))]),
                {"m__alpha": [1e-4, 1e-3, 1e-2], "m__learning_rate_init": [1e-3, 1e-2]}),
    }


def run_experiment(df: pd.DataFrame, n_jobs: int = -1) -> dict:
    """df: dedup + conflict-removed data (notebook A protocol)."""
    X = compute_features("rf15", df)
    classes = sorted(df["Fault"].unique())
    y = df["Fault"].map({c: i for i, c in enumerate(classes)}).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=SEED)
    idx = list(range(len(classes)))
    records, cms, pcs, best, preds = [], {}, {}, {}, {}
    for name, (pipe, grid) in make_specs().items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=n_jobs).fit(X_tr, y_tr)
        best[name] = gs
        p = gs.predict(X_te); preds[name] = p
        base = {"feature_set": "rf15", "model": name}
        records.append({**base, "stage": "gridsearch_cv", "n": int(len(y_tr)),
                        "metrics": {"accuracy": float(gs.best_score_)}, "best_params": gs.best_params_})
        records.append({**base, "stage": "test", "n": int(len(y_te)), "metrics": full_metrics(y_te, p)})
        cms[name] = confusion(y_te, p, idx) | {"labels": classes}
        pcs[name] = {classes[k]: v for k, v in per_class(y_te, p, idx).items()}
    rf = best["Random Forest"]
    pi = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=SEED, scoring="accuracy")
    importance = sorted(({"feature": c, "importance": float(m), "std": float(s)}
                         for c, m, s in zip(X.columns, pi.importances_mean, pi.importances_std)),
                        key=lambda r: -r["importance"])
    # OOB curve (notebook Step 10)
    bp = {k.replace("m__", ""): v for k, v in rf.best_params_.items()}
    bp.pop("n_estimators", None)
    Xs, ys = SMOTE(k_neighbors=1, random_state=SEED).fit_resample(X_tr, y_tr)
    est = RandomForestClassifier(n_estimators=1, warm_start=True, oob_score=True, bootstrap=True,
                                 random_state=SEED, n_jobs=n_jobs, **bp)
    oob = []
    for n in range(20, 501, 10):
        est.set_params(n_estimators=n); est.fit(Xs, ys); oob.append({"trees": n, "oob_error": float(1 - est.oob_score_)})
    return {"records": records, "confusion": cms, "per_class": pcs, "labels": classes,
            "feature_importance": importance, "oob_curve": oob, "fitted": best,
            "split": {"test_size": 0.2, "random_state": SEED, "stratified": True, "n_rows": int(len(df)),
                      "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                      "train_class_counts": np.bincount(y_tr).tolist(), "test_class_counts": np.bincount(y_te).tolist(),
                      "cv": "RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=42)"}}
