"""Notebook B: Duval / Doernenburg / Rogers feature cases x six classifiers.

Protocol (verbatim): drop duplicates (no conflict removal), stratified 80/20 split with
random_state=42, 10-fold StratifiedKFold(shuffle, 42) cross_val_predict on the 80%, then fit on
the 80% and predict the 20%. Models use the Table-4 parameters.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from .evaluation import full_metrics, confusion, per_class
from .feature_engineering import compute_features

CASES = {"duval3": "A (Duval)", "doern4": "B (Doernenburg)", "rogers3": "C (Rogers)"}
MODEL_IDS = ["RF", "BPNN", "KNN", "SVM", "DT", "NB"]
SEED = 42


def make_models() -> dict:
    return {
        "RF": RandomForestClassifier(n_estimators=25, max_depth=3, min_samples_split=10,
                                     max_features=None, random_state=42),
        "BPNN": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,),
                              activation="logistic", alpha=0.005, solver="sgd",
                              max_iter=1000, random_state=42)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(5, metric="euclidean", weights="uniform")),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, tol=1e-3)),
        "DT": DecisionTreeClassifier(min_samples_leaf=2, max_depth=100, min_samples_split=5, random_state=42),
        "NB": make_pipeline(StandardScaler(), GaussianNB(var_smoothing=2e-8)),
    }


def split_case(df: pd.DataFrame, case: str):
    X = compute_features(case, df)
    y = df["Fault"].values
    Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=0.2, random_state=SEED, stratify=y)
    return X, Xtr, Xte, ytr, yte


def run_experiment(df: pd.DataFrame) -> dict:
    """df: dedup-cleaned data (notebook B protocol). Returns records + confusion matrices."""
    labels = sorted(set(df["Fault"]))
    records, cms, class_metrics, warns, fitted = [], {}, {}, [], {}
    for case in CASES:
        X, Xtr, Xte, ytr, yte = split_case(df, case)
        skf = StratifiedKFold(10, shuffle=True, random_state=42)
        for name, m in make_models().items():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ycv = cross_val_predict(m, Xtr, ytr, cv=skf)
                m.fit(Xtr, ytr)
                yp = m.predict(Xte)
            for x in {str(i.message) for i in w}:
                warns.append({"case": case, "model": name, "warning": x})
            base = {"feature_set": case, "model": name}
            records.append({**base, "stage": "cv10_train", "n": int(len(ytr)),
                            "metrics": full_metrics(ytr, ycv)})
            records.append({**base, "stage": "test", "n": int(len(yte)), "metrics": full_metrics(yte, yp)})
            cms[f"{case}|{name}"] = confusion(yte, yp, labels)
            class_metrics[f"{case}|{name}"] = per_class(yte, yp, labels)
            fitted[(case, name)] = m
    return {"records": records, "confusion": cms, "per_class": class_metrics, "labels": labels,
            "warnings": warns, "fitted": fitted,
            "split": {"test_size": 0.2, "random_state": SEED, "stratified": True,
                      "n_rows": int(len(df)), "cv": "StratifiedKFold(10, shuffle=True, random_state=42)"}}
