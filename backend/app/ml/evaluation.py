"""Metric computation. Both macro and weighted averages are produced from the same
predictions so the UI can label the averaging explicitly. Notebook A reports macro,
notebook B reports weighted (+ specificity, MCC), notebook C reports accuracy."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
                             confusion_matrix, classification_report)


def specificity_per_class(yt, yp, labels):
    cm = confusion_matrix(yt, yp, labels=labels)
    tot = cm.sum()
    out = []
    for i in range(len(labels)):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp; tn = tot - tp - fp - fn
        out.append(tn / (tn + fp) if tn + fp else 0.0)
    return np.array(out), cm.sum(axis=1)


def specificity_weighted(yt, yp):
    """Verbatim from notebook B (labels = classes present in yt)."""
    labs = np.unique(yt)
    spec, support = specificity_per_class(yt, yp, labs)
    return float(np.average(spec, weights=support))


def full_metrics(yt, yp, labels=None) -> dict:
    """Return all metrics as fractions in [0, 1]."""
    yt = np.asarray(yt); yp = np.asarray(yp)
    pm, rm, fm, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro": {"precision": float(pm), "recall": float(rm), "f1": float(fm)},
        "weighted": {"precision": float(pw), "recall": float(rw), "f1": float(fw)},
        "specificity_weighted": specificity_weighted(yt, yp),
        "mcc": float(matthews_corrcoef(yt, yp)),
    }


def confusion(yt, yp, labels) -> dict:
    return {"labels": list(labels), "matrix": confusion_matrix(yt, yp, labels=labels).tolist()}


def per_class(yt, yp, labels) -> dict:
    rep = classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0)
    return {l: {k: float(rep[str(l)][k]) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[str(l)]["support"])}
            for l in labels}
