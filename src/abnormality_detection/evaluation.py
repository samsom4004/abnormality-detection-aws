"""Metrics suited to an imbalanced, safety-relevant task (accuracy alone is misleading)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .labels import CLASSES, NORMAL_LIKE, abnormal_indices


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    ece, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def abnormal_scores(probs: np.ndarray, classes=CLASSES, normal_like=NORMAL_LIKE) -> np.ndarray:
    return probs[:, abnormal_indices(classes, normal_like)].sum(1)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes=CLASSES, threshold: float = 0.5) -> dict:
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    y_pred = probs.argmax(1)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ece": expected_calibration_error(probs, y_true),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    abn_idx = set(abnormal_indices(classes))
    is_abn = np.array([t in abn_idx for t in y_true])
    score = abnormal_scores(probs, classes)
    flagged = score >= threshold
    a = {"threshold": threshold, "n_abnormal": int(is_abn.sum())}
    if is_abn.any():
        a["recall"] = float(flagged[is_abn].mean())  # sensitivity: abnormal images that are flagged
    if (~is_abn).any():
        a["specificity"] = float((~flagged[~is_abn]).mean())
    if is_abn.any() and (~is_abn).any():
        a["auroc"] = float(roc_auc_score(is_abn, score))
    out["abnormal_vs_normal"] = a
    return out
