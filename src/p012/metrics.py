"""Evaluation metrics (CLAUDE.md 4.8). All functions take probabilities of shape (n, K)."""
import numpy as np
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

from .config import load_config


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int | None = None) -> float:
    """Expected calibration error with equal-width bins on [0, 1]:
    sum over bins of (bin share) * |mean outcome - mean confidence|."""
    n_bins = n_bins or load_config()["metrics"]["ece_bins"]
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def auc(y: np.ndarray, proba: np.ndarray) -> float:
    if proba.shape[1] == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr", labels=np.arange(proba.shape[1])))


def evaluate(y: np.ndarray, proba: np.ndarray) -> dict:
    """Binary: positive-class metrics at the config threshold, ECE on P(positive).
    Multiclass: macro P/R/F1, one-vs-rest macro AUC, ECE on the top-class confidence."""
    K = proba.shape[1]
    labels = np.arange(K)
    out = {"auc": auc(y, proba), "log_loss": float(log_loss(y, proba, labels=labels))}
    if K == 2:
        pred = (proba[:, 1] >= load_config()["metrics"]["threshold"]).astype(int)
        out.update(
            accuracy=float(accuracy_score(y, pred)),
            precision=float(precision_score(y, pred, zero_division=0)),
            recall=float(recall_score(y, pred, zero_division=0)),
            f1=float(f1_score(y, pred, zero_division=0)),
            brier=float(brier_score_loss(y, proba[:, 1])),
            ece=ece(proba[:, 1], y.astype(float)),
        )
    else:
        pred = proba.argmax(axis=1)
        onehot = np.eye(K)[y]
        out.update(
            accuracy=float(accuracy_score(y, pred)),
            precision=float(precision_score(y, pred, average="macro", labels=labels, zero_division=0)),
            recall=float(recall_score(y, pred, average="macro", labels=labels, zero_division=0)),
            f1=float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
            brier=float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
            ece=ece(proba.max(axis=1), (pred == y).astype(float)),
        )
    return out
