"""TabPFN probabilities for a split: cross-fitted (out-of-fold) training scores, full-fit val/test
scores, conversion to centred log-probabilities (CLAUDE.md 4.1-4.2), and an .npz disk cache."""
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split

from .config import load_config, path

log = logging.getLogger(__name__)

# fit_predict(X_context, y_context, X_query, seed) -> probabilities (n_query, K) for classes 0..K-1.
FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]

_models = {}


def _tabpfn(seed: int):
    from tabpfn import TabPFNClassifier
    cfg = load_config()["tabpfn"]
    device = "cuda" if cfg["device"] == "auto" and _cuda() else ("cpu" if cfg["device"] == "auto" else cfg["device"])
    key = (device, seed)
    if key not in _models:
        _models[key] = TabPFNClassifier(device=device, N_ensemble_configurations=cfg["n_ensemble"], seed=seed)
    return _models[key]


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def subsample_context(y: np.ndarray, seed: int, max_context: int) -> np.ndarray:
    """Row positions of at most max_context context rows (stratified when possible)."""
    idx = np.arange(len(y))
    if len(y) <= max_context:
        return idx
    strat = y if np.bincount(y).min() >= 2 else None
    keep, _ = train_test_split(idx, train_size=max_context, random_state=seed, stratify=strat)
    return np.sort(keep)


def tabpfn_fit_predict(Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray, seed: int, n_classes: int) -> np.ndarray:
    """Fit TabPFN on (Xc, yc) and return P(class k) for k = 0..n_classes-1 on Xq.
    Classes missing from the context get probability 0 here (clipped later)."""
    if len(np.unique(yc)) < 2:
        # Single-class context (leave-one-out fold that holds out the only row of a class, bank n=10):
        # TabPFN cannot fit, and there is nothing to compare, so return uniform scores (z = 0).
        log.info("single-class TabPFN context (n=%d): uniform scores for %d row(s)", len(yc), len(Xq))
        return np.full((len(Xq), n_classes), 1.0 / n_classes)
    keep = subsample_context(yc, seed, load_config()["tabpfn"]["max_context"])
    model = _tabpfn(seed)
    model.fit(Xc[keep], yc[keep], overwrite_warning=True)
    # Query rows attend only to the context and normalisation uses the context only, so chunking the
    # query changes nothing but peak memory (one pass over ~38k jungle rows was killed for OOM).
    chunk = load_config()["tabpfn"]["predict_chunk"]
    p = np.vstack([model.predict_proba(Xq[i:i + chunk]) for i in range(0, len(Xq), chunk)])
    out = np.zeros((len(Xq), n_classes))
    out[:, np.asarray(model.classes_, dtype=int)] = p   # align TabPFN's class order with our codes
    return out


def clip_renorm(p: np.ndarray) -> np.ndarray:
    eps = load_config()["tabpfn"]["prob_clip"]
    p = np.clip(p, eps, 1 - eps)
    return p / p.sum(axis=1, keepdims=True)


def logits(p: np.ndarray) -> np.ndarray:
    """Centred per-class log-probabilities z (n, K), CLAUDE.md 4.1."""
    z = np.log(clip_renorm(p))
    return z - z.mean(axis=1, keepdims=True)


def binary_logit(p: np.ndarray) -> np.ndarray:
    """z1 = log p1 - log p0 (equals z[:,1] - z[:,0])."""
    z = logits(p)
    return z[:, 1] - z[:, 0]


def crossfit_folds(y: np.ndarray, seed: int) -> tuple[list, str]:
    """(train_part, held_out) position pairs. Stratified K_cf folds, or leave-one-out when the
    folds would hold fewer than 2 rows per class (tiny sizes)."""
    k = load_config()["tabpfn"]["crossfit_folds"]
    n_classes = len(np.unique(y))
    if len(y) / k < 2 * n_classes or np.bincount(y).min() < k:
        return list(LeaveOneOut().split(y)), "loo"
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)), f"skf{k}"


def compute_scores(X_tr, y_tr, X_va, X_te, seed: int, n_classes: int, mode: str = "oof",
                   fit_predict: FitPredict | None = None) -> dict:
    """TabPFN probabilities for train/val/test.
    mode 'oof': each training row is scored by a TabPFN whose context excludes it.
    mode 'insample': training rows are scored by the TabPFN that holds them in context (original code)."""
    fp = fit_predict or (lambda Xc, yc, Xq, s: tabpfn_fit_predict(Xc, yc, Xq, s, n_classes))
    X_tr, X_va, X_te = (np.asarray(a, dtype=float) for a in (X_tr, X_va, X_te))
    t0 = time.perf_counter()
    full = fp(X_tr, y_tr, np.vstack([X_tr, X_va, X_te]), seed)
    n_tr, n_va = len(X_tr), len(X_va)
    out = {"p_val": full[n_tr:n_tr + n_va], "p_test": full[n_tr + n_va:], "cf_scheme": "insample",
           "seconds_full": time.perf_counter() - t0, "seconds_crossfit": 0.0}
    if mode == "insample":
        out["p_train"] = full[:n_tr]
    elif mode == "oof":
        folds, scheme = crossfit_folds(y_tr, seed)
        if scheme == "loo":
            log.info("cross-fitting falls back to leave-one-out (n_train=%d)", n_tr)
        t0 = time.perf_counter()
        p_tr = np.full((n_tr, n_classes), np.nan)
        for rest, held in folds:
            p_tr[held] = fp(X_tr[rest], y_tr[rest], X_tr[held], seed)
        out["p_train"], out["cf_scheme"] = p_tr, scheme
        out["seconds_crossfit"] = time.perf_counter() - t0
    else:
        raise ValueError(mode)
    return out


def cache_file(protocol: str, dataset: str, size: str, seed: int, mode: str) -> Path:
    return path("cache") / "tabpfn" / protocol / dataset / str(size) / str(seed) / f"{mode}.npz"


def save_scores(f: Path, scores: dict, idx: dict) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **scores, **{f"idx_{k}": v for k, v in idx.items()})


def load_scores(f: Path) -> dict:
    with np.load(f, allow_pickle=False) as d:
        return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def get_scores(split, ds, protocol: str, size: str, seed: int, mode: str = "oof") -> dict:
    """Cached TabPFN probabilities for a data.Split; recomputed if the cached row indices differ
    (or the file predates the timing fields)."""
    f = cache_file(protocol, ds.name, size, seed, mode)
    idx = {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}
    if f.exists():
        d = load_scores(f)
        if "seconds_full" in d and all(np.array_equal(d[f"idx_{k}"], v) for k, v in idx.items()):
            return d
    X_tr, y_tr = split.part("train")
    scores = compute_scores(X_tr, y_tr, split.part("val")[0], split.part("test")[0], seed, ds.n_classes, mode)
    save_scores(f, scores, idx)
    return load_scores(f)
