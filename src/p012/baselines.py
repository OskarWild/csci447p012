"""Baselines (CLAUDE.md 4.5): XGBoost, TabPFN, Selection, Stacking."""
import time

import numpy as np
import pandas as pd

from .config import load_config
from .metrics import auc
from .pfn_boost import Result, Task, _result, tune_trees
from .tabpfn_scores import clip_renorm


def xgb_baseline(task: Task, kind="main", rounds=None, early=True, name="XGB"):
    """Plain XGBoost tuned with `trials_xgb` trials. Binary: base_score = mean(y_train), so it equals
    PFN-Boost at s = 0. Returns (Result, history) - the history feeds PFNB-Optuna's stage 1."""
    t0 = time.perf_counter()
    base = float(task.y_tr.mean()) if task.n_classes == 2 else None
    tf, history = tune_trees(task, None, load_config()["hpo"]["trials_xgb"], kind, rounds, early, base)
    return _result(name, task, tf, None, t0), history


def tabpfn_baseline(task: Task, seconds: float = 0.0) -> Result:
    """TabPFN alone: probabilities from one fit on the training rows (no HPO).
    `seconds` is the TabPFN scoring time measured when the cache was built."""
    return Result("TabPFN", clip_renorm(task.p_va), clip_renorm(task.p_te), seconds)


def selection(xgb: Result, tab: Result, y_va: np.ndarray) -> Result:
    """Whichever of XGB and TabPFN has the higher validation AUC."""
    pick = xgb if auc(y_va, xgb.p_val) >= auc(y_va, tab.p_val) else tab
    return Result("Selection", pick.p_val, pick.p_test, xgb.seconds + tab.seconds, {"picked": pick.method})


def _with_probs(X: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    X = X.copy()
    for k in range(p.shape[1]):
        X[f"tabpfn_p{k}"] = p[:, k]
    return X


def stacking(task: Task) -> Result:
    """XGBoost on [X, TabPFN probabilities]: out-of-fold probabilities for training rows,
    full-fit probabilities for validation and test rows. Same budget as XGB."""
    t0 = time.perf_counter()
    stacked = Task(_with_probs(task.X_tr, clip_renorm(task.p_tr)), task.y_tr,
                   _with_probs(task.X_va, clip_renorm(task.p_va)), task.y_va,
                   _with_probs(task.X_te, clip_renorm(task.p_te)), task.y_te,
                   task.n_classes, task.seed, task.p_tr, task.p_va, task.p_te)
    base = float(task.y_tr.mean()) if task.n_classes == 2 else None
    tf, _ = tune_trees(stacked, None, load_config()["hpo"]["trials_xgb"], base_score=base)
    res = _result("Stacking", stacked, tf, None, t0)
    return res
