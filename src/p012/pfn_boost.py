"""PFN-Boost via XGBoost `base_margin` (CLAUDE.md 4.3) and the learned scaling s (4.4).

The transformer's centred log-probabilities z act as a frozen "tree 0":
  binary      margin = b + s * z1            (z1 = log p1 - log p0)
  multiclass  margin = s * z                 (shape (n, K))
s = 0 with b = logit(mean(y_train)) is exactly plain XGBoost with base_score = mean(y_train).
"""
import time
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_softmax, logit, softmax

from . import hpo
from .config import load_config
from .metrics import auc
from .tabpfn_scores import binary_logit, logits


@dataclass
class Fusion:
    """The transformer part of the margin: s, intercept b (binary only)."""
    s: float
    b: float = 0.0

    def margin(self, z: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        """z is z1 (n,) for binary or z (n, K) for multiclass."""
        m = self.b + self.s * z if z.ndim == 1 else self.s * z
        return m if offset is None else m + offset


def b0(y: np.ndarray) -> float:
    return float(logit(np.clip(y.mean(), 1e-6, 1 - 1e-6)))


def base_params(n_classes: int, seed: int) -> dict:
    cfg = load_config()["xgb"]
    p = {"tree_method": cfg["tree_method"], "nthread": cfg["nthread"], "seed": seed, "verbosity": 0}
    if n_classes == 2:
        p.update(objective="binary:logistic", eval_metric="logloss")
    else:
        p.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
    return p


def dmatrix(X, y=None, margin: np.ndarray | None = None) -> xgb.DMatrix:
    d = xgb.DMatrix(X, label=y, enable_categorical=True)
    if margin is not None:
        # xgboost 2.0.3 stores the margin flat in row-major order: (n, K) -> n*K
        d.set_base_margin(np.asarray(margin, dtype=float).reshape(-1))
    return d


def train(params: dict, X_tr, y_tr, X_va, y_va, margin_tr=None, margin_va=None,
          num_boost_round: int | None = None, early_stopping: bool = True, base_score: float | None = None):
    """Train one booster. Returns (booster, evals_result).
    Early stopping uses the validation set only; num_boost_round defaults to config max_rounds."""
    cfg = load_config()["xgb"]
    p = dict(params)
    if base_score is not None:
        p["base_score"] = base_score
    d_tr, d_va = dmatrix(X_tr, y_tr, margin_tr), dmatrix(X_va, y_va, margin_va)
    evals_result = {}
    booster = xgb.train(p, d_tr, num_boost_round=num_boost_round or cfg["max_rounds"],
                        evals=[(d_tr, "train"), (d_va, "val")], evals_result=evals_result,
                        early_stopping_rounds=cfg["early_stopping_rounds"] if early_stopping else None,
                        verbose_eval=False)
    return booster, evals_result


def predict_proba(booster: xgb.Booster, X, margin=None, n_classes: int = 2) -> np.ndarray:
    """Probabilities (n, K); the margin must be the same transformer margin used in training."""
    it = (0, booster.best_iteration + 1) if hasattr(booster, "best_iteration") else (0, 0)
    p = booster.predict(dmatrix(X, margin=margin), iteration_range=it)
    return np.column_stack([1 - p, p]) if n_classes == 2 else p.reshape(len(p), n_classes)


def tree_margin(booster: xgb.Booster, X, n_classes: int) -> np.ndarray:
    """Sum of tree outputs only (no base_score, no transformer): predict with a zero base_margin."""
    it = (0, booster.best_iteration + 1) if hasattr(booster, "best_iteration") else (0, 0)
    zero = np.zeros(len(X) if n_classes == 2 else (len(X), n_classes))
    m = booster.predict(dmatrix(X, margin=zero), output_margin=True, iteration_range=it)
    return m if n_classes == 2 else m.reshape(len(X), n_classes)


def fit_s_binary(z1: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None) -> Fusion:
    """Platt scaling: (s, b) = argmin mean BCE(y, sigmoid(b + s*z1 + offset)), s >= 0 (L-BFGS-B)."""
    off = np.zeros_like(z1) if offset is None else offset

    def nll(theta):
        s, b = theta
        m = b + s * z1 + off
        q = expit(m)
        # mean BCE written with logaddexp for stability; gradient w.r.t. (s, b)
        loss = np.mean(np.logaddexp(0, m) - y * m)
        r = q - y
        return loss, np.array([np.mean(r * z1), np.mean(r)])

    res = minimize(nll, x0=np.array([1.0, b0(y)]), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None), (None, None)])
    return Fusion(float(res.x[0]), float(res.x[1]))


def fit_s_multiclass(z: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None) -> Fusion:
    """Temperature scaling: s = argmin mean CE(y, softmax(s*z + offset)), searched on log s."""
    lo, hi = np.log(load_config()["hpo"]["s_range"])
    off = np.zeros_like(z) if offset is None else offset
    rows = np.arange(len(y))

    def nll(log_s):
        return -np.mean(log_softmax(np.exp(log_s) * z + off, axis=1)[rows, y])

    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    return Fusion(float(np.exp(res.x)))


def fit_s(z: np.ndarray, y: np.ndarray, offset=None) -> Fusion:
    """s-MLE for either task; z is z1 (n,) for binary, z (n, K) for multiclass."""
    return fit_s_binary(z, y, offset) if z.ndim == 1 else fit_s_multiclass(z, y, offset)


def transformer_proba(fusion: Fusion, z: np.ndarray) -> np.ndarray:
    """Probabilities of the scaled transformer alone (no trees), (n, K)."""
    m = fusion.margin(z)
    return np.column_stack([1 - expit(m), expit(m)]) if z.ndim == 1 else softmax(m, axis=1)


# ---------------------------------------------------------------------------------------------
# Method runners: one Task (a split + its cached TabPFN probabilities) -> Result per method.
# ---------------------------------------------------------------------------------------------
@dataclass
class Task:
    X_tr: object
    y_tr: np.ndarray
    X_va: object
    y_va: np.ndarray
    X_te: object
    y_te: np.ndarray
    n_classes: int
    seed: int
    p_tr: np.ndarray      # TabPFN probabilities on training rows (out-of-fold by default)
    p_va: np.ndarray
    p_te: np.ndarray

    def z(self, which: str) -> np.ndarray:
        """Transformer scores: z1 (n,) for binary, centred log-probs (n, K) for multiclass."""
        p = getattr(self, f"p_{which}")
        return binary_logit(p) if self.n_classes == 2 else logits(p)


@dataclass
class Result:
    method: str
    p_val: np.ndarray
    p_test: np.ndarray
    seconds: float
    info: dict = field(default_factory=dict)
    evals_result: dict = field(default_factory=dict)


@dataclass
class TreeFit:
    params: dict
    booster: xgb.Booster
    evals_result: dict
    val_auc: float
    seconds: float = 0.0


def fit_trees(task: Task, params: dict, fusion: Fusion | None, rounds=None, early=True,
              base_score=None) -> TreeFit:
    """Train with fixed hyperparameters; fusion=None means plain XGBoost (no margin)."""
    t0 = time.perf_counter()
    m_tr = m_va = None
    if fusion is not None:
        m_tr, m_va = fusion.margin(task.z("tr")), fusion.margin(task.z("va"))
    p = {**base_params(task.n_classes, task.seed), **params}
    booster, ev = train(p, task.X_tr, task.y_tr, task.X_va, task.y_va, m_tr, m_va,
                        num_boost_round=rounds, early_stopping=early, base_score=base_score)
    val = auc(task.y_va, predict_proba(booster, task.X_va, m_va, task.n_classes))
    return TreeFit(params, booster, ev, val, time.perf_counter() - t0)


def tune_trees(task: Task, fusion: Fusion | None, n_trials: int, kind="main", rounds=None, early=True,
               base_score=None):
    """Optuna over tree hyperparameters with the transformer margin held fixed.
    Returns (best TreeFit, history [(val_auc, TreeFit)])."""
    def run(trial):
        tf = fit_trees(task, hpo.xgb_space(trial, kind), fusion, rounds, early, base_score)
        return tf.val_auc, tf
    _, best, history = hpo.search(run, n_trials, task.seed)
    return best, history


def _result(name, task, tf: TreeFit, fusion, t0, **info) -> Result:
    m_va = m_te = None
    if fusion is not None:
        m_va, m_te = fusion.margin(task.z("va")), fusion.margin(task.z("te"))
    return Result(name,
                  predict_proba(tf.booster, task.X_va, m_va, task.n_classes),
                  predict_proba(tf.booster, task.X_te, m_te, task.n_classes),
                  time.perf_counter() - t0,
                  {"params": tf.params, "best_iteration": getattr(tf.booster, "best_iteration", None),
                   **({"s": fusion.s, "b": fusion.b} if fusion is not None else {}), **info},
                  tf.evals_result)


def pfnb_mle(task: Task, name="PFNB-MLE", rounds=None, early=True) -> Result:
    """s-MLE (4.4): fit (s, b) on the out-of-fold training scores, then tune trees with them fixed."""
    t0 = time.perf_counter()
    fusion = fit_s(task.z("tr"), task.y_tr)
    tf, _ = tune_trees(task, fusion, load_config()["hpo"]["trials_pfnb"], rounds=rounds, early=early)
    return _result(name, task, tf, fusion, t0)


def pfnb_joint(task: Task, mle: Result) -> Result:
    """s-joint (4.4): from PFNB-MLE's trees, refit (s, b) on validation with the tree margin fixed,
    retrain the trees with the same hyperparameters; repeat `s_joint.rounds` times."""
    t0 = time.perf_counter()
    params = mle.info["params"]
    fusion = Fusion(mle.info["s"], mle.info["b"])
    tf = fit_trees(task, params, fusion)
    s_log = [fusion.s]
    for _ in range(load_config()["s_joint"]["rounds"]):
        offset = tree_margin(tf.booster, task.X_va, task.n_classes)
        fusion = fit_s(task.z("va"), task.y_va, offset)
        tf = fit_trees(task, params, fusion)
        s_log.append(fusion.s)
    return _result("PFNB-Joint", task, tf, fusion, t0 - mle.seconds, s_history=s_log)


def pfnb_optuna(task: Task, xgb_history: list, name="PFNB-Optuna", rounds=None, early=True) -> Result:
    """Paper replica (4.4): stage 1 = trees tuned with s = 0 (reused from the XGB study's first
    `trials_pfnb` trials), stage 2 = Optuna over s only (`trials_s`), b fixed at logit(mean y)."""
    h = load_config()["hpo"]
    t0 = time.perf_counter()
    stage1 = max(xgb_history[:h["trials_pfnb"]], key=lambda t: t[0])[1]
    b = b0(task.y_tr) if task.n_classes == 2 else 0.0

    def run(trial):
        fusion = Fusion(hpo.s_space(trial), b)
        tf = fit_trees(task, stage1.params, fusion, rounds, early)
        return tf.val_auc, (tf, fusion)
    _, (tf, fusion), _ = hpo.search(run, h["trials_s"], task.seed)
    stage1_seconds = sum(t[1].seconds for t in xgb_history[:h["trials_pfnb"]])
    return _result(name, task, tf, fusion, t0 - stage1_seconds, stage1_trials=h["trials_pfnb"])
