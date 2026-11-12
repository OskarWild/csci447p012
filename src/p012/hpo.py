"""Optuna wrappers with fixed budgets. Generic: the caller supplies a function that trains with
given hyperparameters and returns (validation AUC, payload); the best payload is kept."""
from typing import Any, Callable

import optuna

from .config import load_config

optuna.logging.set_verbosity(optuna.logging.WARNING)


def xgb_space(trial: optuna.Trial, kind: str = "main") -> dict:
    """'main' = CLAUDE.md 4.6; 'paper' = the paper's Appendix A search space (4.4 paper-config ablation)."""
    if kind == "main":
        return {
            "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 1e2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1, log=True),
        }
    if kind == "paper":
        return {
            "eta": trial.suggest_float("eta", 1e-5, 1, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1e2, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1e2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True),
        }
    raise ValueError(kind)


def s_space(trial: optuna.Trial) -> float:
    """s for PFNB-Optuna: the config's enqueued values first (0 and 1e6), then log-uniform s_range.
    Enqueued values sit outside the log range, so they are set directly rather than suggested."""
    h = load_config()["hpo"]
    if trial.number < len(h["s_enqueue"]):
        s = float(h["s_enqueue"][trial.number])
    else:
        s = trial.suggest_float("s", *h["s_range"], log=True)
    trial.set_user_attr("s", s)
    return s


def search(train_and_score: Callable[[optuna.Trial], tuple[float, Any]], n_trials: int, seed: int):
    """Run a seeded TPE study maximising validation AUC. Returns (study, best_payload, all_payload_scores)
    where all_payload_scores is [(val_auc, payload)] in trial order."""
    best = {"score": -float("inf"), "payload": None}
    history = []

    def objective(trial):
        score, payload = train_and_score(trial)
        history.append((score, payload))
        if score > best["score"]:
            best.update(score=score, payload=payload)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study, best["payload"], history
