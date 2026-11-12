"""Run every method on one split and write results/runs/<protocol>/<dataset>/<size>/<seed>/<method>/
{metrics.json, preds_test.npz, preds_val.npz, config.json, time.json} (CLAUDE.md 7). A unit is skipped when
all of its methods already have metrics.json, so re-running a script resumes where it stopped."""
import json
import logging
import time
from pathlib import Path

import numpy as np

from . import baselines as bl
from . import pfn_boost as pb
from .config import load_config, path
from .data import Dataset, Split, as_xgb_frame
from .metrics import auc, evaluate
from .tabpfn_scores import clip_renorm, get_scores

log = logging.getLogger(__name__)

MAIN = ["XGB", "TabPFN", "Selection", "Stacking", "PFNB-Optuna", "PFNB-MLE", "PFNB-Joint"]
# Ablations (CLAUDE.md 5.1): insample scores, fixed 20 rounds, paper config, and s-MLE without trees.
ABLATIONS = ["PFNB-MLE-insample", "PFNB-MLE-20r", "XGB-Paper", "PFNB-Paper", "TabPFN-sMLE"]


def run_dir(protocol, dataset, size, seed, method) -> Path:
    return path("results") / "runs" / protocol / dataset / str(size) / str(seed) / method


def budgets(protocol: str) -> dict:
    """Optuna trials per search. CV uses `trials_cv` for every tree search, with s-Optuna's stage 1 / stage 2
    split in the same 40:20 ratio as the main budget (logged in docs/DEVIATIONS.md)."""
    h = load_config()["hpo"]
    if protocol != "cv5":
        return {"xgb": h["trials_xgb"], "pfnb": h["trials_pfnb"], "s": h["trials_s"]}
    n = h["trials_cv"]
    stage1 = round(n * h["trials_pfnb"] / h["trials_xgb"])
    return {"xgb": n, "pfnb": n, "s": n - stage1, "stage1": stage1}


def make_task(split: Split, ds: Dataset, scores: dict, seed: int) -> pb.Task:
    def part(which):
        X, y = split.part(which)
        return as_xgb_frame(X.reset_index(drop=True), ds.cat_cols), y
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = part("train"), part("val"), part("test")
    return pb.Task(X_tr, y_tr, X_va, y_va, X_te, y_te, ds.n_classes, seed,
                   scores["p_train"], scores["p_val"], scores["p_test"])


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def save(res: pb.Result, task: pb.Task, split: Split, unit: dict, extra_seconds: float, budget: dict):
    d = run_dir(unit["protocol"], unit["dataset"], unit["size"], unit["seed"], res.method)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "preds_test.npz", proba=res.p_test, y=task.y_te, idx=split.test_idx)
    np.savez(d / "preds_val.npz", proba=res.p_val, y=task.y_va, idx=split.val_idx)
    metrics = {"test": evaluate(task.y_te, res.p_test), "val_auc": auc(task.y_va, res.p_val), "info": res.info}
    (d / "config.json").write_text(json.dumps(_jsonable({**unit, "method": res.method, "budget": budget,
                                                         "n_train": len(task.y_tr), "n_val": len(task.y_va),
                                                         "n_test": len(task.y_te)}), indent=1))
    (d / "time.json").write_text(json.dumps({"method_seconds": res.seconds, "tabpfn_seconds": extra_seconds,
                                             "total_seconds": res.seconds + extra_seconds}, indent=1))
    if res.evals_result:
        (d / "evals_result.json").write_text(json.dumps(_jsonable(res.evals_result)))
    # metrics.json last: its presence marks the run as finished
    (d / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=1))


def unit_done(unit: dict, methods: list) -> bool:
    return all((run_dir(unit["protocol"], unit["dataset"], unit["size"], unit["seed"], m) / "metrics.json").exists()
               for m in methods)


def run_unit(protocol: str, ds: Dataset, split: Split, size: str, seed: int, ablations=False,
             curves: bool = False) -> str:
    """All main methods plus ablations on one split. `ablations` is True (all of ABLATIONS), False, or a list
    of ablation names (the sweep runs only the paper-config pair). If the main methods are already finished,
    only the missing ablations run, so a finished unit is never re-trained. Returns 'skipped' or 'done'."""
    unit = {"protocol": protocol, "dataset": ds.name, "size": size, "seed": seed,
            "stratified": split.stratified, "note": split.note}
    extra = ABLATIONS if ablations is True else list(ablations or [])
    if unit_done(unit, MAIN + extra):
        return "skipped"
    import torch
    torch.set_num_threads(load_config()["xgb"]["nthread"])
    budget = budgets(protocol)
    t_unit = time.perf_counter()

    sc = get_scores(split, ds, protocol, size, seed, "oof")
    task = make_task(split, ds, sc, seed)
    unit["cf_scheme"] = sc["cf_scheme"]
    t_full, t_cf = float(sc["seconds_full"]), float(sc["seconds_crossfit"])

    main_done = unit_done(unit, MAIN)
    if main_done:
        if curves:
            raise ValueError("curves need the main methods' boosters; run the unit from scratch")
        f = run_dir(protocol, ds.name, size, seed, "PFNB-MLE") / "metrics.json"
        mle_fusion = pb.Fusion(**{k: json.loads(f.read_text())["info"][k] for k in ("s", "b")})
        results = []
    else:
        xgb_res, hist = bl.xgb_baseline(task, n_trials=budget["xgb"])
        tab = bl.tabpfn_baseline(task)
        mle = pb.pfnb_mle(task, n_trials=budget["pfnb"])
        mle_fusion = pb.Fusion(mle.info["s"], mle.info["b"])
        results = [
            (xgb_res, 0.0), (tab, t_full), (bl.selection(xgb_res, tab, task.y_va), t_full),
            (bl.stacking(task, n_trials=budget["xgb"]), t_full + t_cf),
            (pb.pfnb_optuna(task, hist, n_stage1=budget.get("stage1"), n_s=budget["s"]), t_full),
            (mle, t_full + t_cf), (pb.pfnb_joint(task, mle), t_full + t_cf),
        ]
    if "PFNB-MLE-insample" in extra:
        sc_in = get_scores(split, ds, protocol, size, seed, "insample")
        task_in = make_task(split, ds, sc_in, seed)
        results.append((pb.pfnb_mle(task_in, "PFNB-MLE-insample", n_trials=budget["pfnb"]),
                        float(sc_in["seconds_full"])))
    if "PFNB-MLE-20r" in extra:
        results.append((pb.pfnb_mle(task, "PFNB-MLE-20r", rounds=20, early=False, n_trials=budget["pfnb"]),
                        t_full + t_cf))
    if "XGB-Paper" in extra or "PFNB-Paper" in extra:
        xgb_paper, hist_paper = bl.xgb_baseline(task, kind="paper", rounds=20, early=False, name="XGB-Paper",
                                                n_trials=budget["xgb"])
        results += [(xgb_paper, 0.0),
                    (pb.pfnb_optuna(task, hist_paper, "PFNB-Paper", rounds=20, early=False), t_full)]
    if "TabPFN-sMLE" in extra:
        t0 = time.perf_counter()
        scaled = pb.Result("TabPFN-sMLE", pb.transformer_proba(mle_fusion, task.z("va")),
                           pb.transformer_proba(mle_fusion, task.z("te")), time.perf_counter() - t0,
                           {"s": mle_fusion.s, "b": mle_fusion.b})
        results.append((scaled, t_full + t_cf))
    for res, extra_seconds in results:
        save(res, task, split, unit, extra_seconds, budget)
    if curves:
        d = run_dir(protocol, ds.name, size, seed, "curves")
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "XGB.npz", **pb.learning_curves(xgb_res.booster, task, None, xgb_res.info["base_score"]))
        np.savez(d / "PFNB-MLE.npz", **pb.learning_curves(mle.booster, task, mle_fusion))
        s_sweep(task, mle, d.parent / "s_sweep")
    if not main_done:
        # wall-clock of the main methods (+ ablations, + TabPFN scoring if not cached), for the estimate (5.3)
        (run_dir(protocol, ds.name, size, seed, "x").parent / "unit_time.json").write_text(json.dumps(
            {"wall_seconds": time.perf_counter() - t_unit, "n_train": len(task.y_tr), "n_val": len(task.y_va),
             "n_test": len(task.y_te), "n_features": task.X_tr.shape[1], "n_classes": task.n_classes}))
    return "done"


def tabpfn_oof_summary(protocol: str, ds: Dataset, split: Split, size: str, seed: int) -> Path:
    """TabPFN's out-of-fold metrics on the training rows (meta-features, CLAUDE.md 4.9), read from the score
    cache. results/cache/ is not in git, so this small file carries them from the machine that ran the unit."""
    f = run_dir(protocol, ds.name, size, seed, "x").parent / "tabpfn_oof.json"
    if not f.exists():
        sc = get_scores(split, ds, protocol, size, seed, "oof")
        m = evaluate(split.part("train")[1], clip_renorm(sc["p_train"]))
        f.write_text(json.dumps({"cf_scheme": str(sc["cf_scheme"]), "log_loss": m["log_loss"], "ece": m["ece"],
                                 "auc": m["auc"]}))
    return f


def s_sweep(task: pb.Task, mle: pb.Result, out: Path):
    """CLAUDE.md 5.1 s-sweep: PFNB-MLE's tree parameters, retrained for each s on a log grid plus s = 0.
    b is fixed at logit(mean y_train), as in s-Optuna, so the curve isolates s."""
    cfg = load_config()["s_sweep"]
    grid = np.r_[0.0, np.logspace(*np.log10(cfg["range"]), cfg["n_points"])]
    b = pb.b0(task.y_tr) if task.n_classes == 2 else 0.0
    val, test = [], []
    for s in grid:
        fusion = pb.Fusion(float(s), b)
        tf = pb.fit_trees(task, mle.info["params"], fusion)
        val.append(tf.val_auc)
        test.append(auc(task.y_te, pb.predict_proba(tf.booster, task.X_te, fusion.margin(task.z("te")),
                                                    task.n_classes)))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "s_sweep.npz", s=grid, val_auc=np.array(val), test_auc=np.array(test), b=b)
