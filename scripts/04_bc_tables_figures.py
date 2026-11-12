"""Phase 3 outputs from results/runs: tables T2-T5 and figures F02-F08, F14 (CLAUDE.md 7)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from p012 import plots  # noqa: E402
from p012.config import load_config, path  # noqa: E402
from p012.data import load_dataset  # noqa: E402
from p012.report import METRICS, load_runs, mean_std, write_table  # noqa: E402
from p012.runner import MAIN  # noqa: E402

# T5 rows: (method, what it changes relative to PFNB-MLE)
ABLATION_ROWS = [
    ("PFNB-MLE", "main method: s-MLE on out-of-fold scores, early stopping"),
    ("PFNB-MLE-insample", "s-MLE on in-sample TabPFN scores (original code's leak)"),
    ("PFNB-Optuna", "s tuned by Optuna (paper) instead of s-MLE"),
    ("PFNB-Joint", "s-MLE, then s refit on validation with trees fixed (2 rounds)"),
    ("PFNB-MLE-20r", "fixed 20 boosting rounds, no early stopping"),
    ("PFNB-Paper", "paper config: paper search space, 20 rounds, s by Optuna"),
    ("XGB-Paper", "plain XGBoost with the paper config"),
    ("TabPFN-sMLE", "TabPFN scaled by s-MLE, no trees"),
    ("XGB", "plain XGBoost (s = 0)"),
    ("TabPFN", "TabPFN alone (s -> infinity)"),
]


def bc_table(runs, protocol, stem):
    sub = runs[(runs.protocol == protocol) & runs.method.isin(MAIN)]
    num, shown = mean_std(sub, "method", METRICS + ["seconds"], MAIN)
    write_table(num, stem, display=shown.rename(columns={"seconds": "fit time (s)"}))
    return num


def cv_table(runs):
    cv = runs[(runs.protocol == "cv5") & runs.method.isin(MAIN)]
    num, shown = mean_std(cv, "method", METRICS + ["seconds"], MAIN)
    per_fold = cv.pivot_table(index="method", columns="seed", values=["accuracy", "auc"]).reindex(num.method)
    per_fold.columns = [f"fold{k + 1}_{m}" for m, k in per_fold.columns]
    per_fold = per_fold.reset_index()
    write_table(per_fold.merge(num, on="method"), "T4_bc_cv5",
                display=per_fold.round(4).astype({c: str for c in per_fold.columns[1:]})
                .merge(shown.rename(columns={"seconds": "fit time (s)"}), on="method"))


def ablation_table(runs):
    a = runs[runs.protocol == "holdoutA"]
    order = [m for m, _ in ABLATION_ROWS]
    num, shown = mean_std(a[a.method.isin(order)], "method", ["auc", "log_loss", "ece", "seconds"], order)
    s = a.groupby("method").s.mean()
    what = dict(ABLATION_ROWS)
    for t in (num, shown):
        t.insert(1, "change", t.method.map(what))
        t["s_mean"] = t.method.map(s)
    shown["s_mean"] = shown.s_mean.map(lambda v: "" if pd.isna(v) else f"{v:.3g}")
    write_table(num, "T5_bc_ablation", display=shown.rename(columns={"seconds": "fit time (s)", "s_mean": "mean s"}))
    return num


def main():
    runs = load_runs()
    ds = load_dataset(load_config()["datasets"]["primary"])
    t2 = bc_table(runs, "holdoutA", "T2_bc_holdoutA")
    bc_table(runs, "holdoutB", "T3_bc_holdoutB")
    cv_table(runs)
    t5 = ablation_table(runs)
    unit0 = path("results") / "runs" / "holdoutA" / ds.name / "pool" / "0"
    for f in (plots.f02_bc_roc(runs, MAIN), plots.f03_bc_confusion(runs, MAIN, ds.classes),
              plots.f04_bc_training_curves(unit0 / "curves"), plots.f05_bc_model_comparison(t2),
              plots.f06_bc_cv_boxplot(runs, MAIN), plots.f07_bc_s_sweep(unit0 / "s_sweep" / "s_sweep.npz", runs),
              plots.f08_bc_calibration(runs), plots.f14_ablation(t5)):
        print(f)


if __name__ == "__main__":
    main()
