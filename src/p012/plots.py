"""All figures, one function per figure ID (CLAUDE.md 7). PNG at 300 DPI plus PDF, white background.
Every function reads its numbers from files under results/ and returns the PNG path."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from .config import load_config, path  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def _save(fig, stem: str) -> Path:
    d = path("results") / "figures"
    d.mkdir(parents=True, exist_ok=True)
    cfg = load_config()["figures"]
    for ext in cfg["formats"]:
        fig.savefig(d / f"{stem}.{ext}", dpi=cfg["dpi"], bbox_inches="tight")
    plt.close(fig)
    return d / f"{stem}.png"


def _preds(run_dir: str, which="test"):
    with np.load(path("results") / run_dir / f"preds_{which}.npz") as d:
        return d["proba"], d["y"]


def _unit(runs, protocol, seed, methods):
    sub = runs[(runs.protocol == protocol) & (runs.seed == seed)].set_index("method")
    return sub.loc[[m for m in methods if m in sub.index]]


def f02_bc_roc(runs: pd.DataFrame, methods: list) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m, r in _unit(runs, "holdoutA", 0, methods).iterrows():
        p, y = _preds(r.run_dir)
        fpr, tpr, _ = roc_curve(y, p[:, 1])
        ax.plot(fpr, tpr, label=f"{m} (AUC {r.auc:.3f})", lw=1.4)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate (malignant)",
           title="ROC curves, BreastCancer hold-out A test set, seed 0")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, "F02_bc_roc")


def f03_bc_confusion(runs: pd.DataFrame, methods: list, labels: list) -> Path:
    """Large panel: method with the best mean validation AUC on hold-out A; small grid: every method (seed 0)."""
    a = runs[runs.protocol == "holdoutA"]
    best = a[a.method.isin(methods)].groupby("method").val_auc.mean().idxmax()
    unit = _unit(runs, "holdoutA", 0, methods)
    fig = plt.figure(figsize=(12, 5))
    ncols = (len(unit) + 1) // 2
    grid = fig.add_gridspec(2, 1 + ncols, width_ratios=[2] + [1] * ncols)

    def draw(ax, m, big):
        p, y = _preds(unit.loc[m].run_dir)
        cm = confusion_matrix(y, p.argmax(1), labels=range(len(labels)))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=12 if big else 9,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(f"{m}" + (" (best validation AUC)" if big else ""), fontsize=11 if big else 9)
        if big:
            ax.set(xlabel="Predicted class", ylabel="True class")

    draw(fig.add_subplot(grid[:, 0]), best, True)
    for k, m in enumerate(unit.index):
        draw(fig.add_subplot(grid[k // ncols, 1 + k % ncols]), m, False)
    fig.suptitle("Confusion matrices, BreastCancer hold-out A test set (n = 114), seed 0, threshold 0.5")
    return _save(fig, "F03_bc_confusion")


def f04_bc_training_curves(curves_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, color in (("XGB", "tab:blue"), ("PFNB-MLE", "tab:red")):
        with np.load(curves_dir / f"{name}.npz") as c:
            for which, ls in (("train", "--"), ("val", "-")):
                axes[0].plot(c["round"], c[f"{which}_logloss"], ls, color=color, label=f"{name} {which}")
                axes[1].plot(c["round"], c[f"{which}_accuracy"], ls, color=color, label=f"{name} {which}")
            for ax in axes:
                ax.axvline(int(c["best_iteration"]) + 1, color=color, lw=0.8, ls=":")
    axes[0].set(xlabel="Boosting round", ylabel="Log-loss", title="Log-loss per round")
    axes[1].set(xlabel="Boosting round", ylabel="Accuracy", title="Accuracy per round (threshold 0.5)")
    for ax in axes:
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("Training curves, BreastCancer hold-out A seed 0 (dotted: early-stopping round)")
    return _save(fig, "F04_bc_training_curves")


def f05_bc_model_comparison(table: pd.DataFrame) -> Path:
    """table: T2 numeric (method, <metric>_mean, <metric>_std)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(table))
    for ax, m, label in ((axes[0], "auc", "ROC-AUC"), (axes[1], "f1", "F1 (malignant)")):
        ax.bar(x, table[f"{m}_mean"], yerr=table[f"{m}_std"], capsize=3, color="tab:gray")
        ax.set_xticks(x, table.method, rotation=35, ha="right")
        lo = (table[f"{m}_mean"] - table[f"{m}_std"]).min()
        ax.set_ylim(max(0, lo - 0.03), 1.0)
        ax.set(ylabel=f"Test {label}", title=f"{label}, mean ± std over 5 seeds")
    fig.suptitle("Model comparison, BreastCancer hold-out A")
    return _save(fig, "F05_bc_model_comparison")


def f06_bc_cv_boxplot(runs: pd.DataFrame, methods: list) -> Path:
    cv = runs[runs.protocol == "cv5"]
    ms = [m for m in methods if m in set(cv.method)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.boxplot([cv[cv.method == m].sort_values("seed").auc for m in ms], labels=ms)
    for k, m in enumerate(ms):
        v = cv[cv.method == m].auc
        ax.scatter(np.full(len(v), k + 1), v, s=12, color="k", zorder=3)
    ax.set(ylabel="Held-out fold ROC-AUC", title="BreastCancer 5-fold CV: AUC per fold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, "F06_bc_cv_boxplot")


def f07_bc_s_sweep(sweep_file: Path, runs: pd.DataFrame) -> Path:
    with np.load(sweep_file) as d:
        s, val, test = d["s"], d["val_auc"], d["test_auc"]
    unit = _unit(runs, "holdoutA", 0, ["TabPFN", "PFNB-MLE", "PFNB-Optuna"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pos = s > 0
    ax.plot(s[pos], val[pos], "-o", ms=3, color="tab:blue", label="validation AUC")
    ax.plot(s[pos], test[pos], "-o", ms=3, color="tab:orange", label="test AUC")
    ax.axhline(val[~pos][0], color="tab:blue", ls=":", lw=1, label="s = 0 (plain XGBoost), val")
    ax.axhline(test[~pos][0], color="tab:orange", ls=":", lw=1, label="s = 0 (plain XGBoost), test")
    ax.axhline(unit.loc["TabPFN"].auc, color="k", ls="--", lw=1, label="s → ∞ (TabPFN alone), test")
    for m, color in (("PFNB-MLE", "tab:red"), ("PFNB-Optuna", "tab:green")):
        sm, label = unit.loc[m].s, m.replace("PFNB-", "s-")
        if sm > 0:
            ax.axvline(min(sm, s.max()), color=color, lw=1.2,
                       label=f"{label}: s = {sm:.3g}" + (" (off scale)" if sm > s.max() else ""))
        else:   # s = 0 has no place on a log axis: legend entry only
            ax.plot([], [], color=color, lw=1.2, label=f"{label}: s = 0 (same as XGBoost)")
    ax.set_xscale("log")
    ax.set(xlabel="Transformer scale s (log scale)", ylabel="ROC-AUC",
           title="Effect of s, BreastCancer hold-out A seed 0 (trees = PFNB-MLE parameters)")
    ax.legend(fontsize=7.5, loc="lower left")
    return _save(fig, "F07_bc_s_sweep")


def f08_bc_calibration(runs: pd.DataFrame, n_bins: int = 10) -> Path:
    """Reliability diagram on P(malignant), test predictions of the 5 hold-out A seeds pooled."""
    a = runs[runs.protocol == "holdoutA"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    edges = np.linspace(0, 1, n_bins + 1)
    for m, label in (("TabPFN", "TabPFN (raw)"), ("TabPFN-sMLE", "TabPFN scaled by s-MLE"),
                     ("PFNB-MLE", "PFNB-MLE")):
        rows = a[a.method == m]
        if rows.empty:
            continue
        p = np.concatenate([_preds(r)[0][:, 1] for r in rows.run_dir])
        y = np.concatenate([_preds(r)[1] for r in rows.run_dir])
        b = np.minimum(np.digitize(p, edges) - 1, n_bins - 1)
        keep = [k for k in range(n_bins) if (b == k).any()]
        ax.plot([p[b == k].mean() for k in keep], [y[b == k].mean() for k in keep], "-o", ms=4,
                label=f"{label} (ECE {rows.ece.mean():.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="perfect calibration")
    ax.set(xlabel="Predicted P(malignant)", ylabel="Observed share malignant",
           title=f"Reliability diagram, hold-out A test ({n_bins} bins, 5 seeds pooled)")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, "F08_bc_calibration")


def f14_ablation(table: pd.DataFrame) -> Path:
    """table: T5 numeric (method, auc_mean/std, log_loss_mean/std)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    y = np.arange(len(table))[::-1]
    for ax, m, label in ((axes[0], "auc", "Test ROC-AUC"), (axes[1], "log_loss", "Test log-loss")):
        ax.barh(y, table[f"{m}_mean"], xerr=table[f"{m}_std"], capsize=3, color="tab:gray")
        ax.set_yticks(y, table.method)
        ax.set(xlabel=f"{label} (mean ± std, 5 seeds)", title=label)
        if m == "auc":
            ax.set_xlim(max(0, (table.auc_mean - table.auc_std).min() - 0.02), 1.0)
    fig.suptitle("Ablations, BreastCancer hold-out A")
    return _save(fig, "F14_ablation")
