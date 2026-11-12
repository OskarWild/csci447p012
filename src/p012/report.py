"""Tables -> CSV / Markdown / LaTeX (booktabs) in results/tables/ (CLAUDE.md 7), and the run collector."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import path

METRICS = ["accuracy", "precision", "recall", "f1", "auc", "log_loss", "brier", "ece"]


def tables_dir() -> Path:
    d = path("results") / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _markdown(df: pd.DataFrame) -> str:
    """Pipe table without the `tabulate` dependency that DataFrame.to_markdown needs."""
    cells = [list(map(str, df.columns))] + [["" if pd.isna(v) else str(v) for v in r] for r in df.itertuples(index=False)]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    lines.insert(1, "|" + "|".join("---" for _ in df.columns) + "|")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, stem: str, float_fmt: str = "{:.4f}", display: pd.DataFrame | None = None) -> dict:
    """Write `df` as <stem>.csv, and `display` (default: `df` with floats formatted) as <stem>.md and <stem>.tex.
    Returns the three paths."""
    d = tables_dir()
    out = {ext: d / f"{stem}.{ext}" for ext in ("csv", "md", "tex")}
    df.to_csv(out["csv"], index=False)
    shown = display if display is not None else \
        df.apply(lambda c: c.map(float_fmt.format) if c.dtype.kind == "f" else c)
    out["md"].write_text(_markdown(shown))
    # pandas' to_latex uses booktabs rules (\toprule/\midrule/\bottomrule) by default
    out["tex"].write_text(shown.to_latex(index=False, escape=True))
    return out


def load_runs() -> pd.DataFrame:
    """One row per finished run (metrics.json present) under results/runs/, also written to all_runs.csv.
    Columns: protocol, dataset, size, seed, method, test metrics, val_auc, s, b, seconds."""
    rows = []
    for f in sorted((path("results") / "runs").glob("*/*/*/*/*/metrics.json")):
        d = f.parent
        cfg, m = json.loads((d / "config.json").read_text()), json.loads(f.read_text())
        t = json.loads((d / "time.json").read_text())
        rows.append({k: cfg[k] for k in ("protocol", "dataset", "size", "seed", "method")}
                    | m["test"] | {"val_auc": m["val_auc"], "s": m["info"].get("s"), "b": m["info"].get("b"),
                                   "picked": m["info"].get("picked"), "seconds": t["total_seconds"],
                                   "cf_scheme": cfg.get("cf_scheme"), "run_dir": str(d.relative_to(path("results")))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(path("results") / "runs" / "all_runs.csv", index=False)
    return df


def mean_std(df: pd.DataFrame, by: str, cols: list, order: list | None = None):
    """(numeric table with <col>_mean / <col>_std, display table with 'mean ± std' strings)."""
    g = df.groupby(by)[cols].agg(["mean", "std"])
    g.columns = [f"{c}_{a}" for c, a in g.columns]
    if order:
        g = g.reindex([o for o in order if o in g.index])
    g = g.reset_index()
    shown = pd.DataFrame({by: g[by]})
    for c in cols:
        fmt = "{:.1f} ± {:.1f}" if c == "seconds" else "{:.4f} ± {:.4f}"
        shown[c] = [fmt.format(a, b if np.isfinite(b) else 0.0) for a, b in zip(g[f"{c}_mean"], g[f"{c}_std"])]
    return g, shown
