"""Tables -> CSV / Markdown / LaTeX (booktabs) in results/tables/ (CLAUDE.md 7)."""
from pathlib import Path

import pandas as pd

from .config import path


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


def write_table(df: pd.DataFrame, stem: str, float_fmt: str = "{:.4f}") -> dict:
    """Write `df` as <stem>.csv, <stem>.md and <stem>.tex. Returns the three paths."""
    d = tables_dir()
    out = {ext: d / f"{stem}.{ext}" for ext in ("csv", "md", "tex")}
    df.to_csv(out["csv"], index=False)
    shown = df.apply(lambda c: c.map(float_fmt.format) if c.dtype.kind == "f" else c)
    out["md"].write_text(_markdown(shown))
    # pandas' to_latex uses booktabs rules (\toprule/\midrule/\bottomrule) by default
    out["tex"].write_text(shown.to_latex(index=False, escape=True))
    return out
