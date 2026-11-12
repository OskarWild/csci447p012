"""Phase 1: table T1_datasets (classes, pool, test, numeric/categorical features, minority share, paper full counts)
and a copy of the BreastCancer CSVs in data/BreastCancer/ for the Moodle upload."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from p012.config import ROOT, load_config, path  # noqa: E402
from p012.data import load_dataset, paper_full_counts  # noqa: E402
from p012.report import write_table  # noqa: E402


def main():
    cfg = load_config()
    rows = []
    for name in cfg["datasets"]["sweep"]:
        ds = load_dataset(name)
        counts = paper_full_counts(name)
        share = np.bincount(ds.y_pool, minlength=ds.n_classes) / len(ds.y_pool)
        rows.append({
            "dataset": name, "classes": ds.n_classes, "pool": len(ds.y_pool), "test": len(ds.y_test),
            "n_features": ds.X_pool.shape[1], "n_num": ds.X_pool.shape[1] - len(ds.cat_cols),
            "n_cat": len(ds.cat_cols), "minority_share": float(share.min()),
            "positive_class": "" if ds.positive_label is None else str(ds.positive_label),
            "paper_full_train": counts[0] if counts else None, "paper_full_val": counts[1] if counts else None,
        })
    df = pd.DataFrame(rows)
    df[["paper_full_train", "paper_full_val"]] = df[["paper_full_train", "paper_full_val"]].astype("Int64")
    for ext, f in write_table(df, "T1_datasets").items():
        print(f)
    print(df.to_string(index=False))

    src = path("data_root") / cfg["datasets"]["primary"] / cfg["paths"]["data_subdir"]
    dst = ROOT / "data" / cfg["datasets"]["primary"]
    dst.mkdir(parents=True, exist_ok=True)
    for part in ("train.csv", "test.csv"):
        shutil.copy2(src / part, dst / part)
    print(f"copied {src} -> {dst}")


if __name__ == "__main__":
    main()
