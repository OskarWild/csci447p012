"""Sweep protocol (CLAUDE.md 5.2): for each dataset, size and seed, n train + n val rows from the pool,
the shipped test set, all main methods; plus the paper-config ablation (XGB-Paper, PFNB-Paper) on the
datasets in datasets.paper_config_ablation. Resumable (finished units are skipped).
Phase 4 pilot:  python scripts/05_sweep.py --datasets BreastCancer diabetes creditg --seeds 0 1
Phase 5:        python scripts/05_sweep.py                      (all sweep datasets, all seeds)"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")   # for joblib workers

from joblib import Parallel, delayed  # noqa: E402

from p012.config import load_config  # noqa: E402
from p012.data import load_dataset, sweep_sizes, sweep_split  # noqa: E402
from p012.runner import run_unit  # noqa: E402

PAPER_ABLATION = ["XGB-Paper", "PFNB-Paper"]
from p012.tabpfn_scores import get_scores  # noqa: E402


def job(name, size, seed):
    t0 = time.time()
    ds = load_dataset(name)
    split = sweep_split(ds, size, seed)
    abl = PAPER_ABLATION if name in load_config()["datasets"]["paper_config_ablation"] else []
    status = run_unit("sweep", ds, split, size, seed, ablations=abl)
    note = f" ({split.note})" if split.note else ""
    return f"{name} {size} seed {seed}: {status} in {time.time() - t0:.0f}s{note}"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=cfg["datasets"]["sweep"])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    todo = [(n, size, seed) for n in args.datasets for size in sweep_sizes(load_dataset(n)) for seed in args.seeds]
    logging.info("%d units", len(todo))
    for n, size, seed in todo:    # TabPFN scores first, one at a time (cached)
        ds = load_dataset(n)
        get_scores(sweep_split(ds, size, seed), ds, "sweep", size, seed, "oof")
    logging.info("TabPFN scores cached")
    n_jobs = max(1, os.cpu_count() // cfg["xgb"]["nthread"])
    for msg in Parallel(n_jobs=n_jobs, return_as="generator")(delayed(job)(*u) for u in todo):
        logging.info(msg)
    logging.info("all units finished")


if __name__ == "__main__":
    main()
