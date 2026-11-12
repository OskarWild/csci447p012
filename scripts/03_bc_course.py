"""Phase 3: BreastCancer course protocol (CLAUDE.md 5.1) - hold-out A (+ ablations, curves, s-sweep),
hold-out B and 5-fold CV, all methods. Resumable: finished units are skipped.
Run:  python scripts/03_bc_course.py   (log: results/logs/bc_course.log when started with nohup)"""
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
from p012.data import cv_splits, holdout_a, holdout_b, load_dataset  # noqa: E402
from p012.runner import run_unit  # noqa: E402
from p012.tabpfn_scores import get_scores  # noqa: E402


def units():
    cfg = load_config()
    ds = load_dataset(cfg["datasets"]["primary"])
    out = []
    for seed in cfg["seeds"]:
        out.append(dict(protocol="holdoutA", split=holdout_a(ds, seed), seed=seed, ablations=True, curves=seed == 0))
    for seed in cfg["seeds"]:
        out.append(dict(protocol="holdoutB", split=holdout_b(ds, seed), seed=seed, ablations=False, curves=False))
    for k, split in enumerate(cv_splits(ds)):
        out.append(dict(protocol="cv5", split=split, seed=k, ablations=False, curves=False))
    return ds, out


def job(ds, u):
    t0 = time.time()
    status = run_unit(u["protocol"], ds, u["split"], "pool", u["seed"], u["ablations"], u["curves"])
    return f"{u['protocol']} seed {u['seed']}: {status} in {time.time() - t0:.0f}s"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds, todo = units()
    # TabPFN scores first, one at a time (cached; workers only read them)
    for u in todo:
        get_scores(u["split"], ds, u["protocol"], "pool", u["seed"], "oof")
        if u["ablations"]:
            get_scores(u["split"], ds, u["protocol"], "pool", u["seed"], "insample")
    logging.info("TabPFN scores cached for %d units", len(todo))
    n_jobs = max(1, os.cpu_count() // load_config()["xgb"]["nthread"])
    for msg in Parallel(n_jobs=n_jobs, return_as="generator")(delayed(job)(ds, u) for u in todo):
        logging.info(msg)
    logging.info("all units finished")


if __name__ == "__main__":
    main()
