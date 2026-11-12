"""Phase 4 timing probe for the largest sweep unit (CLAUDE.md 5.3): the pilot only reaches n_train = 640,
so the full-size units of bank, jungle, Adult and calhousing (13k-29k training rows) are measured here
instead of extrapolated. Scores jungle `full` seed 0 with TabPFN (kept in the cache, so the sweep reuses
it) and times a few XGBoost trials of the main search space. Writes results/P4_probe_large.json.
    python scripts/07_probe_large.py"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import optuna  # noqa: E402

from p012 import hpo  # noqa: E402
from p012 import pfn_boost as pb  # noqa: E402
from p012.config import ROOT, path  # noqa: E402
from p012.data import load_dataset, sweep_split  # noqa: E402
from p012.runner import make_task  # noqa: E402
from p012.tabpfn_scores import get_scores  # noqa: E402

DATASET, SIZE, SEED, N_TRIALS = "jungle", "full", 0, 5


def main():
    import torch
    from p012.config import load_config
    torch.set_num_threads(load_config()["xgb"]["nthread"])
    ds = load_dataset(DATASET)
    split = sweep_split(ds, SIZE, SEED)
    sc = get_scores(split, ds, "sweep", SIZE, SEED, "oof")   # seconds are those of the first (uncached) run
    task = make_task(split, ds, sc, SEED)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    trials = []
    for _ in range(N_TRIALS):
        tr = study.ask()
        t0 = time.perf_counter()
        tf = pb.fit_trees(task, hpo.xgb_space(tr), None, base_score=float(task.y_tr.mean()))
        trials.append({"seconds": time.perf_counter() - t0, "rounds": tf.booster.num_boosted_rounds()})
        study.tell(tr, tf.val_auc)
    out = {"dataset": DATASET, "size": SIZE, "seed": SEED,
           "n_train": len(split.train_idx), "n_val": len(split.val_idx), "n_test": len(split.test_idx),
           # rows TabPFN scores: OOF pass over train + one full-fit pass over train, val and test
           "tabpfn_rows_scored": 2 * len(split.train_idx) + len(split.val_idx) + len(split.test_idx),
           "tabpfn_seconds": float(sc["seconds_full"]) + float(sc["seconds_crossfit"]),
           "xgb_trials": trials,
           "xgb_trial_seconds_mean": sum(t["seconds"] for t in trials) / len(trials)}
    f = path("results") / "P4_probe_large.json"
    f.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f.relative_to(ROOT))


if __name__ == "__main__":
    main()
