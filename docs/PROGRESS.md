# PROGRESS

Append one entry per phase: date, what was done, outputs (paths), failures, reproduce command.

## Phases 0-2 (2026-11-12, MacBook, branch `sagyzdop/phase0-2`)

**Phase 0 - setup and checks.** Submodule at `96c9828`. Paper PDF present. Env `p012` (clone of `llmboost-run`, CPU, Option B; see DEVIATIONS). Table 3 numbers re-verified from the PDF, missing rows added, one cell corrected. `configs/experiment.yaml` + `src/p012/config.py`.
- `python scripts/00_check_env.py` -> versions, GPU none (TabPFN on CPU), **TabPFN smoke AUC on BreastCancer test = 0.9930**, `OK`. Env logged to `results/env/` (`system.txt`, `pip_freeze.txt`, `nvidia_smi.txt`).

**Phase 1 - data.** `src/p012/data.py`: loaders for all 16 datasets (ordinal encoding fit on the pool), hold-out A/B, 5-fold CV, sweep sampling. `scripts/01_make_T1.py` -> `results/tables/T1_datasets.{csv,md,tex}`; BreastCancer CSVs copied to `data/BreastCancer/`. Tests `tests/test_data.py`: shapes for all 16 datasets match CLAUDE.md 3, no train/val/test overlap, same seed same sample, all classes in train.
- Fix during testing: Wine `full` (141 of 142 rows) failed a stratified discard of 1 row -> random discard (DEVIATIONS).

**Phase 2 - methods.** `tabpfn_scores.py` (OOF cross-fitting, LOO fallback, npz cache), `pfn_boost.py` (base_margin PFN-Boost, s-MLE, s-joint, s-Optuna), `baselines.py` (XGB, TabPFN, Selection, Stacking), `hpo.py`, `metrics.py`, `report.py`. Tests in `tests/`: s = 0 equals XGB with base_score = mean(y) (< 1e-5); eta = 1e-8, 1 round gives sigmoid(b + s*z1) / softmax(s*z) on the test DMatrix (binary and 3-class); s-MLE recovers s = 2.5, b = 0.3 (binary) and T = 2.5 (3-class); OOF rows never in context (fake TabPFN that reveals in-context rows), insample mode does the opposite; metrics equal sklearn; ECE = 0 when calibrated; cache round-trip.
- **`pytest -q`: 37 passed.**
- Smoke run of all 7 methods on BreastCancer hold-out A seed 0 with tiny budgets (3-4 trials) completed without errors (numbers not kept; Phase 3 produces the real ones).

Reproduce:
```
conda create -y -n p012 --clone llmboost-run
PIP_KEYRING_PROVIDER=disabled conda run -n p012 pip install --no-input optuna==3.5.0 pyyaml pytest seaborn "matplotlib<3.10" jupyter nbconvert ipykernel numpy==1.26.2 scipy==1.15.2
conda run -n p012 python scripts/00_check_env.py
conda run -n p012 python scripts/01_make_T1.py
conda run -n p012 python -m pytest -q
```

## Phase 3 (2026-11-12, MacBook CPU, branch `sagyzdop/phase3`)

Ran CLAUDE.md 5.1 completely: hold-out A (5 seeds, 7 methods + 5 ablation methods, training curves and s-sweep on seed 0), hold-out B (5 seeds), 5-fold CV (25 trials per search). 130 runs, about 15 minutes wall-clock on 2 parallel workers (`results/logs/bc_course.log`).
- Outputs: `results/tables/T2_bc_holdoutA`, `T3_bc_holdoutB`, `T4_bc_cv5`, `T5_bc_ablation` (.csv/.md/.tex); `results/figures/F02`-`F08`, `F14` (.png 300 dpi + .pdf); `results/RESULTS_SUMMARY.md`; per-run files in `results/runs/` and `results/runs/all_runs.csv`.
- Headline numbers are in `results/RESULTS_SUMMARY.md` (generated, with source paths).
- Surprises, both written up in `docs/QUESTIONS.md`: (1) validation AUC saturates at 1.0 with 91 validation rows, so HPO can pick an overfit configuration (hold-out A seed 0); (2) the paper's enqueued s = 1e6 makes probabilities exactly 0/1, so AUC collapses through ties (PFNB-Optuna seed 3).
- New tests: per-round training curves rebuilt from tree outputs match XGBoost's own eval log (`pytest -q`: 39 passed).

Reproduce:
```
PYTHONWARNINGS=ignore conda run -n p012 python scripts/03_bc_course.py      # resumable
conda run -n p012 python scripts/04_bc_tables_figures.py
conda run -n p012 python scripts/90_make_summary.py
```

## Phase 4 (2026-11-12, MacBook CPU, branch `sagyzdop/phase4`)

Ran the sweep protocol (CLAUDE.md 5.2) on BreastCancer, diabetes and creditg: every size plus `full`, seeds 0 and 1, all 7 main methods. 34 units, written to `results/runs/sweep/` (each unit also writes `unit_time.json` for the estimate).
- Paper check: `scripts/06_pilot_check.py` -> `results/tables/P4_pilot_vs_paper.{csv,md,tex}` and the summary in `results/P4_time_estimate.md`. At full size, TabPFN is within 0.005 of the paper on all three datasets, and XGB/Selection/PFNB-Optuna are within 0.02 on diabetes and creditg. Small sizes (n = 10-50) differ more in both directions; with 2 seeds and 10-50 rows this is mostly seed noise. The 5-seed sweep will show it more clearly.
- Flagged (> 0.02) at full size: BreastCancer XGB, Selection and PFNB-Optuna (about -0.025 to -0.033), and creditg Stacking (+0.042, ours better). The BreastCancer gap has the same cause as Q1: the paper's full size leaves only 45 validation rows, and validation AUC sits at 1.0 for many configurations, so the XGB HPO picks one of them almost at random. PFNB-Optuna's stage 1 equals the XGB search (CLAUDE.md 4.4), so it inherits that choice. TabPFN, which has no HPO, matches the paper. We found no encoding, positive-class or sampling problem, so we keep going (Q1 answer: keep the protocol, report the limitation).
- Time estimate (CLAUDE.md 5.3): the pilot only reaches n_train = 640, so the largest unit was measured separately: `scripts/07_probe_large.py` -> `results/P4_probe_large.json` (jungle full, 28,684 training rows: TabPFN scoring 451 s, about 3.9 s per XGBoost fit). **Estimated full sweep: 7.2 hours on this Mac with 2 workers** (`results/P4_time_estimate.md`), under the 12-hour limit, so no sizes or seeds are cut.
- Failure fixed: the first jungle probe was killed by the OS for running out of memory (exit 137). TabPFN scored all ~38k query rows in one forward pass. `tabpfn_fit_predict` now scores queries in chunks of `tabpfn.predict_chunk` = 2000 rows. In TabPFN 0.1.9, query rows attend only to the context and normalisation uses the context only, so the result is unchanged. A new test confirms chunked = single pass on BreastCancer (`pytest -q`: 40 passed). After the fix the probe peaked at 3.7 GB.

Reproduce:
```
PYTHONWARNINGS=ignore conda run -n p012 python scripts/05_sweep.py --datasets BreastCancer diabetes creditg --seeds 0 1
conda run -n p012 python scripts/07_probe_large.py
conda run -n p012 python scripts/06_pilot_check.py
```

## Phase 5 sweep run (lab PC)

Ran `scripts/05_sweep.py` on the lab PC (Ubuntu, AMD Ryzen 7 9700X, 8 cores / 16 threads, 30 GiB RAM, CPU only: `.venv` has torch 2.1.2+cpu, so the RTX 5080 was not used). **4 workers** (`cpu_count // 4` = 16 // 4). Env check printed `OK` (TabPFN smoke AUC 0.9930) and `pytest -q` passed (40) before the run.

- **Run 1** (2026-11-12 21:47 -> crashed 22:51): `ValueError: Statlog: 446+111 > pool 552`. The paper's Statlog full counts (446 + 111 + 138 test = 695) exceed the 690-row dataset; checked in the PDF (Table 3, every variant). Not noticed until the next morning: the watcher used `pgrep -f 05_sweep.py`, which matched its own command line. **Fix (humans' decision):** when the paper's full counts exceed the pool, use the whole pool at the paper's train/val ratio (Statlog 442 + 110).
- **Run 2** (2026-11-13 12:12 -> crashed 12:13): `ValueError: The number of classes has to be greater than one; got 1 class` at bank n=10. Random (unstratified) samples could miss a class (24 of 485 units), and a leave-one-out cross-fitting fold could hold a single class. **Fixes:** redraw such samples with a derived seed until train and val hold every class (first draw unchanged, so the other 461 units are unaffected); single-class TabPFN contexts return uniform scores. Before restarting, the 24 redrawn units plus Statlog full were run end to end (all passed).
- **Run 3** (2026-11-13 12:27 -> finished 16:57, **4.5 h**): TabPFN scoring 12:27-16:12 (sequential; bank, calhousing and jungle `full` take 6-12 min per seed), tree stage 16:12-16:57. Log ends with `all units finished`. Memory stayed at 7-10 GiB used (19-23 GiB available) throughout.
- **Result: all 485 units finished, every unit dir has all its methods with `metrics.json`, 0 failed units, 0 error lines in the run-3 log.** (455 run in run 3, 30 skipped: 4 Mac pilot cells already done earlier plus the 25 edge-test units.) Crash logs kept locally as `results/logs/sweep_run1_crashed.log`, `sweep_run2_crashed.log` (not in git).
- Code changed (all in `docs/DEVIATIONS.md`): `src/p012/data.py` (Statlog full; redraw), `src/p012/tabpfn_scores.py` (single-class context), `tests/test_data.py` (2 new asserts). `pytest -q`: 40 passed.
- Also decided by the humans today (DEVIATIONS): RQ3 uses only the L2-logistic meta-model; RQ4 (LLM-Boost) dropped; the 60-vs-130-trial HPO budget is a listed limitation; newer TabPFN versions are future work.
- `scripts/06_pilot_check.py` was re-run: `results/tables/P4_pilot_vs_paper.*` now covers all 16 datasets x sizes. Its time estimate in `results/P4_time_estimate.md` ("0.6 hours") is not meaningful any more (it is fitted on finished units with the TabPFN time already cached); the real wall time is 4.5 h (run 3).
- Commit `190c234` ("phase5: full sweep runs (lab PC)"), **pushed** to `origin/main`. `git pull --rebase` refused because of an unrelated unstaged deletion of `PC_SETUP_COMMANDS.md` (present before this session, left untouched); the remote had no new commits, so the push was a clean fast-forward.

Reproduce:
```
.venv/bin/python scripts/00_check_env.py && .venv/bin/python -m pytest -q
PYTHONWARNINGS=ignore nohup .venv/bin/python scripts/05_sweep.py > results/logs/sweep.log 2>&1 &   # resumable
PYTHONWARNINGS=ignore .venv/bin/python scripts/06_pilot_check.py
```
