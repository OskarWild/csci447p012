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
