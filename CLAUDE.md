# CLAUDE.md — CSCI447P012 lab-PC instructions

You are working on a graded university ML project on the lab PC. Read this whole file before doing anything. Then read `docs/PROJECT_BRIEF.md`. Follow the phases in order. **Stop at the end of every phase and report to the humans.**

- **Project ID:** CSCI447P012 (CSCI 447 Machine Learning: Theory and Practice, Nazarbayev University, Fall 2026)
- **Students:** Askar Matayev and his teammate (check the registered names in `docs/PROJECT_BRIEF.md`, section "Open questions")
- **Paper:** Jayawardhana et al., *Transformers Boost the Performance of Decision Trees on Tabular Data across Sample Sizes*, arXiv 2502.02672 v2 (Feb 2025). PDF: `docs/paper/2502.02672v2.pdf`
- **Original code (reference only):** `external/LLM-Boost`, a git submodule pinned to commit `96c9828dcbc2d07d727ce5e24bace2ac71ff8d1e`
- **Deadlines:** presentation around 18–19 Nov 2026 (week 13–15 slot). Final report, `CSCI447P012.ipynb` and dataset go to Moodle by the end of week 15.

---

## 0. What the project is (short version)

The paper's PFN-Boost uses a transformer's prediction (TabPFN) as the **starting margin** of XGBoost, so that the trees learn only the correction. One scalar *s* controls how much the transformer is trusted. The paper tunes *s* by Optuna search, and only reports wins on average.

Our assigned target has three parts:

1. **Learn *s*** in a principled way (maximum likelihood, i.e. temperature scaling of TabPFN's logits) instead of tuning it.
2. **Predict when hybridisation helps:** build a small meta-model that uses dataset meta-features to say whether PFN-Boost will beat the better of XGBoost and TabPFN, and evaluate it **leave-one-dataset-out**.
3. **Local / private deployment:** everything runs locally. A small local LLM for LLM-Boost on Breast Cancer is a **stretch goal only**.

Research questions (use these names everywhere):

- **RQ1** Clean reproduction: does a correct PFN-Boost match the paper's Table 3?
- **RQ2** Learned *s*: does s-MLE match Optuna-tuned *s* at lower cost?
- **RQ3** When does it help: can meta-features predict the gain on unseen datasets?
- **RQ4** (stretch) Local LLM-Boost on Breast Cancer.

---

## 1. Hard rules (never break these)

1. **Never use the test set for any decision.** Not for HPO, not for choosing *s*, not for early stopping, not for model selection, and not as an input to meta-features. Test data is touched only at final evaluation.
2. **Never type a result by hand.** Every number in a table, figure, `RESULTS_SUMMARY.md` or the notebook must come from code that reads files in `results/`. If a run fails, record the failure. Do not fill the gap.
3. **Do not modify or import `external/LLM-Boost`.** It is read-only reference. Its code has known bugs (see section 3) and Unix-only imports (`fcntl`). You may read its data CSVs.
4. **Do not change scientific decisions on your own.** That means the dataset version, metrics, protocol, method definitions or budgets. A *small* necessary change (for example a budget cut to fit time) goes in `docs/DEVIATIONS.md` with the reason. A *big* change goes in `docs/QUESTIONS.md`, and then you stop and wait for a human.
5. **Same budget for every method you compare.** Report wall-clock time per method.
6. **Never delete results.** Each run writes to its own directory. Re-running must skip finished runs (resumable).
7. **Seeds are fixed and logged.** Seeds are 0, 1, 2, 3, 4 unless the config says otherwise.
8. **Keep the system clean.** The lab PC is shared. Use only the project's conda environments. Do not touch system Python, drivers or other users' files.
9. **Git:** commit at the end of each phase (`phaseN: <what>`) and push. Never commit `results/cache/`, model weights, `.cpkt` files, `assets/`, or anything over 50 MB.
10. **Plain-English logs.** After each phase, append to `docs/PROGRESS.md` what was done, what was produced (file paths), what failed, and the exact commands to reproduce it.

---

## 2. Repository layout (create what is missing; do not rename)

```
CLAUDE.md                     # this file
LAB_SETUP.md                  # for humans
README.md
configs/experiment.yaml       # all settings; code reads only from here
docs/
  PROJECT_BRIEF.md            # background, audit of the original code, plan
  paper/2502.02672v2.pdf      # the paper (humans put it here, or you download it)
  paper_reference_numbers.csv # numbers from the paper's Table 3 (verify against the PDF in Phase 0)
  previous_work/              # Askar's checkpoint answers and Mac notebook, if present
  PROGRESS.md  DEVIATIONS.md  QUESTIONS.md
env/requirements-core.txt  env/requirements-llm.txt
external/LLM-Boost/           # submodule, read-only
src/p012/
  __init__.py
  config.py        # load YAML, resolve paths
  data.py          # load the repo's prepared datasets, encoding, sampling protocol
  tabpfn_scores.py # TabPFN fit/predict, cross-fitted scores, disk cache
  pfn_boost.py     # PFN-Boost via XGBoost base_margin; s-MLE; s-joint
  baselines.py     # XGBoost, TabPFN, Selection, Stacking
  hpo.py           # Optuna wrappers with fixed budgets
  metrics.py       # acc, precision, recall, F1, ROC-AUC, log-loss, Brier, ECE
  meta.py          # meta-features, LODO meta-model, regret
  plots.py         # all figures (one function per figure ID)
  report.py        # tables -> CSV/Markdown/LaTeX; RESULTS_SUMMARY.md
scripts/           # numbered entry points, one per phase step (see below)
tests/             # pytest
results/           # outputs (see section 7); cache/ is git-ignored
CSCI447P012.ipynb  # final notebook (Phase 8)
data/BreastCancer/ # copy of the primary dataset for Moodle upload (tiny)
```

---

## 3. Known facts you must respect (from the audit in `docs/PROJECT_BRIEF.md`)

- **Primary dataset = the repo's Breast Cancer (WDBC) with 9 named features.** Files: `external/LLM-Boost/data/BreastCancer/prototypes-naturallanguage-performance-0/train.csv` (455 rows) and `test.csv` (114 rows). Label column `y_temp` takes the values `B` and `M`. **Positive class = `M` (malignant).** The first CSV column is an index; use `index_col=0`. Class counts: train 277 B / 178 M, test 80 B / 34 M. This shipped split is unstratified.
- **All datasets** use the same folder pattern `external/LLM-Boost/data/<Name>/prototypes-naturallanguage-performance-0/{train,test}.csv` with label `y_temp`. Datasets in scope (16):

  | Folder | Classes | Train pool | Test | In paper Table 3 |
  |---|---|---|---|---|
  | Abalone | 3 | 3341 | 836 | yes |
  | Adult | 2 | 39073 | 1000 | yes |
  | BreastCancer | 2 | 455 | 114 | yes (primary) |
  | Churn | 2 | 5634 | 1000 | yes |
  | HeartDisease | 2 | 242 | 61 | yes |
  | Sharktank | 2 | 396 | 99 | yes |
  | Statlog | 2 | 552 | 138 | yes |
  | Wine | 3 | 142 | 36 | yes |
  | bank | 2 | 36168 | 9043 | yes |
  | blood | 2 | 598 | 150 | yes |
  | calhousing | 2 | 16512 | 4128 | yes |
  | car | 4 | 1382 | 346 | no reference row |
  | creditg | 2 | 800 | 200 | yes |
  | diabetes | 2 | 614 | 154 | yes |
  | heart | 2 | 734 | 184 | yes |
  | jungle | 2 | 35855 | 8964 | yes |

  **Out of scope:** `income`, `SoybeanLarge` (18 classes, above TabPFN's 10-class limit), `UEFA` (not in the paper).
- **Paper's "full" size.** For Abalone, Adult, BreastCancer, Churn and HeartDisease, the paper's full-size train+val is exactly **50%** of the repo's training pool. For the other datasets it is about 100%. Implement full size as the paper's train/val counts from `docs/paper_reference_numbers.csv` (`size_label = full`). Also run `full_all` (the whole pool, split 80/20 into train/val) for BreastCancer only. Record this in `docs/DEVIATIONS.md` as "likely explanation of the 181/45 puzzle, unconfirmed by the authors".
- **Bugs in the original PFN-Boost code. Do not copy them.**
  1. It uses `TabPFNClassifier.predict(..., return_winning_probability=True)`, which returns a 1-D *max-class probability*, not per-class scores.
  2. That 1-D array is added to the (n, K) margin, which raises a broadcast error. A bare `except: pass` hides it, so every run yields NaN.
  3. TabPFN scores the same training rows it holds in its context, so the training scores are over-confident.
  4. Boosting is fixed at 20 rounds.
  5. Scores are centred by subtracting max/2, not the mean the paper describes.
- **TabPFN 0.1.9 limits:** up to about 1000 context rows (subsample if more), up to 100 features and up to 10 classes. Use `N_ensemble_configurations=32`, as in the original code.

---

## 4. Method specifications (implement exactly; tests in Phase 2)

Notation: *K* is the number of classes and *p* = TabPFN `predict_proba`, shape (n, K). Clip *p* to [1e-6, 1 − 1e-6] and renormalise.

**4.1 TabPFN scores — always per-class log-probabilities.** `z = log(p)`, centred per row: `z = z - z.mean(axis=1, keepdims=True)`. For binary tasks, the equivalent single logit is `z1 = log p1 − log p0`. **Encode y so that the positive class is 1** (BreastCancer: M = 1, B = 0), and check that TabPFN's `classes_` order matches before taking column 1.

**4.2 Cross-fitted (out-of-fold) training scores — default.** Use stratified K_cf = 5 folds over the *training* rows. For each fold, fit TabPFN on the other folds (subsample to ≤ 1000 rows, seed-dependent) and score the held-out fold. Validation and test rows are scored by one TabPFN fit on all training rows (≤ 1000 subsample). For folds smaller than 2 rows per class at tiny sizes (n = 10), fall back to leave-one-out cross-fitting, and log it. **Ablation mode `insample`:** score the training rows with a TabPFN that holds them in its context, as the original code does.
Cache every score array to `results/cache/tabpfn/<dataset>/<size>/<seed>/<mode>.npz`, with the row indices. Every method reuses the cache.

**4.3 PFN-Boost = XGBoost with `base_margin`.** Do not monkeypatch XGBoost. When `base_margin` is set, XGBoost **ignores `base_score`**, so the intercept must be handled explicitly. This intercept is the paper's constant *C*.
- Binary: `objective="binary:logistic"`, `base_margin = b + s * z1`, shape (n,). For s-Optuna and *s* = 0, `b = b0 = logit(mean(y_train))`. For s-MLE, *b* is fitted together with *s* (4.4).
- **The XGB baseline sets `base_score = mean(y_train)` explicitly**, so that PFN-Boost with *s* = 0 is exactly plain XGB.
- Multiclass: `objective="multi:softprob"`, `num_class=K`, `base_margin = s * z`, shape (n, K). The XGB baseline keeps the default scalar `base_score`, which cancels in the softmax. Check that XGBoost 2.0.3 accepts the 2-D margin; if it wants it flattened, flatten in row-major order and test it.
- Set `base_margin` on **every** DMatrix: train, validation and test. Prediction must include it.
- `eval_metric = logloss` (binary) or `mlogloss`. Early stopping on the validation set, `early_stopping_rounds=50`, `num_boost_round ≤ 2000`. Keep `evals_result` for training curves.

**4.4 Learning *s*.**
- **s-MLE (main method):** fit on the **out-of-fold training scores** only.
  - Binary: Platt scaling, `(s*, b*) = argmin mean BCE(y, sigmoid(b + s · z1_oof))` with *s* ≥ 0. This is a 2-parameter logistic regression: use `scipy.optimize.minimize` (L-BFGS-B) or Newton.
  - Multiclass: temperature only, `s* = argmin_{s ≥ 0} mean CE(y, softmax(s · z_oof))`. Solve it in 1-D on log *s* ∈ [log 1e-4, log 1e4] with `scipy.optimize.minimize_scalar`.
  - It must be deterministic. Then tune the tree hyperparameters (Optuna, budget `hpo.trials_pfnb`) with *s* and *b* fixed.
- **s-joint:** start from s-MLE and tune the trees. Then refit *s* (and *b* for binary) on the **validation** set with the tree margin fixed (same solver), and retrain the trees with the best parameters and the new *s*. Do 2 rounds. Log *s* after each round.
- **s-Optuna (paper replica):** first tune the trees with *s* = 0 (this equals the XGBoost HPO, so reuse it). Then tune *s* only, log-uniform [1e-4, 1e4], with enqueued trials *s* = 0 and *s* = 1e6, budget `hpo.trials_s`. Same early stopping as above.
- **Paper-config ablation:** the paper's search space (eta log [1e-5, 1], λ, α, γ log [1e-8, 1e2], max_depth 3–10, subsample, colsample_bytree and colsample_bylevel [0.5, 1], min_child_weight log [1e-8, 1e5]) and **fixed 20 rounds**, no early stopping, with *s* from Optuna. BreastCancer and 4 other datasets only.

**4.5 Baselines.**
- **XGB:** XGBoost, no margin, Optuna budget `hpo.trials_xgb` (default = `trials_pfnb + trials_s`, for a fair comparison against s-Optuna).
- **TabPFN:** `predict_proba` from one fit on the training rows (≤ 1000). No HPO.
- **Selection:** XGB or TabPFN, whichever has the higher **validation** AUC.
- **Stacking:** XGBoost on `[X, TabPFN out-of-fold probabilities]` for train, and on probabilities from the full fit for val/test. Optuna budget `hpo.trials_xgb`.

**4.6 XGBoost search space (our main config).** eta log [1e-3, 0.3]; max_depth 2–8; min_child_weight log [1e-2, 1e2]; subsample [0.5, 1]; colsample_bytree [0.5, 1]; lambda log [1e-3, 10]; alpha log [1e-3, 10]; gamma log [1e-8, 1]. TPE sampler seeded with the run seed. Objective = validation ROC-AUC.

**4.7 Categoricals.** Ordinal-encode object columns, with the encoder fit on the training pool. Pass them to XGBoost as `category` dtype with `enable_categorical=True`, `tree_method="hist"`. TabPFN gets the ordinal codes, as in the original code.

**4.8 Metrics (`metrics.py`).** Binary: accuracy, precision, recall and F1 for the positive class at threshold 0.5; ROC-AUC; log-loss; Brier; ECE (15 equal-width bins). Multiclass: accuracy, macro precision/recall/F1, ROC-AUC one-vs-rest macro (matching the paper's `roc_auc_score(..., multi_class='ovr')`), log-loss, ECE on the top-class confidence. Positive class per dataset: BreastCancer `M`; for other binary sets use the minority class and log the choice.

**4.9 Meta-study (`meta.py`).**
- **Unit** = (dataset, size_label, seed). **Target** Δ = test AUC(PFNB-MLE) − max(test AUC(XGB), test AUC(TabPFN)). Make the same target for PFNB-Optuna. Label `helps = Δ > 0.001`, and log that threshold choice.
- **Meta-features, from train/val only:** n_train, n_features, log(n_train/n_features), share of categorical features, number of classes, minority-class share, validation AUC of XGB and of TabPFN and their gap, Spearman correlation between XGB and TabPFN validation predictions, TabPFN out-of-fold ECE and log-loss, the learned s-MLE, a decision-stump validation AUC, and a logistic-regression validation AUC.
- **Meta-models:** L2-logistic regression on standardised features, and a depth-2 decision tree. Keep them tiny.
- **Evaluation: leave-one-dataset-out (LODO), strictly.** All rows of the held-out dataset are excluded from meta-training, including feature standardisation.
- **Policies** compared by **regret** = best-in-hindsight test AUC among {XGB, TabPFN, PFNB-MLE} minus the chosen method's test AUC: always PFNB-MLE, always XGB, always TabPFN, Selection-by-validation (XGB vs TabPFN), meta-logistic, and meta-tree. A meta policy picks PFNB-MLE when it predicts `helps`; otherwise it falls back to Selection-by-validation. Report mean regret ± SE over datasets, plus wins/ties/losses.

**4.10 RQ4, local LLM-Boost (stretch; only after Phase 8 is done and committed).** Use `env/requirements-llm.txt` on the GPU. Model `Qwen/Qwen2.5-3B-Instruct` in bf16 (`Qwen2.5-7B-Instruct` only if it fits in 16 GB). Serialise each row as "The <column> is <value>." with 3 shots drawn from the **training** rows; a row is never in its own prompt. Score each class label by mean token log-likelihood, centre per row, and plug the result into 4.3 exactly like TabPFN scores. BreastCancer only (and diabetes if there is time). Cache the scores.

---

## 5. Protocols

**5.1 Course protocol on BreastCancer (required by the course).**
- **Hold-out A (shipped split):** train pool 455, test 114. Inside the pool, a stratified 80/20 inner split gives train and validation for HPO and early stopping. Run 5 seeds (the seed changes the inner split, TabPFN subsampling and Optuna). Report mean ± std.
- **Hold-out B (stratified 80/20 re-split of all 569 rows)**, seeds 0–4. Same inner procedure. This shows the effect of the unstratified shipped split.
- **5-fold stratified CV on the 455-row pool** (seed 0 for the fold split). In each fold: training pool = 4 folds, then the same inner 80/20 split for HPO, then evaluation on the held-out fold. Report per-fold numbers and mean ± std. HPO budget `hpo.trials_cv`.
- **Methods:** XGB, TabPFN, Selection, Stacking, PFNB-Optuna, PFNB-MLE, PFNB-Joint. **Ablations:** PFNB-MLE with insample scores; paper-config; fixed 20 rounds vs early stopping.
- **s-sweep figure data:** on Hold-out A seed 0, take the PFNB-MLE best tree parameters and evaluate validation and test AUC for *s* on a log grid of 41 points from 1e-4 to 1e4, plus *s* = 0. Mark s-MLE and s-Optuna on the plot.

**5.2 Sweep protocol (RQ1–RQ3, mirrors the paper).** For each dataset, take its training pool P and fixed test T. For each size n in `[10, 25, 50, 100, 250, 500]` with 2n ≤ |P|, and for `full` (the paper's counts): for each seed, sample n training rows and n validation rows from P without overlap. Stratify when every class has at least 2 rows in each part; otherwise sample at random and log it. For `full`, use the paper's train/val counts. The test set is always T. Run all methods from 4.4–4.5. The paper-config ablation runs on its subset only.

**5.3 Compute.** TabPFN 0.1.9 on ≤ 1000 rows takes seconds on CPU and may run on the GPU (see 6.2). XGBoost HPO is the main cost. Parallelise over runs with joblib (`n_jobs = physical cores // xgb_nthread`, `xgb_nthread = 4`). Run long jobs in the background (`nohup ... > results/logs/<name>.log 2>&1 &` or tmux) and keep them resumable: a run is done when its `metrics.json` exists. Before the full sweep, estimate the total time from a 1-dataset pilot and write the estimate in `docs/PROGRESS.md`. If it is over 12 hours, reduce sizes or seeds via the config, log the change in `docs/DEVIATIONS.md`, and tell the humans.

---

## 6. Environment

**6.1 Core env (`p012`).** Python 3.10, conda (Miniforge). Install from `env/requirements-core.txt`. Pinned: xgboost 2.0.3, tabpfn 0.1.9, scikit-learn 1.3.2, optuna 3.5.0, numpy 1.26.4 (**numpy < 2**), pandas 2.1.4, scipy, matplotlib, seaborn, pyyaml, joblib, tqdm, pytest, jupyter, nbconvert, ipykernel.

**6.2 PyTorch and the RTX 5080.** The 5080 is a Blackwell GPU (compute capability 12.0). It needs a PyTorch build with CUDA 12.8 or newer (torch ≥ 2.7, wheel index `https://download.pytorch.org/whl/cu128`). The original `torch==2.1.2` cannot use this GPU.
- **Option A (try first):** torch ≥ 2.7 cu128 + tabpfn 0.1.9. torch ≥ 2.6 defaults `torch.load(weights_only=True)`, which breaks TabPFN 0.1.9's checkpoint loading. Set the environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. That is acceptable here only because the checkpoint is the official TabPFN file. Smoke test: fit and predict on BreastCancer with `device="cuda"`, and compare against a CPU run (probabilities must agree to about 1e-4).
- **Option B (fallback):** torch 2.1.2 CPU build, TabPFN on CPU. This is fine: TabPFN is not the bottleneck.
- Record the choice and the smoke-test output in `docs/PROGRESS.md`. XGBoost stays on CPU (`tree_method="hist"`); do not spend time on XGBoost GPU.

**6.3 TabPFN checkpoint.** TabPFN 0.1.9 downloads its weights on first use. The old URL now returns an HTML page, and the result is `_pickle.UnpicklingError: invalid load key '<'`. Fix, in this order:
1. If `assets/tabpfn_ckpt/*.cpkt` exists (copied by Askar from his working Mac install), copy it to `<site-packages>/tabpfn/models_diff/` under the file name the error message shows.
2. Otherwise download `https://github.com/PriorLabs/TabPFN/raw/refs/tags/v1.0.0/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt` and save it under the expected name.
Verify the file is larger than 10 MB and starts with zip or pickle magic bytes, not `<`. Never commit the checkpoint.

**6.4 LLM env (`p012-llm`, stretch only).** torch ≥ 2.7 cu128, transformers, accelerate, sentencepiece. See `env/requirements-llm.txt`.

**6.5 Log the environment** to `results/env/`: `pip freeze`, `nvidia-smi`, CPU model and core count, OS, and the Python version. The tech-stack table (T10) is generated from these files.

---

## 7. Required outputs (the report and slides are built only from these)

Every figure: PNG at **300 DPI** plus a PDF, white background, readable 10 pt+ fonts, a title, axis labels with units, and the matplotlib defaults otherwise. One function per figure in `plots.py`. File names start with the ID. Every table: CSV plus Markdown plus a LaTeX `.tex` (booktabs), in `results/tables/`. Figures go in `results/figures/`.

| ID | File stem | Content | Needed for |
|---|---|---|---|
| T1 | `T1_datasets` | per dataset: classes, pool, test, n features (num/cat), minority share, paper full counts | Method, dataset slide |
| T2 | `T2_bc_holdoutA` | BC hold-out A: every method × acc, prec, rec, F1, AUC, log-loss, Brier, ECE, fit time (mean ± std, 5 seeds) | Required comparison table |
| T3 | `T3_bc_holdoutB` | same for the stratified re-split | Results |
| T4 | `T4_bc_cv5` | BC 5-fold CV: per-fold accuracy and AUC, plus mean ± std for all metrics | Required CV table |
| T5 | `T5_bc_ablation` | insample vs OOF; tuned vs MLE vs joint *s*; 20 rounds vs early stopping; paper-config | Ablation slide |
| T6 | `T6_paper_vs_ours` | dataset × size: paper XGB / TabPFN / Stacking / PFNB vs ours, with the difference | RQ1, SOTA comparison |
| T7 | `T7_sweep_summary` | dataset × method: test AUC mean ± SE at full size | RQ1/RQ2 |
| T8 | `T8_s_cost` | s-MLE vs s-Optuna: AUC difference and wall-clock time per run | RQ2 |
| T9 | `T9_lodo_regret` | policy × mean regret ± SE, wins/ties/losses | RQ3 |
| T10 | `T10_tech_stack` | layer, tool, version (from `results/env/`) | Required stack table |
| T11 | `T11_meta_model` | logistic coefficients, the depth-2 tree as text, and which features matter | RQ3 |
| F01 | `F01_block_diagram` | draft workflow diagram (matplotlib or graphviz if installed); humans redraw it in draw.io | Required block diagram |
| F02 | `F02_bc_roc` | ROC curves, all methods, hold-out A, seed 0 | Required |
| F03 | `F03_bc_confusion` | confusion matrix of the best method by **validation** AUC (labels B/M, counts) + a grid for all methods | Required |
| F04 | `F04_bc_training_curves` | log-loss **and** accuracy per boosting round, train vs val, XGB vs PFNB-MLE | Required training curves |
| F05 | `F05_bc_model_comparison` | AUC and F1 per method with error bars (hold-out A) | Required performance chart |
| F06 | `F06_bc_cv_boxplot` | per-fold AUC per method | CV slide |
| F07 | `F07_bc_s_sweep` | val and test AUC vs *s* (log x-axis), *s* = 0 and *s* → ∞ reference lines, s-MLE and s-Optuna marked | Explains *s* |
| F08 | `F08_bc_calibration` | reliability diagram: TabPFN raw, TabPFN scaled by s-MLE, PFNB-MLE | Calibration point |
| F09 | `F09_gain_vs_size` | Δ vs training size, one small panel per dataset, PFNB-MLE and PFNB-Optuna | RQ1/RQ3 |
| F10 | `F10_gain_full` | diverging bars of Δ at full size per dataset: ours next to the paper's | RQ1 headline |
| F11 | `F11_paper_vs_ours_scatter` | paper AUC vs our AUC for XGB, TabPFN, PFNB, with y = x line | RQ1 |
| F12 | `F12_lodo_regret` | regret per policy with error bars | RQ3 headline |
| F13 | `F13_meta_tree` | the depth-2 meta-tree drawn (sklearn `plot_tree`) | RQ3 |
| F14 | `F14_ablation` | ablation bars (from T5) | Ablation slide |

Also produce:
- `results/RESULTS_SUMMARY.md`, generated by `scripts/90_make_summary.py`. It lists every headline number the slides need (best BC method and its five metrics, CV mean ± std, s-MLE vs s-Optuna, number of datasets where PFN-Boost helps, LODO regret of each policy). **Each number has the path of the file it came from.** Add a "Limitations and caveats" list that is generated from `docs/DEVIATIONS.md`.
- `results/report_assets/`: copies of the final figures and tables with stable names, ready for the IEEE report and the slides.
- Raw per-run outputs: `results/runs/<protocol>/<dataset>/<size>/<seed>/<method>/{metrics.json, preds_test.npz, config.json, time.json}`. Also `results/runs/all_runs.csv`, one row per run.

---

## 8. Phases (do them in order; stop and report after each)

**Phase 0 — Setup and checks.**
1. Confirm the submodule is at `96c9828...`; if missing, run `git submodule update --init`.
2. If `docs/paper/2502.02672v2.pdf` is missing, download it from `https://arxiv.org/pdf/2502.02672v2`.
3. Create the env (6.1–6.3); log it (6.5).
4. Read the paper PDF, Section 3, Section 4 and Appendix A–B Table 3. **Verify every number in `docs/paper_reference_numbers.csv` against the PDF.** Correct any wrong cell and list the corrections in `docs/DEVIATIONS.md`. The CSV holds the full-size rows plus all sizes for BreastCancer, Abalone and bank. **Add the missing rows** (every other dataset × size in Table 3, and car if it appears) from the PDF, with `note = "added from PDF"`. If a PDF table cannot be read reliably, write `docs/QUESTIONS.md` and leave those cells empty rather than guessing.
5. Write `configs/experiment.yaml` (template provided) and `src/p012/config.py`.
**Done when:** `python scripts/00_check_env.py` prints the versions, the GPU status, the TabPFN smoke-test AUC on BreastCancer (test set, CPU), and "OK".

**Phase 1 — Data.** `data.py` with loaders for all 16 datasets, encoding, the sampling protocol (5.1, 5.2) and fold creation. Write T1. Copy BreastCancer CSVs to `data/BreastCancer/`.
**Tests:** shapes match section 3; no overlap between train/val/test indices; the same seed gives the same sample; classes are present.

**Phase 2 — Methods and unit tests.** `tabpfn_scores.py`, `pfn_boost.py`, `baselines.py`, `hpo.py`, `metrics.py`. Required tests (`pytest -q` must pass):
- Binary: PFN-Boost with *s* = 0 and `b = b0` gives predictions identical to the XGB baseline (with `base_score = mean(y_train)`), same parameters and seed (max abs diff < 1e-5).
- With `eta = 1e-8` and 1 round, PFN-Boost predictions on the **test** DMatrix equal `sigmoid(b + s · z1)` (multiclass: `softmax(s · z)`) to 1e-5. This proves that `base_margin` is applied at train and test time.
- s-MLE recovers known values: simulate y from `sigmoid(0.3 + 2.5 · z1)` with n = 5000 and expect *s* ∈ [2.3, 2.7], *b* ∈ [0.2, 0.4]. Run the multiclass analogue with temperature 2.5.
- Out-of-fold scores: no training row is ever scored by a TabPFN that had it in context (check the indices).
- Metrics match sklearn on a toy example. ECE is 0 for a perfectly calibrated toy example.
- The cache round-trip returns identical arrays.

**Phase 3 — BreastCancer course experiments (top priority for the presentation).** Run 5.1 completely. Produce T2, T3, T4, T5, F02–F08 and F14. Write a first `RESULTS_SUMMARY.md`. **Commit and tag `bc-results`.** Report to the humans with the summary table.

**Phase 4 — Pilot sweep and time estimate.** Run 5.2 on BreastCancer, diabetes and creditg, all sizes and 2 seeds. Check that our XGB and TabPFN AUCs are in the same range as the paper's (within about 0.02 at full size). If not, investigate (encoding, positive class, sampling) before scaling up, and write what you find. Estimate the full-sweep time (5.3).

**Phase 5 — Full sweep.** All 16 datasets × sizes × 5 seeds. Paper-config ablation on BreastCancer, diabetes, creditg, Abalone and blood. Produce T6, T7, T8, F09, F10 and F11.

**Phase 6 — Meta-study.** Meta-features, LODO, policies, regret. Produce T9, T11, F12 and F13. Also write a short `results/meta_findings.md`: which meta-features predict a gain, in plain English, with numbers. If no policy beats "always XGB" or Selection, **say so plainly**. That is a valid result.

**Phase 7 — Reporting assets.** F01 draft, T10, the final `RESULTS_SUMMARY.md`, and `results/report_assets/`. Check that every ID in section 7 exists. Print a checklist with ✓/✗ per ID. **Commit and tag `final-results`.**

**Phase 8 — Notebook `CSCI447P012.ipynb`.** Use the exact name. Structure: title and team; overview; environment; data loading and EDA (class balance plot, feature summary); the method explained in markdown with the equations (Eq. 1, the logit form, the s-MLE objective); cross-fitting; a live BreastCancer run in a fast configuration (fewer Optuna trials, under 15 minutes), producing the confusion matrix, ROC, training curves and the s-sweep; then loading the saved sweep and meta results from `results/` to show T6–T9 and F09–F13; then limitations. Every step gets a markdown cell and commented code. Execute it end to end with `jupyter nbconvert --to notebook --execute --inplace CSCI447P012.ipynb` so that all outputs are saved inside it.

**Phase 9 (stretch) — RQ4 local LLM-Boost.** Only after Phase 8 is committed. Same protocol as 5.1, with the method `LLMB-MLE`. Add a row to T2 and a note to the summary.

---

## 9. How to report to the humans (end of each phase)

Write in short, plain English:
- what you did (3–6 bullets)
- output files created (paths)
- key numbers (only from files, with file paths)
- anything that failed or surprised you
- decisions you need from them (also written in `docs/QUESTIONS.md`)
- the exact command to reproduce

Never claim something worked unless a test or an output file shows it.
