# CSCI447P012 — Project brief (context for Claude on the lab PC)

This file collects the background behind `CLAUDE.md`, as of 12 Nov 2026. If something here conflicts with `CLAUDE.md`, `CLAUDE.md` wins.

## 1. Assignment

| Field | Value |
|---|---|
| Project ID | CSCI447P012 |
| Title | LLM-Boost and PFN-Boost: Transformers Boost the Performance of Decision Trees |
| Technique | Gradient boosting plus tabular transformers |
| Code | https://github.com/MayukaJ/LLM-Boost |
| Registered students (Moodle) | Askar Matayev, Bakhtiyar Yesbolsyn |
| Limitations listed by the course | The combination weight is tuned, not learned or theoretically motivated. There is no analysis of when the transformer helps vs hurts. The LLM component "needs API calls" (**inaccurate**: the repo runs local Hugging Face models). Not peer reviewed (true as far as we can find; arXiv only). |
| Target | Derive a meta-feature-based predictor of when transformer-plus-tree hybridisation helps; learn the combination weight rather than tuning it; replace the API-dependent LLM with a local model so the pipeline is deployable on private clinical data. "The 'when does it help' theory is the publishable core." |

## 2. The paper in plain English

- **Idea:** the transformer makes the first guess, and the trees fix its mistakes. XGBoost normally starts from one constant prediction. PFN-Boost starts from the transformer's per-row score instead, so the trees learn the residual of the transformer.
- **Eq. 1:** pred(0,i) = pred(1,i) + s · SCORE_Transformer + C. The transformer acts as a frozen "tree 0". *s* = 0 gives plain XGBoost; *s* → ∞ gives plain TabPFN or the LLM.
- **How *s* is chosen:** Optuna. First 100 trials tune the tree hyperparameters with *s* = 0, then 30 trials tune *s*, log-uniform [1e-4, 1e4]. Baselines get 130 trials.
- **Transformers:** TabPFN (PFN-Boost); an LLM, Qwen-2.5-72B-Instruct with 3 shots (LLM-Boost). LLM class scores are the negative mean token cross-entropy of each label, centred.
- **Baselines:** XGBoost, TabPFN or LLM alone, Selection (better on validation), Stacking (transformer scores as extra features), plus TabLLM and AutoGluon in the appendices.
- **Data:** 16 datasets (8 UCI + 8 from TabLLM). Training sizes 10, 25, 50, 100, 200 or 250, 500, and full. Mean AUC ± standard error over 5 seeds.
- **Claims:** PFN-Boost has the best *average* rank and z-score at all sizes except the tiniest. LLM-Boost is best at the tiniest sizes with meaningful headers.
- **Admitted limitations (Sec. 6):** LLM-Boost needs meaningful headers and sometimes prompt engineering; LLMs are slow (up to about 18 h of score precomputation on 4× RTX A4000); C++ changes to GBDT libraries may be needed for speed. That last point is avoidable with XGBoost's `base_margin`.

## 3. What the paper's own Table 3 shows

At full size, PFN-Boost beats the better of XGBoost and TabPFN by more than 0.001 AUC on only **3 of 15** datasets (Credit-g +0.0082, Abalone +0.0028, Heart +0.0017). It is worse by more than 0.001 on **4** (Diabetes −0.0147, Blood −0.0113, Shark Tank −0.0094, Bank −0.0023). The remaining 8 are within ±0.001. On Breast Cancer at 10–100 training rows, PFN-Boost is *below* plain TabPFN every time (at 10 rows: 0.9817 vs 0.9865). At full size it equals TabPFN (0.9931 vs 0.9930, SE 0.0000), which suggests the tuned *s* became huge. The numbers are in `paper_reference_numbers.csv`. They were read from the arXiv HTML page, so **verify them against the PDF**.

## 4. Audit of the original code (commit 96c9828, 10 Mar 2025, MIT licence)

| What | Where |
|---|---|
| Training-time fusion (margin + `scores*scale` before each gradient step) | `XGBscale/utils.py`, `scale_update`, lines 29–67, key line 58 |
| Monkeypatch installing it | `xgb_tabpfn.py` lines 11–12 |
| Prediction-time fusion | `llm_boost_utils.py`, `predict`, lines 165–190, key line 178 |
| Custom softmax objective | `llm_boost_utils.py` line 51 |
| TabPFN scores | `llm_boost_utils.py`, `load_tabpfn_data`, lines 446–448 |
| *s* search | `xgb_tabpfn.py`, `objective2`, lines 128–129 |
| Bare `except: pass` | `xgb_tabpfn.py` line 97 |
| Fixed 20 boosting rounds | `xgb_tabpfn.py` line 120 |

Problems found (checked against the TabPFN 0.1.9 source, `automl/TabPFN` branch `tabpfn_v1`, `predict()` line 293, which returns `y, p.max(axis=-1)`):

1. **Wrong score:** the code uses the 1-D *winning-class probability*, not per-class scores. 0.9 does not say which class.
2. **Shape bug, hidden:** (n,) + (n, 2) raises a NumPy broadcast error, even at *s* = 0. The bare `except` swallows it, so every split is skipped and the result is NaN. Askar's as-is run (on a Mac, Python 3.10.21) crashed earlier, at `llm_boost_utils.py` line 454 (`IndexError` in the stacking loader). That is the same root cause.
3. **In-context scoring:** TabPFN scores the training rows it holds in its context, so the training scores are over-confident and the trees see too-small residuals.
4. Fixed 20 rounds, with eta searched down to 1e-5, so the trees can barely move a strong start.
5. Centring subtracts max/2, not the mean.
6. The LLM path reads (n, K) scores from CSV, so problems 1–2 do not apply there.

Environment lessons from Askar's Mac run: CUDA-only wheels (`nvidia-*`, `triton`, `bitsandbytes`) do not install on macOS. xgboost 3.x removed `xgboost.training._configure_custom_metric`, which the repo imports. Newer torch broke tabpfn 0.1.9 loading (most likely the torch ≥ 2.6 `weights_only=True` default). TabPFN 0.1.9's checkpoint URL returns HTML, which gives `UnpicklingError: invalid load key`; the fix is to put the real checkpoint file in place.

## 5. Dataset facts

- **Primary:** the repo's Breast Cancer Wisconsin (Diagnostic) subset. It has 569 rows = 455 train + 114 test, **9 named features** (`concavity3, concave_points1, smoothness3, radius3, symmetry1, area3, radius1, concave_points3, perimeter3`; suffix 1 = mean, 3 = worst), and label `y_temp` ∈ {B, M}. Overall 357 B / 212 M (≈1.68 : 1). Train 277/178, test 80/34, so the split is unstratified. UCI licence: CC BY 4.0. Collected by Wolberg, Street and Mangasarian (Univ. of Wisconsin), donated to UCI in 1995.
- OpenML `wdbc` (ID 1510) has the same 569 rows but 30 features named V1–V30. It is **not** what the paper used. Use it only as an optional extra.
- **Paper full-size vs repo pool:** for Abalone, Adult, BreastCancer, Churn and HeartDisease, the paper's train+val is exactly 50% of the repo pool (BreastCancer: 181 + 45 = 226 ≈ 455 / 2). For the other datasets it is about 100%. This likely explains why the paper's full Breast Cancer row is 181/45/114. That is unconfirmed.

## 6. Our plan (the target, made concrete)

- **RQ1** Clean PFN-Boost: per-class log-probabilities, cross-fitted training scores, XGBoost `base_margin`. Compare with Table 3.
- **RQ2** s-MLE: Platt or temperature scaling of TabPFN's out-of-fold logits. Interpretation: *s* is an inverse temperature. *s* < 1 means TabPFN is over-confident (shrink it); *s* > 1 means sharpen it. This is the "theoretically motivated" weight. Compare it with Optuna-tuned *s* on AUC and cost.
- **RQ3** Meta-model on dataset meta-features predicting Δ = AUC(PFN-Boost) − max(AUC(XGB), AUC(TabPFN)), evaluated leave-one-dataset-out, with regret compared against always-XGB, always-TabPFN, always-PFN-Boost and Selection. With only about 15 datasets, a null result is possible and is still reportable.
- **RQ4** (stretch) Local small LLM (Qwen2.5-3B-Instruct) as LLM-Boost on Breast Cancer, on the lab RTX 5080.
- Breast Cancer is near the ceiling (TabPFN ≈ 0.993 AUC). We use it to *demonstrate* the pipeline and meet the course requirements; the sweep and meta-study *answer* the question. Diabetes is the clinical case where the choice matters (−0.015 for PFN-Boost in Table 3).

## 7. Course requirements (summary)

The IEEE report must be at least 8 pages with sections Abstract / Introduction / Background and Related Work / Methodology / Implementation and System Design / Evaluation and Results / Discussion / Conclusion and Future Work / References. It needs: a train–test split (70/30 or 80/20) **and** 5- or 10-fold CV; accuracy, precision, recall, F1 and ROC-AUC; at least one comparison table and one performance chart; a confusion matrix; training curves (accuracy and loss); ROC curves; a methodology block diagram; algorithm pseudocode with a step-by-step explanation; equations; an explanation of the CV strategy; a comparison with the state of the art; project timeline and tech-stack tables; figures at 300 DPI or higher with 10 pt Times New Roman captions; IEEE citations. Notebook named `CSCI447P012.ipynb` with markdown per step, commented code and output visualisations. Upload the dataset if under 100 MB. Turnitin: plagiarism < 10%, AI-generated content < 20%. **The students write the report text themselves; Claude produces results, figures, tables and the notebook.** Presentation: Presenter 1 covers sections 1–4, Presenter 2 covers 5–9, graded individually.

## 8. Open questions

- Why 181/45/114 for Breast Cancer full size (see section 5 for the likely answer)?
- Which TabPFN version and score did the authors actually use?
- The TabPFN column in Table 3 has no SE. Was it a single run per size?
- Names: Moodle lists "Askar Matayev, Bakhtiyar Yesbolsyn". Askar refers to his partner as Sagyzdop. Confirm the names on the slides match registration.

## 9. Sources

- Paper: https://arxiv.org/abs/2502.02672 (HTML v2: https://arxiv.org/html/2502.02672v2)
- Code: https://github.com/MayukaJ/LLM-Boost @ 96c9828dcbc2d07d727ce5e24bace2ac71ff8d1e
- TabPFN 0.1.9 source: https://github.com/automl/TabPFN/tree/tabpfn_v1
