# LAB_SETUP — how to set up the lab PC (for Askar and Sagyzdop)

Claude on the lab PC reads `CLAUDE.md` automatically when you start it in the repo folder. It **cannot** open claude.ai links, including the brief doc, so everything it needs is in this repo. Follow the steps below once.

## 1. What goes where

| Item | GitHub (private repo) | Lab PC only | Why |
|---|---|---|---|
| `CLAUDE.md`, `LAB_SETUP.md`, `README.md`, `docs/*.md`, `docs/paper_reference_numbers.csv`, `configs/`, `env/` | ✓ | | Instructions and context: both of you and Claude need them |
| Mayuka's repo (`external/LLM-Boost`) | ✓ as a **submodule** (a link pinned to commit 96c9828), not copied files | | Includes the prepared datasets (125 MB); MIT licence |
| Paper PDF `docs/paper/2502.02672v2.pdf` | ✓ only if the repo is **private** | ✓ | Claude reads it to verify numbers; Claude can also download it itself |
| Your checkpoint answers + `main.ipynb` from the Mac → `docs/previous_work/` | ✓ | | Context: your Mac run errors and the loading code |
| Working TabPFN checkpoint `.cpkt` from your Mac → `assets/tabpfn_ckpt/` | ✗ (git-ignored) | ✓ | Saves fixing the broken download again |
| Code `src/`, `scripts/`, `tests/`, notebook `CSCI447P012.ipynb` | ✓ | | Produced by Claude, reviewed by you |
| `results/tables/`, `results/figures/`, `results/report_assets/`, `RESULTS_SUMMARY.md`, `results/runs/all_runs.csv` | ✓ | | Everything the report and slides need |
| `results/cache/`, logs, per-run prediction files | ✗ | ✓ | Large and reproducible |
| `data/BreastCancer/` (copy, about 300 KB) | ✓ | | For the Moodle dataset upload |

## 2. One-time setup on your own laptop (10 minutes)

1. Create a **private** GitHub repo, e.g. `csci447p012`. Invite your teammate.
2. Unzip the starter kit (it creates a `csci447p012/` folder), then:
   ```bash
   cd csci447p012
   git init && git add . && git commit -m "starter kit"
   git submodule add https://github.com/MayukaJ/LLM-Boost external/LLM-Boost
   cd external/LLM-Boost && git checkout 96c9828dcbc2d07d727ce5e24bace2ac71ff8d1e && cd ../..
   git add .gitmodules external/LLM-Boost && git commit -m "pin original repo @96c9828"
   git branch -M main && git remote add origin <your repo URL> && git push -u origin main
   ```
3. Optional but useful: download the paper PDF into `docs/paper/2502.02672v2.pdf`, and put your checkpoint answers (as `checkpoint_answers.md`) and `main.ipynb` into `docs/previous_work/`. Commit and push.
4. From your Mac, copy the working TabPFN checkpoint to a USB stick or cloud folder. Find it with:
   ```bash
   python -c "import tabpfn, os; print(os.path.join(os.path.dirname(tabpfn.__file__), 'models_diff'))"
   ```

## 3. On the lab PC

1. **If it runs Windows**, use WSL2 with Ubuntu. CUDA works there, and every command in `CLAUDE.md` assumes Linux. Check the GPU with `nvidia-smi`; it should list the RTX 5080.
2. Install, if missing: git, Miniforge (conda), Node.js and Claude Code. Sign in to Claude Code with your account.
3. GitHub access on a **shared** PC: use a fine-grained personal access token limited to this one repo, and sign out or delete it when the project ends.
4. Clone:
   ```bash
   git clone --recurse-submodules <your repo URL> && cd csci447p012
   mkdir -p assets/tabpfn_ckpt   # put the .cpkt from your Mac here
   ```
5. Start a terminal session that survives disconnects (`tmux new -s p012`), then start `claude` in the repo root.

## 4. First prompt to give Claude on the lab PC

> Read CLAUDE.md fully, then docs/PROJECT_BRIEF.md. Summarise the plan back to me in 10 lines, list anything unclear, then start Phase 0. Stop at the end of each phase and report as section 9 of CLAUDE.md describes.

Later prompts are short, e.g. "Phase 0 looks good — continue with Phase 1."

## 5. Your review points (do not skip these)

| After | Check | Takes |
|---|---|---|
| Phase 0 | GPU option A or B chosen; paper numbers verified against the PDF (any corrections in `DEVIATIONS.md`) | 5 min |
| Phase 2 | `pytest -q` passes; the *s* = 0 and `base_margin` tests exist and pass | 5 min |
| **Phase 3** | `results/tables/T2_bc_holdoutA.md` and figures F02–F08 look sensible; enough for the presentation | 20 min |
| Phase 4 | Our XGB and TabPFN numbers are near the paper's; time estimate fits before the deadline | 10 min |
| Phase 6 | `results/meta_findings.md`: an honest result, even if negative | 15 min |
| Phase 8 | The notebook opens with all outputs; name is `CSCI447P012.ipynb` | 10 min |

## 6. Good to know

- The core work (TabPFN + XGBoost) is mainly **CPU** work. The 5080 speeds up TabPFN if PyTorch with CUDA 12.8 works with TabPFN 0.1.9 (Claude tests this in Phase 0). The GPU matters most for the stretch goal (local LLM).
- Long runs keep going in the background even if the Claude session ends. Claude can pick up later, because runs are resumable.
- Claude writes results, figures, tables and the notebook. **You write the report text yourselves** (Turnitin AI limit < 20%). `docs/PROJECT_BRIEF.md` and `results/RESULTS_SUMMARY.md` are for understanding, not for pasting.
- On Moodle, upload `data/BreastCancer/` (tiny) and give the GitHub link to the original repo for the other datasets.
