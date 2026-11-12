# PC_RUN.md — the one job for Claude on the lab PC (Phase 5 sweep)

You are Claude on the lab PC (Ubuntu). The human has started you and **left**. Nobody will answer questions until the job is done, so do not stop to ask. Do the steps below in order. Then write the report (step 6) and stop. Work only on the `main` branch.

## Context: what is already done

- Phases 0–4 of `CLAUDE.md` are **done on the MacBook**. See `docs/PROGRESS.md`. **Do not redo them.** Do not start Phase 6 or later.
- The code is finished and tested (`pytest -q`: 40 passed on the Mac). The only job here is to **run** it. Do not change files in `src/`, `scripts/`, `configs/` or `tests/`. If something fails, record it and report it. Do not fix the science.
- `CLAUDE.md` is still the rulebook: hard rules in section 1 and the git rules. But it mentions a conda env `p012`. **Here the env is the project venv `.venv/`** (created with `uv`, Python 3.10, CPU PyTorch 2.1.2, same versions as the Mac). Run every Python command as `.venv/bin/python ...`. Do not install, upgrade or remove any package, and do not touch system Python.
- The TabPFN checkpoint is already in `.venv/lib/python3.10/site-packages/tabpfn/models_diff/`.

## Steps

1. **Check the env.**
   ```bash
   .venv/bin/python scripts/00_check_env.py
   .venv/bin/python -m pytest -q
   ```
   Both must succeed: the first prints `OK` and a BreastCancer TabPFN AUC of about 0.993, and the second shows 40 passed. If either fails, do not run the sweep. Go to step 6 and report the error output.

2. **Stay on `main`.** All work goes to `main`, with no branches. Check with `git branch --show-current`.

3. **Start the sweep in the background.** It survives the session ending, and it is resumable: re-running it skips finished units.
   ```bash
   mkdir -p results/logs
   PYTHONWARNINGS=ignore nohup .venv/bin/python scripts/05_sweep.py > results/logs/sweep.log 2>&1 &
   ```
   - It runs all 16 datasets × sizes × 5 seeds. The 34 pilot units from the Mac are skipped, since their results are in git.
   - It first computes the TabPFN scores one unit at a time (`TabPFN scores cached` appears in the log), then runs the units in parallel.
   - The Mac estimate is 7.2 h with 2 workers. This PC uses `cpu_count // 4` workers, so it should be faster.

4. **Wait for it.** Check `results/logs/sweep.log` about every 20–30 minutes, not more often. It is finished when the log ends with `all units finished`.
   - If the process dies (no python process and no final line), look at the last 50 log lines. If the cause is a one-off, such as running out of memory, restart the same command once; finished units are skipped. If it dies again, go to step 6.
   - Check memory with `free -g` once the parallel part starts. If memory is nearly full, tell the human in the report; do not change the number of workers.

5. **Summarise, commit and push.**
   ```bash
   PYTHONWARNINGS=ignore .venv/bin/python scripts/06_pilot_check.py
   git add results/runs/sweep results/runs/all_runs.csv results/tables/P4_pilot_vs_paper.* results/P4_time_estimate.md results/env
   git commit -m "phase5: full sweep runs (lab PC)" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
   git pull --rebase origin main
   git push origin main
   ```
   - The Mac may have pushed code to `main` while the sweep ran, so `git pull --rebase` first. If the rebase conflicts only in `docs/PROGRESS.md`, keep both sides. For any other conflict, run `git rebase --abort`, do not push, and say so in the report. Never force-push.
   - Do **not** commit `results/cache/`, `results/logs/`, `.venv/` or anything over 50 MB. Check with `git status` before committing.
   - `00_check_env.py` rewrites `results/env/` with this PC's setup. That is wanted: it documents the machine.
   - If the push fails (for example, no credentials), leave the commit in place. Say so in the report.

6. **Report.** Append a short entry to `docs/PROGRESS.md` under `## Phase 5 sweep run (lab PC)`, and commit and push it if the push works. The entry covers:
   - start and end time, and total hours
   - the number of workers (`cpu_count // 4`), the CPU model and the RAM
   - how many units finished, plus any failed units with the error lines from the log
   - whether you had to restart
   - the commit hash and whether the push worked

   Then print the same summary in the terminal and stop.
