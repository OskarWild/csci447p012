# Lab PC setup: commands for you to run (Ubuntu)

Run these yourself, in order, in a terminal. They need no `sudo` except step 1 (and only if git, curl or tmux are missing). Everything is installed in your home folder and in the project folder, so nothing touches other users.

We use **uv** instead of conda. It is a single small program, needs no admin rights, downloads Python 3.10 by itself, and installs packages fast. The install problems on the Mac were Mac-specific (a broken scipy wheel and a password-keyring hang). On Ubuntu, normal pip wheels work.

## 1. Basic tools (skip if `git --version`, `curl --version` and `tmux -V` all work)
```bash
sudo apt update && sudo apt install -y git curl tmux
```

## 2. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
source $HOME/.local/bin/env
```

## 3. Get the code (the phase4 branch has everything)
```bash
git clone --recurse-submodules https://github.com/OskarWild/csci447p012.git
```
```bash
cd csci447p012 && git checkout sagyzdop/phase4
```

## 4. Create the environment (same versions as the Mac)
```bash
uv venv --python 3.10 .venv
```
```bash
uv pip install --python .venv torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```
```bash
uv pip install --python .venv numpy==1.26.2 scipy==1.15.2 pandas==2.1.4 scikit-learn==1.3.2 xgboost==2.0.3 tabpfn==0.1.9 optuna==3.5.0 pyyaml joblib tqdm matplotlib==3.9.4 seaborn==0.13.2 pytest jupyter nbconvert ipykernel
```

## 5. TabPFN model file
Copy `prior_diff_real_checkpoint_n_0_epoch_100.cpkt` (103 MB) from your Mac to the PC by USB stick or cloud. On the Mac it is in `/opt/homebrew/Caskroom/miniforge/base/envs/p012/lib/python3.10/site-packages/tabpfn/models_diff/`. Then, with the file in your PC's `~/Downloads`:
```bash
cp ~/Downloads/prior_diff_real_checkpoint_n_0_epoch_100.cpkt .venv/lib/python3.10/site-packages/tabpfn/models_diff/
```

## 6. Quick check (should end with `OK`, BreastCancer AUC about 0.993)
```bash
.venv/bin/python scripts/00_check_env.py
```

## 7. GitHub login, so Claude can push the results while you are away
Make a **classic** token at github.com → Settings → Developer settings → Personal access tokens → Tokens (classic). Give it the `public_repo` scope and a 7-day expiry. (Fine-grained tokens do not work on a repo owned by someone else.) Then:
```bash
git config --global credential.helper 'cache --timeout=172800'
```
```bash
git push --dry-run origin sagyzdop/phase4
```
Username = your GitHub name, password = the token. Git then remembers it for 48 hours (in memory only, not on disk).

## 8. Start Claude and leave
Delete this file first (`rm PC_SETUP_COMMANDS.md`), then:
```bash
tmux new -s p012
```
```bash
claude
```
Switch Claude to auto mode (Shift+Tab) so it does not wait for permission prompts while you are away. Then type:

> Read PC_RUN.md and do exactly what it says.

Detach from tmux with `Ctrl+B`, then `D`. You can now log out; the session keeps running. Later, reattach with `tmux attach -t p012`.
