# QUESTIONS for the humans

Write a question here and STOP when a decision is scientific or irreversible.

## Q1 (Phase 3, 2026-11-12): validation AUC saturates at 1.0 on BreastCancer
Hold-out A validation has 91 rows. In seed 0, many PFNB-MLE trials reach validation AUC = 1.000, and Optuna keeps the first one, an overfit configuration (eta 0.29, depth 6, min_child_weight 0.01). Its test AUC is 0.9676 against 0.99+ in the other seeds (`results/runs/holdoutA/BreastCancer/pool/0/PFNB-MLE/metrics.json`). PFNB-Joint inherits these trees (0.9518).
Options: (a) keep the protocol (validation AUC, CLAUDE.md 4.6) and report it as a limitation; (b) break ties by validation log-loss; (c) use validation log-loss as the Optuna objective for all methods. (b) and (c) change a protocol decision, so we wait for the humans. Nothing has been changed.

## Q2 (Phase 3): the paper's enqueued s = 1e6 breaks AUC through float saturation
At s = 1e6 (and visibly from s ~ 1e2 on, `F07_bc_s_sweep`), sigmoid(b + s*z1) is exactly 0 or 1 for most rows, so AUC is computed on ties and falls far below TabPFN's own AUC. PFNB-Optuna picked s = 1e6 in hold-out A seed 3 (test AUC 0.9290). We keep the paper's enqueue (it is the paper replica) and report it. Should the report also include a variant without the 1e6 trial?
