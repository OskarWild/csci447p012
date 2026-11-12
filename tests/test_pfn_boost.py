"""Phase 2 checks of PFN-Boost (CLAUDE.md 4.3-4.4)."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, softmax

from p012.pfn_boost import Fusion, Task, b0, fit_s, fit_trees, predict_proba
from p012.tabpfn_scores import binary_logit, logits

PARAMS = {"eta": 0.1, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0}


def make_task(n_classes, n=300, seed=0):
    """Synthetic task; the 'TabPFN' probabilities are noisy versions of the truth."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(3 * n, 5)), columns=[f"f{i}" for i in range(5)])
    true = softmax(X.to_numpy()[:, :n_classes] * 1.5, axis=1)
    y = np.array([rng.choice(n_classes, p=p) for p in true])
    p = softmax(np.log(true) + rng.normal(scale=0.5, size=true.shape), axis=1)
    s = [slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)]
    return Task(X.iloc[s[0]], y[s[0]], X.iloc[s[1]], y[s[1]], X.iloc[s[2]], y[s[2]], n_classes, seed,
                p[s[0]], p[s[1]], p[s[2]])


def test_s0_equals_xgb_baseline():
    t = make_task(2)
    pfnb = fit_trees(t, PARAMS, Fusion(0.0, b0(t.y_tr)))
    xgb = fit_trees(t, PARAMS, None, base_score=float(t.y_tr.mean()))
    zero = Fusion(0.0, b0(t.y_tr)).margin(t.z("te"))
    p_pfnb = predict_proba(pfnb.booster, t.X_te, zero, 2)
    p_xgb = predict_proba(xgb.booster, t.X_te, None, 2)
    assert pfnb.booster.best_iteration == xgb.booster.best_iteration
    assert np.abs(p_pfnb - p_xgb).max() < 1e-5


@pytest.mark.parametrize("n_classes", [2, 3])
def test_margin_applied_at_train_and_test(n_classes):
    """eta ~ 0 and 1 round: the trees add nothing, so test predictions are the transformer alone."""
    t = make_task(n_classes)
    fusion = Fusion(1.7, 0.4)
    tf = fit_trees(t, {**PARAMS, "eta": 1e-8}, fusion, rounds=1, early=False)
    z = t.z("te")
    got = predict_proba(tf.booster, t.X_te, fusion.margin(z), n_classes)
    want = np.column_stack([1 - expit(0.4 + 1.7 * z), expit(0.4 + 1.7 * z)]) if n_classes == 2 \
        else softmax(1.7 * z, axis=1)
    assert np.abs(got - want).max() < 1e-5
    # and the margin really matters: without it the prediction is different
    assert np.abs(predict_proba(tf.booster, t.X_te, None, n_classes) - want).max() > 1e-2


def test_smle_recovers_binary():
    rng = np.random.default_rng(0)
    z1 = rng.normal(scale=1.5, size=5000)
    y = (rng.random(5000) < expit(0.3 + 2.5 * z1)).astype(int)
    f = fit_s(z1, y)
    assert 2.3 <= f.s <= 2.7 and 0.2 <= f.b <= 0.4


def test_smle_recovers_multiclass_temperature():
    rng = np.random.default_rng(0)
    z = logits(softmax(rng.normal(size=(5000, 3)), axis=1))
    p = softmax(2.5 * z, axis=1)
    y = (rng.random(5000)[:, None] > p.cumsum(1)).sum(1)
    f = fit_s(z, y)
    assert 2.3 <= f.s <= 2.7 and f.b == 0.0


def test_smle_deterministic():
    t = make_task(2)
    a, b = fit_s(t.z("tr"), t.y_tr), fit_s(t.z("tr"), t.y_tr)
    assert (a.s, a.b) == (b.s, b.b)


def test_binary_logit_matches_centred_logits():
    p = softmax(np.random.default_rng(0).normal(size=(10, 2)), axis=1)
    z = logits(p)
    assert np.allclose(binary_logit(p), z[:, 1] - z[:, 0])
    assert np.allclose(z.sum(1), 0)
