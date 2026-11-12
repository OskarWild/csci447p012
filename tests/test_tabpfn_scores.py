"""Out-of-fold scores never come from a TabPFN that held the row in its context; cache round-trip."""
import numpy as np
import pytest

from p012.tabpfn_scores import compute_scores, load_scores, save_scores


def fake_fit_predict(Xc, yc, Xq, seed):
    """Column 0 of X is a row id. P(class 1) = 1 if the queried row was in the context, else 0."""
    inside = np.isin(Xq[:, 0], Xc[:, 0]).astype(float)
    return np.column_stack([1 - inside, inside])


def _data(n_tr, seed=0):
    rng = np.random.default_rng(seed)
    X_tr = np.column_stack([np.arange(n_tr), rng.normal(size=n_tr)])
    y_tr = np.arange(n_tr) % 2
    X_va = np.column_stack([1000 + np.arange(7), rng.normal(size=7)])
    X_te = np.column_stack([2000 + np.arange(5), rng.normal(size=5)])
    return X_tr, y_tr, X_va, X_te


@pytest.mark.parametrize("n_tr,scheme", [(60, "skf5"), (10, "loo")])
def test_oof_rows_never_in_context(n_tr, scheme):
    X_tr, y_tr, X_va, X_te = _data(n_tr)
    s = compute_scores(X_tr, y_tr, X_va, X_te, 0, 2, "oof", fake_fit_predict)
    assert s["cf_scheme"] == scheme
    assert not np.isnan(s["p_train"]).any()
    assert (s["p_train"][:, 1] == 0).all()        # every training row scored from outside its context
    assert (s["p_val"][:, 1] == 0).all() and (s["p_test"][:, 1] == 0).all()


def test_insample_mode_holds_rows_in_context():
    X_tr, y_tr, X_va, X_te = _data(60)
    s = compute_scores(X_tr, y_tr, X_va, X_te, 0, 2, "insample", fake_fit_predict)
    assert (s["p_train"][:, 1] == 1).all()


def test_cache_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    scores = {"p_train": rng.random((20, 3)), "p_val": rng.random((5, 3)), "p_test": rng.random((4, 3)),
              "cf_scheme": "skf5"}
    idx = {"train": np.arange(20), "val": np.arange(20, 25), "test": np.arange(25, 29)}
    f = tmp_path / "a" / "oof.npz"
    save_scores(f, scores, idx)
    d = load_scores(f)
    for k in ("p_train", "p_val", "p_test"):
        assert np.array_equal(d[k], scores[k])
    for k, v in idx.items():
        assert np.array_equal(d[f"idx_{k}"], v)
    assert d["cf_scheme"] == "skf5"
