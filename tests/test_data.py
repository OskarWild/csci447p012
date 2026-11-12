"""Phase 1: dataset shapes (CLAUDE.md 3), no index overlap, determinism, classes present."""
import numpy as np
import pytest

from p012.data import cv_splits, holdout_a, holdout_b, load_dataset, sweep_sizes, sweep_split

# name: (classes, pool, test), from CLAUDE.md section 3
EXPECTED = {
    "Abalone": (3, 3341, 836), "Adult": (2, 39073, 1000), "BreastCancer": (2, 455, 114),
    "Churn": (2, 5634, 1000), "HeartDisease": (2, 242, 61), "Sharktank": (2, 396, 99),
    "Statlog": (2, 552, 138), "Wine": (3, 142, 36), "bank": (2, 36168, 9043), "blood": (2, 598, 150),
    "calhousing": (2, 16512, 4128), "car": (4, 1382, 346), "creditg": (2, 800, 200),
    "diabetes": (2, 614, 154), "heart": (2, 734, 184), "jungle": (2, 35855, 8964),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_shapes(name):
    ds = load_dataset(name)
    k, n_pool, n_test = EXPECTED[name]
    assert (ds.n_classes, len(ds.y_pool), len(ds.y_test)) == (k, n_pool, n_test)
    assert len(ds.X_pool) == n_pool and len(ds.X_test) == n_test
    assert list(ds.X_pool.columns) == list(ds.X_test.columns)
    assert not ds.X_pool.isna().all().any()


def test_breastcancer_encoding():
    ds = load_dataset("BreastCancer")
    assert ds.classes == ["B", "M"] and ds.positive_label == "M"
    assert np.bincount(ds.y_pool).tolist() == [277, 178]
    assert np.bincount(ds.y_test).tolist() == [80, 34]


def _check(split, n_classes):
    tr, va, te = map(set, (split.train_idx, split.val_idx, split.test_idx))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert set(np.unique(split.y[split.train_idx])) == set(range(n_classes))


def _all_splits(ds, seed):
    out = [holdout_a(ds, seed), holdout_b(ds, seed)] + cv_splits(ds)
    return out + [sweep_split(ds, s, seed) for s in sweep_sizes(ds)]


@pytest.mark.parametrize("name", ["BreastCancer", "Abalone", "Wine", "creditg"])
def test_no_overlap_and_classes_present(name):
    ds = load_dataset(name)
    for split in _all_splits(ds, seed=0):
        _check(split, ds.n_classes)


def test_same_seed_same_sample():
    ds = load_dataset("BreastCancer")
    for a, b in zip(_all_splits(ds, 3), _all_splits(ds, 3)):
        for part in ("train_idx", "val_idx", "test_idx"):
            assert np.array_equal(getattr(a, part), getattr(b, part))
    assert not np.array_equal(holdout_a(ds, 0).train_idx, holdout_a(ds, 1).train_idx)


def test_sweep_sizes_and_counts():
    ds = load_dataset("BreastCancer")
    assert sweep_sizes(ds) == ["10", "25", "50", "100", "full"]
    s = sweep_split(ds, "50", 0)
    assert len(s.train_idx) == len(s.val_idx) == 50 and len(s.test_idx) == 114
    full = sweep_split(ds, "full", 0)
    assert (len(full.train_idx), len(full.val_idx)) == (181, 45)
    a = holdout_a(ds, 0)
    assert len(a.train_idx) + len(a.val_idx) == 455 and len(a.test_idx) == 114
    folds = cv_splits(ds)
    assert sorted(np.concatenate([f.test_idx for f in folds]).tolist()) == list(range(455))
