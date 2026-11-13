"""Load the prepared datasets shipped in external/LLM-Boost/data and build the sampling protocols.

Every split is expressed as positional row indices into the dataset's train pool (and, for
hold-out B, into pool+test), so the same seed always gives the same rows and overlap is easy to test.
"""
import logging
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

from .config import load_config, path

log = logging.getLogger(__name__)

SWEEP_SIZES_KEY = "sizes"


@dataclass
class Dataset:
    name: str
    X_pool: pd.DataFrame          # features, positional index 0..n_pool-1
    y_pool: np.ndarray            # int labels 0..K-1
    X_test: pd.DataFrame
    y_test: np.ndarray
    classes: list                 # original label for each code; classes[1] is the positive class if binary
    cat_cols: list = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def positive_label(self):
        return self.classes[1] if self.n_classes == 2 else None


@dataclass
class Split:
    """One train/val/test split. `X` and `y` are the frame the indices point into."""
    X: pd.DataFrame
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    stratified: bool = True
    note: str = ""

    def part(self, which: str):
        idx = getattr(self, f"{which}_idx")
        return self.X.iloc[idx], self.y[idx]


def _read(name: str, part: str) -> pd.DataFrame:
    cfg = load_config()["paths"]
    f = path("data_root") / name / cfg["data_subdir"] / f"{part}.csv"
    # Positional index: some shipped CSVs (Adult) have duplicate index values.
    return pd.read_csv(f, index_col=0).reset_index(drop=True)


def _class_order(name: str, y_pool: pd.Series, y_test: pd.Series) -> list:
    """Label order. Binary: [negative, positive], positive from config or else the pool's minority class."""
    labels = sorted(set(y_pool) | set(y_test), key=str)
    if len(labels) != 2:
        return labels
    pos = load_config()["positive_class"].get(name)
    if pos is None:
        counts = y_pool.value_counts()
        pos = min(labels, key=lambda c: (counts.get(c, 0), str(c)))
    neg = next(c for c in labels if c != pos)
    return [neg, pos]


@lru_cache(maxsize=None)
def load_dataset(name: str) -> Dataset:
    label = load_config()["paths"]["label_col"]
    pool, test = _read(name, "train"), _read(name, "test")
    classes = _class_order(name, pool[label], test[label])
    code = {c: i for i, c in enumerate(classes)}
    y_pool = pool.pop(label).map(code).to_numpy(dtype=int)
    y_test = test.pop(label).map(code).to_numpy(dtype=int)

    # Ordinal-encode object columns, encoder fit on the pool (CLAUDE.md 4.7); unseen test values -> -1.
    cat_cols = list(pool.select_dtypes(exclude="number").columns)
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        pool[cat_cols] = enc.fit_transform(pool[cat_cols].astype(str))
        test[cat_cols] = enc.transform(test[cat_cols].astype(str))
    return Dataset(name, pool.astype(float), y_pool, test.astype(float), y_test, classes, cat_cols)


def as_xgb_frame(X: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Copy with categorical columns as pandas `category` dtype for enable_categorical=True."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(int).astype("category")
    return X


def _can_stratify(y: np.ndarray, n_per_part: int) -> bool:
    """True when a stratified part of size n_per_part gets at least 2 rows of every class (5.2)."""
    share = np.bincount(y) / len(y)
    return bool((share[share > 0] * n_per_part >= 2).all())


def _split(idx, y, size, seed, stratify):
    return train_test_split(idx, train_size=size, random_state=seed,
                            stratify=y[idx] if stratify else None)


def paper_full_counts(name: str) -> tuple[int, int] | None:
    ref = pd.read_csv(path("paper_reference"))
    row = ref[(ref.dataset == name) & (ref.size_label == "full")]
    return None if row.empty else (int(row.train.iloc[0]), int(row.val.iloc[0]))


def sweep_split(ds: Dataset, size_label: str, seed: int) -> Split:
    """Protocol 5.2: n train + n val from the pool without overlap; test = shipped test set.
    size_label 'full' uses the paper's train/val counts; 'full_all' uses the whole pool split 80/20."""
    n_pool = len(ds.y_pool)
    if size_label == "full":
        counts = paper_full_counts(ds.name)
        if counts is None:
            raise ValueError(f"{ds.name}: no paper 'full' row")
        n_tr, n_va = counts
        if n_tr + n_va > n_pool:
            # Paper counts larger than the repo pool (Statlog: 446+111 = 557 > 552): use the whole pool,
            # keeping the paper's train/val ratio. See docs/DEVIATIONS.md.
            n_va = int(round(n_pool * n_va / (n_tr + n_va)))
            n_tr = n_pool - n_va
            log.info("%s full: paper counts %s exceed pool %d; using %d+%d", ds.name, counts, n_pool, n_tr, n_va)
    elif size_label == "full_all":
        n_va = int(round(n_pool * load_config()["course_protocol"]["inner_val_fraction"]))
        n_tr = n_pool - n_va
    else:
        n_tr = n_va = int(size_label)
    if n_tr + n_va > n_pool:
        raise ValueError(f"{ds.name}: {n_tr}+{n_va} > pool {n_pool}")

    strat = _can_stratify(ds.y_pool, min(n_tr, n_va))
    n_cls = len(np.unique(ds.y_pool))
    # An unstratified (random) sample can miss a class in train or val (bank n=10: one class only,
    # which TabPFN cannot fit). Then redraw with a derived seed until both parts hold every class.
    # Attempt 0 uses `seed` itself, so every split that was already valid is unchanged. See DEVIATIONS.md.
    for attempt in range(10000):
        rs = seed if attempt == 0 else seed + 1_000_003 * attempt
        idx = np.arange(n_pool)
        if n_tr + n_va < n_pool:
            # A stratified discard needs >= 1 left-over row per class (Wine full: 141 of 142 rows);
            # with fewer, the few dropped rows are chosen at random.
            idx, _ = _split(idx, ds.y_pool, n_tr + n_va, rs, strat and n_pool - n_tr - n_va >= ds.n_classes)
        tr, va = _split(idx, ds.y_pool, n_tr, rs, strat)
        if len(np.unique(ds.y_pool[tr])) == n_cls and len(np.unique(ds.y_pool[va])) == n_cls:
            break
    else:
        raise ValueError(f"{ds.name} {size_label} seed {seed}: no sample with every class in train and val")
    note = "" if strat else "random (unstratified) sample: too few rows per class"
    if attempt:
        log.info("%s %s seed %d: random sample missed a class; redrawn %d time(s)", ds.name, size_label, seed, attempt)
        note += f"; redrawn {attempt}x so train and val hold every class"
    X = pd.concat([ds.X_pool, ds.X_test], ignore_index=True)
    y = np.concatenate([ds.y_pool, ds.y_test])
    test = np.arange(n_pool, n_pool + len(ds.y_test))
    return Split(X, y, np.sort(tr), np.sort(va), test, strat, note)


def _inner(train_pool_idx, y, seed):
    frac = load_config()["course_protocol"]["inner_val_fraction"]
    return train_test_split(train_pool_idx, test_size=frac, random_state=seed, stratify=y[train_pool_idx])


def holdout_a(ds: Dataset, seed: int) -> Split:
    """Protocol 5.1 hold-out A: shipped 455/114 split; stratified inner 80/20 train/val of the pool."""
    X = pd.concat([ds.X_pool, ds.X_test], ignore_index=True)
    y = np.concatenate([ds.y_pool, ds.y_test])
    n_pool = len(ds.y_pool)
    tr, va = _inner(np.arange(n_pool), y, seed)
    return Split(X, y, np.sort(tr), np.sort(va), np.arange(n_pool, len(y)))


def holdout_b(ds: Dataset, seed: int) -> Split:
    """Protocol 5.1 hold-out B: stratified 80/20 re-split of pool+test, then the same inner split."""
    X = pd.concat([ds.X_pool, ds.X_test], ignore_index=True)
    y = np.concatenate([ds.y_pool, ds.y_test])
    frac = load_config()["course_protocol"]["holdout_b_test_fraction"]
    pool, test = train_test_split(np.arange(len(y)), test_size=frac, random_state=seed, stratify=y)
    tr, va = _inner(pool, y, seed)
    return Split(X, y, np.sort(tr), np.sort(va), np.sort(test))


def cv_splits(ds: Dataset) -> list[Split]:
    """Protocol 5.1 stratified 5-fold CV on the pool; each fold's training part gets the inner 80/20 split.
    The inner split uses the fold number as its seed."""
    cp = load_config()["course_protocol"]
    skf = StratifiedKFold(cp["cv_folds"], shuffle=True, random_state=cp["cv_seed"])
    out = []
    for k, (pool, test) in enumerate(skf.split(ds.X_pool, ds.y_pool)):
        tr, va = _inner(pool, ds.y_pool, k)
        out.append(Split(ds.X_pool, ds.y_pool, np.sort(tr), np.sort(va), np.sort(test)))
    return out


def sweep_sizes(ds: Dataset) -> list[str]:
    """Size labels that fit this dataset's pool (2n <= pool), plus 'full' if the paper has it."""
    cfg = load_config()["sweep"]
    labels = [str(n) for n in cfg[SWEEP_SIZES_KEY] if 2 * n <= len(ds.y_pool)]
    if cfg["include_full"] and paper_full_counts(ds.name) is not None:
        labels.append("full")
    return labels
