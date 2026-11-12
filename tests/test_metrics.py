import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

from p012.metrics import ece, evaluate


def test_binary_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p1 = np.clip(0.3 * y + rng.uniform(0, 0.7, 200), 0.01, 0.99)
    proba = np.column_stack([1 - p1, p1])
    m = evaluate(y, proba)
    pred = (p1 >= 0.5).astype(int)
    assert np.isclose(m["auc"], roc_auc_score(y, p1))
    assert np.isclose(m["log_loss"], log_loss(y, proba))
    assert np.isclose(m["accuracy"], accuracy_score(y, pred))
    assert np.isclose(m["precision"], precision_score(y, pred))
    assert np.isclose(m["recall"], recall_score(y, pred))
    assert np.isclose(m["f1"], f1_score(y, pred))
    assert np.isclose(m["brier"], np.mean((p1 - y) ** 2))


def test_multiclass_matches_sklearn():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1, 2], 30)
    proba = rng.dirichlet(np.ones(3), 90)
    proba[np.arange(90), y] += 0.5
    proba /= proba.sum(1, keepdims=True)
    m = evaluate(y, proba)
    assert np.isclose(m["auc"], roc_auc_score(y, proba, multi_class="ovr"))
    assert np.isclose(m["f1"], f1_score(y, proba.argmax(1), average="macro"))


def test_ece_zero_when_calibrated():
    conf = np.r_[np.full(10, 0.1), np.full(10, 0.7)]
    y = np.r_[np.eye(10)[0], np.r_[np.ones(7), np.zeros(3)]]
    assert ece(conf, y) < 1e-12
    assert np.isclose(ece(np.full(10, 0.9), np.zeros(10)), 0.9)
