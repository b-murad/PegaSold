"""Tests for BaggedTrees (src/ensemble.py): bootstrap, OOB, soft voting, guards."""
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from src.ensemble import BaggedTrees


@pytest.fixture(scope="module")
def clf():
    return make_classification(n_samples=400, n_features=8, random_state=0)


@pytest.fixture(scope="module")
def reg():
    return make_regression(n_samples=400, n_features=6, noise=5.0, random_state=0)


@pytest.mark.parametrize("kw", [{"task": "x"}, {"n_estimators": 0}, {"max_samples": 0},
                                {"max_samples": 1.5}, {"max_samples": 0.0}])
def test_bad_parameters(kw):
    with pytest.raises(ValueError):
        BaggedTrees(**kw)


def test_n_draw_int_fraction_and_cap():
    assert BaggedTrees(max_samples=50)._n_draw(1000) == 50
    assert BaggedTrees(max_samples=5000)._n_draw(1000) == 1000
    assert BaggedTrees(max_samples=0.25)._n_draw(1000) == 250


def test_fit_builds_requested_number_of_trees(reg):
    X, y = reg
    m = BaggedTrees(task="regression", n_estimators=5, max_samples=100, max_depth=4).fit(X, y)
    assert len(m.estimators_) == 5 and m.n_features_in_ == 6


def test_fit_rejects_bad_shapes():
    with pytest.raises(ValueError):
        BaggedTrees().fit(np.zeros(5), np.zeros(5))
    with pytest.raises(ValueError):
        BaggedTrees().fit(np.zeros((4, 2)), np.zeros(3))


def test_regression_prediction_is_member_mean(reg):
    X, y = reg
    m = BaggedTrees(task="regression", n_estimators=4, max_samples=100, max_depth=4).fit(X, y)
    expected = np.mean([t.predict(X) for t in m.estimators_], axis=0)
    np.testing.assert_allclose(m.predict(X), expected)


def test_same_seed_same_ensemble(reg):
    X, y = reg
    a = BaggedTrees(task="regression", n_estimators=3, max_samples=100, max_depth=4, random_state=1).fit(X, y)
    b = BaggedTrees(task="regression", n_estimators=3, max_samples=100, max_depth=4, random_state=1).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_bagging_not_worse_than_one_tree_on_noisy_data(reg):
    X, y = reg
    Xtr, Xte, ytr, yte = X[:300], X[300:], y[:300], y[300:]
    one = BaggedTrees(task="regression", n_estimators=1, max_samples=300, max_depth=None).fit(Xtr, ytr)
    many = BaggedTrees(task="regression", n_estimators=25, max_samples=300, max_depth=None).fit(Xtr, ytr)
    assert np.mean((many.predict(Xte) - yte) ** 2) < np.mean((one.predict(Xte) - yte) ** 2)


def test_oob_off_by_default_and_on_when_asked(clf):
    X, y = clf
    assert BaggedTrees(task="classification", n_estimators=3, max_samples=100).fit(X, y).oob_score_ is None
    m = BaggedTrees(task="classification", n_estimators=15, max_samples=200, max_depth=6,
                    oob_score=True).fit(X, y)
    assert 0.7 < m.oob_score_ <= 1.0


def test_regression_oob_is_r2(reg):
    X, y = reg
    m = BaggedTrees(task="regression", n_estimators=15, max_samples=200, max_depth=6,
                    oob_score=True).fit(X, y)
    assert 0.3 < m.oob_score_ <= 1.0


def test_soft_voting_probabilities(clf):
    X, y = clf
    m = BaggedTrees(task="classification", n_estimators=6, max_samples=100, max_depth=4).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X), 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    expected = np.mean([t.predict_proba(X) for t in m.estimators_], axis=0)
    np.testing.assert_allclose(p, expected)
    np.testing.assert_array_equal(m.predict(X), m.classes_[p.argmax(axis=1)])


def test_aligned_proba_handles_member_missing_a_class():
    X = np.array([[0.0]] * 9 + [[1.0]])
    y = np.array([0] * 9 + [1])
    m = BaggedTrees(task="classification", n_estimators=1, max_samples=1.0).fit(X, y)
    only_zero = type(m.estimators_[0])(task="classification").fit(X[:9], y[:9])
    out = m._aligned_proba(only_zero, X)
    assert out.shape == (10, 2)
    np.testing.assert_array_equal(out[:, 1], 0.0)


def test_predict_proba_rejects_regression(reg):
    X, y = reg
    m = BaggedTrees(task="regression", n_estimators=2, max_samples=50).fit(X, y)
    with pytest.raises(ValueError):
        m.predict_proba(X)


def test_predict_before_fit():
    with pytest.raises(ValueError):
        BaggedTrees().predict(np.zeros((2, 2)))


def test_predict_wrong_shape(reg):
    X, y = reg
    m = BaggedTrees(task="regression", n_estimators=2, max_samples=50).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(np.zeros(6))
    with pytest.raises(ValueError):
        m.predict(np.zeros((3, 2)))
