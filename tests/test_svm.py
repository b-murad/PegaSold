"""Extensive tests for PegasosSVM (src/svm.py).  Run: pytest tests/test_svm.py -v"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from svm import PegasosSVM
from sklearn.metrics import roc_auc_score
from svm import RandomFourierPegasosSVM


# ------------------------------------------------------------------ helpers
def blobs(n=400, sep=3.0, d=2, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(+sep / 2, 1, (n // 2, d)),
                   rng.normal(-sep / 2, 1, (n // 2, d))])
    y = np.r_[np.ones(n // 2), -np.ones(n // 2)]
    return X, y


def reference_objective(X, y01, lam, w, b):
    s = np.where(y01 == 1, 1.0, -1.0)
    return 0.5 * lam * w @ w + np.maximum(0, 1 - s * (X @ w + b)).mean()


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(n_samples=1000, n_features=10, random_state=0)
    return StandardScaler().fit_transform(X), y


# ------------------------------------------------------- constructor checks
class TestInit:
    @pytest.mark.parametrize("lam", [0, -1, -0.001])
    def test_bad_lambda(self, lam):
        with pytest.raises(ValueError):
            PegasosSVM(lambda_param=lam)

    @pytest.mark.parametrize("ep", [0, -5])
    def test_bad_epochs(self, ep):
        with pytest.raises(ValueError):
            PegasosSVM(epochs=ep)

    @pytest.mark.parametrize("lr", [0, -0.1])
    def test_bad_numeric_learning_rate(self, lr):
        with pytest.raises(ValueError):
            PegasosSVM(learning_rate=lr)

    def test_bad_string_learning_rate(self):
        with pytest.raises(ValueError):
            PegasosSVM(learning_rate="foo")

    def test_constant_lr_times_lambda_must_be_below_one(self):
        with pytest.raises(ValueError):
            PegasosSVM(lambda_param=0.01, learning_rate=200.0)

    def test_defaults_and_types(self):
        m = PegasosSVM(lambda_param=1, epochs=3.0)
        assert isinstance(m.lambda_param, float) and isinstance(m.epochs, int)
        assert m.w is None and m.b == 0.0

    def test_accepts_numpy_scalar_lr(self):
        PegasosSVM(learning_rate=np.float64(0.1))


# ---------------------------------------------------------- input validation
class TestFitValidation:
    def test_x_must_be_2d(self):
        with pytest.raises(ValueError):
            PegasosSVM().fit(np.zeros(5), np.array([0, 1, 0, 1, 0]))

    def test_y_must_be_1d(self):
        with pytest.raises(ValueError):
            PegasosSVM().fit(np.zeros((4, 2)), np.zeros((4, 1)))

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            PegasosSVM().fit(np.zeros((4, 2)), np.array([0, 1, 0]))

    def test_empty(self):
        with pytest.raises(ValueError):
            PegasosSVM().fit(np.zeros((0, 2)), np.array([]))

    def test_zero_features(self):
        with pytest.raises(ValueError):
            PegasosSVM().fit(np.zeros((4, 0)), np.array([0, 1, 0, 1]))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_X(self, bad):
        X = np.array([[0.0, 1.0], [bad, 2.0], [1.0, 1.0], [2.0, 0.0]])
        with pytest.raises(ValueError):
            PegasosSVM().fit(X, np.array([0, 1, 0, 1]))

    def test_single_class(self):
        with pytest.raises(ValueError):
            PegasosSVM().fit(np.random.rand(10, 2), np.ones(10))

    def test_three_classes(self):
        with pytest.raises(ValueError):
            PegasosSVM().fit(np.random.rand(9, 2), np.array([0, 1, 2] * 3))

    def test_predict_before_fit(self):
        with pytest.raises(ValueError):
            PegasosSVM().predict(np.zeros((2, 2)))
        with pytest.raises(ValueError):
            PegasosSVM().decision_function(np.zeros((2, 2)))

    def test_predict_wrong_feature_count(self):
        X, y = blobs()
        m = PegasosSVM(epochs=2).fit(X, y)
        with pytest.raises(ValueError):
            m.predict(np.zeros((3, 5)))

    def test_predict_1d_input(self):
        X, y = blobs()
        m = PegasosSVM(epochs=2).fit(X, y)
        with pytest.raises(ValueError):
            m.predict(np.zeros(2))

    def test_predict_nan(self):
        X, y = blobs()
        m = PegasosSVM(epochs=2).fit(X, y)
        with pytest.raises(ValueError):
            m.predict(np.array([[np.nan, 0.0]]))

    def test_fit_returns_self(self):
        X, y = blobs()
        m = PegasosSVM(epochs=1)
        assert m.fit(X, y) is m

    def test_does_not_mutate_inputs(self):
        X, y = blobs()
        X0, y0 = X.copy(), y.copy()
        PegasosSVM(epochs=3).fit(X, y)
        np.testing.assert_array_equal(X, X0)
        np.testing.assert_array_equal(y, y0)


# ------------------------------------------------------------- label handling
class TestLabels:
    @pytest.mark.parametrize("labels", [(0, 1), (-1, 1), (3, 7), (False, True)])
    def test_numeric_labels_roundtrip(self, labels):
        X, y = blobs(sep=8)
        yy = np.where(y == 1, labels[1], labels[0])
        m = PegasosSVM(lambda_param=0.001, epochs=20).fit(X, yy)
        pred = m.predict(X)
        assert set(np.unique(pred)) <= set(labels)
        assert (pred == yy).mean() > 0.99

    def test_string_labels(self):
        X, y = blobs(sep=8)
        yy = np.where(y == 1, "cat", "dog")
        m = PegasosSVM(lambda_param=0.001, epochs=20).fit(X, yy)
        assert list(m.classes_) == ["cat", "dog"]
        pred = m.predict(X)
        assert pred.dtype.kind == "U" and (pred == yy).mean() > 0.99

    def test_positive_class_is_second_sorted_class(self):
        X, y = blobs(sep=8)
        m = PegasosSVM(lambda_param=0.001, epochs=20).fit(X, y)  # classes_ = [-1, 1]
        assert m.decision_function(np.array([[5.0, 5.0]]))[0] > 0
        assert m.decision_function(np.array([[-5.0, -5.0]]))[0] < 0

    def test_flipping_labels_flips_weights(self):
        X, y = blobs(sep=4)
        a = PegasosSVM(lambda_param=0.01, epochs=30, random_state=1).fit(X, y)
        b = PegasosSVM(lambda_param=0.01, epochs=30, random_state=1).fit(X, -y)
        np.testing.assert_allclose(a.w, -b.w, atol=1e-9)
        assert a.b == pytest.approx(-b.b)

    def test_predict_matches_decision_sign(self):
        X, y = blobs()
        m = PegasosSVM(epochs=5).fit(X, y)
        s = m.decision_function(X)
        np.testing.assert_array_equal(m.predict(X), np.where(s >= 0, 1.0, -1.0))

    def test_list_and_int_inputs(self):
        X = [[0, 0], [1, 1], [0, 1], [5, 5], [6, 6], [5, 6]]
        y = [0, 0, 0, 1, 1, 1]
        m = PegasosSVM(lambda_param=0.01, epochs=50).fit(X, y)
        assert (m.predict(X) == np.array(y)).all()

    def test_pandas_like_input(self):
        pd = pytest.importorskip("pandas")
        X, y = blobs(sep=8)
        m = PegasosSVM(lambda_param=0.001, epochs=10).fit(pd.DataFrame(X), pd.Series(y))
        assert (m.predict(pd.DataFrame(X)) == y).mean() > 0.99


# ---------------------------------------------------------------- behaviour
class TestLearning:
    def test_separable_data_perfect(self):
        X, y = blobs(n=400, sep=10)
        m = PegasosSVM(lambda_param=0.001, epochs=30).fit(X, y)
        assert (m.predict(X) == y).all()

    @pytest.mark.parametrize("sep,min_acc", [(2, 0.80), (3, 0.90), (5, 0.97)])
    def test_accuracy_increases_with_separation(self, sep, min_acc):
        X, y = blobs(n=2000, sep=sep, seed=3)
        m = PegasosSVM(lambda_param=0.01, epochs=30).fit(X, y)
        assert (m.predict(X) == y).mean() >= min_acc

    def test_generalizes_to_holdout(self, clf_data):
        X, y = clf_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        m = PegasosSVM(lambda_param=0.01, epochs=50).fit(Xtr, ytr)
        ref = SVC(kernel="linear", C=1 / (0.01 * len(ytr))).fit(Xtr, ytr)
        assert abs((m.predict(Xte) == yte).mean() - ref.score(Xte, yte)) < 0.03

    def test_1d_feature(self):
        rng = np.random.default_rng(0)
        X = np.r_[rng.normal(3, .5, 100), rng.normal(-3, .5, 100)].reshape(-1, 1)
        y = np.r_[np.ones(100), np.zeros(100)]
        m = PegasosSVM(lambda_param=0.01, epochs=20).fit(X, y)
        assert (m.predict(X) == y).mean() > 0.99

    def test_single_epoch_runs(self):
        X, y = blobs()
        m = PegasosSVM(epochs=1).fit(X, y)
        assert len(m.objective_history_) == 1 and np.isfinite(m.w).all()

    def test_tiny_dataset(self):
        m = PegasosSVM(lambda_param=0.1, epochs=100).fit([[0.0], [1.0]], [0, 1])
        assert list(m.predict([[0.0], [1.0]])) == [0, 1]

    def test_imbalanced_classes_still_learn(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(3, 1, (950, 2)), rng.normal(-3, 1, (50, 2))])
        y = np.r_[np.ones(950), -np.ones(50)]
        m = PegasosSVM(lambda_param=0.001, epochs=30).fit(X, y)
        assert (m.predict(X) == y).mean() > 0.98

    def test_duplicate_points(self):
        X = np.array([[0.0, 0], [0, 0], [4, 4], [4, 4]] * 20)
        y = np.array([0, 0, 1, 1] * 20)
        m = PegasosSVM(lambda_param=0.01, epochs=30).fit(X, y)
        assert (m.predict(X) == y).all()

    def test_constant_feature_ignored(self):
        X, y = blobs(sep=8)
        X = np.hstack([X, np.ones((len(X), 1))])
        m = PegasosSVM(lambda_param=0.001, epochs=20).fit(X, y)
        assert (m.predict(X) == y).mean() > 0.99

    def test_nonlinear_data_limited_but_valid(self):
        X, y = make_moons(500, noise=0.1, random_state=0)
        X = StandardScaler().fit_transform(X)
        m = PegasosSVM(lambda_param=0.01, epochs=30).fit(X, y)
        assert 0.7 < (m.predict(X) == y).mean() < 1.0   # linear model can't be perfect

    def test_refit_resets_state(self):
        X, y = blobs(sep=8)
        m = PegasosSVM(lambda_param=0.001, epochs=5)
        m.fit(X, y)
        w1, b1 = m.w.copy(), m.b
        m.fit(X[:, :1], y)          # different feature count
        assert m.w.shape == (1,) and m.n_features_in_ == 1
        m.fit(X, y)
        np.testing.assert_allclose(m.w, w1)
        assert m.b == pytest.approx(b1)

    def test_larger_lambda_shrinks_weights(self, clf_data):
        X, y = clf_data
        norms = [np.linalg.norm(PegasosSVM(lambda_param=l, epochs=30).fit(X, y).w)
                 for l in (1e-3, 1e-2, 1e-1, 1.0)]
        assert all(a > b for a, b in zip(norms, norms[1:])), norms

    def test_huge_lambda_gives_near_zero_weights(self, clf_data):
        X, y = clf_data
        m = PegasosSVM(lambda_param=1e3, epochs=10).fit(X, y)
        assert np.linalg.norm(m.w) < 0.05

    def test_constant_learning_rate_mode(self, clf_data):
        X, y = clf_data
        m = PegasosSVM(lambda_param=0.01, epochs=30, learning_rate=0.05).fit(X, y)
        assert (m.predict(X) == y).mean() > 0.85

    def test_weights_respect_pegasos_norm_bound_loosely(self, clf_data):
        # Optimal ||w|| <= 1/sqrt(lambda); SGD iterate should be in that ballpark.
        X, y = clf_data
        lam = 0.01
        m = PegasosSVM(lambda_param=lam, epochs=100).fit(X, y)
        assert np.linalg.norm(m.w) <= 1.2 / np.sqrt(lam)


# ------------------------------------------------------------- determinism
class TestDeterminism:
    def test_same_seed_same_result(self, clf_data):
        X, y = clf_data
        a = PegasosSVM(epochs=5, random_state=7).fit(X, y)
        b = PegasosSVM(epochs=5, random_state=7).fit(X, y)
        np.testing.assert_array_equal(a.w, b.w)
        assert a.b == b.b and a.objective_history_ == b.objective_history_

    def test_different_seed_different_path_similar_quality(self, clf_data):
        X, y = clf_data
        a = PegasosSVM(epochs=100, random_state=1).fit(X, y)
        b = PegasosSVM(epochs=100, random_state=2).fit(X, y)
        assert not np.array_equal(a.w, b.w)
        assert abs(a.objective_history_[-1] - b.objective_history_[-1]) < 0.01

    def test_row_order_invariance_of_quality(self, clf_data):
        X, y = clf_data
        perm = np.random.default_rng(0).permutation(len(y))
        a = PegasosSVM(epochs=100).fit(X, y)
        b = PegasosSVM(epochs=100).fit(X[perm], y[perm])
        assert abs(a.objective_history_[-1] - b.objective_history_[-1]) < 0.01


# -------------------------------------------------------- objective & history
class TestObjective:
    def test_history_length(self):
        X, y = blobs()
        assert len(PegasosSVM(epochs=7).fit(X, y).objective_history_) == 7

    def test_history_finite_and_nonnegative(self, clf_data):
        X, y = clf_data
        h = PegasosSVM(epochs=20).fit(X, y).objective_history_
        assert np.isfinite(h).all() and min(h) >= 0

    def test_objective_improves_overall(self, clf_data):
        X, y = clf_data
        h = PegasosSVM(lambda_param=0.01, epochs=50).fit(X, y).objective_history_
        assert h[-1] < h[0]
        assert h[-1] <= min(h) * 1.05  # last epoch near the best

    def test_internal_objective_matches_reference(self, clf_data):
        X, y = clf_data
        m = PegasosSVM(epochs=10).fit(X, y)
        s = np.where(y == m.classes_[1], 1.0, -1.0)
        assert m._objective(X, s) == pytest.approx(reference_objective(X, y, 0.01, m.w, m.b))

    def test_last_history_equals_final_objective(self, clf_data):
        X, y = clf_data
        m = PegasosSVM(epochs=10).fit(X, y)
        s = np.where(y == m.classes_[1], 1.0, -1.0)
        assert m.objective_history_[-1] == pytest.approx(m._objective(X, s))

    def test_zero_weights_objective_is_one(self):
        m = PegasosSVM()
        m.w, m.b = np.zeros(2), 0.0
        assert m._objective(np.random.rand(10, 2), np.ones(10)) == pytest.approx(1.0)


# --------------------------------------------------------- fitted attributes
class TestAttributes:
    def test_margin_width(self):
        X, y = blobs()
        m = PegasosSVM(epochs=5).fit(X, y)
        assert m.margin_width_ == pytest.approx(2 / np.linalg.norm(m.w))

    def test_support_vectors_definition(self):
        X, y = blobs(sep=2)
        m = PegasosSVM(epochs=10).fit(X, y)
        s = np.where(y == m.classes_[1], 1.0, -1.0)
        np.testing.assert_array_equal(m.support_vector_mask_, s * m.decision_function(X) <= 1.0)
        assert m.support_vectors_.shape == (m.support_vector_mask_.sum(), X.shape[1])

    def test_well_separated_has_fewer_support_vectors(self):
        Xa, ya = blobs(n=600, sep=2)
        Xb, yb = blobs(n=600, sep=12)
        a = PegasosSVM(lambda_param=0.001, epochs=20).fit(Xa, ya)
        b = PegasosSVM(lambda_param=0.001, epochs=20).fit(Xb, yb)
        assert b.support_vector_mask_.sum() < a.support_vector_mask_.sum()

    def test_classes_and_feature_count(self):
        X, y = blobs(d=4)
        m = PegasosSVM(epochs=1).fit(X, y)
        assert m.n_features_in_ == 4 and list(m.classes_) == [-1.0, 1.0]
        assert m.w.shape == (4,)


# -------------------------------------------------- comparison with sklearn
class TestAgainstSklearn:
    @pytest.mark.parametrize("lam", [0.1, 0.01])
    def test_matches_exact_solver(self, clf_data, lam):
        X, y = clf_data
        peg = PegasosSVM(lambda_param=lam, epochs=200).fit(X, y)
        ref = SVC(kernel="linear", C=1 / (lam * len(y)), tol=1e-6).fit(X, y)
        w_ref, b_ref = ref.coef_.ravel(), ref.intercept_[0]

        obj_p = peg.objective_history_[-1]
        obj_r = reference_objective(X, y, lam, w_ref, b_ref)
        assert obj_p >= obj_r - 1e-4                      # can't beat the optimum
        assert (obj_p - obj_r) / obj_r < 0.01             # within 1%

        cos = peg.w @ w_ref / (np.linalg.norm(peg.w) * np.linalg.norm(w_ref))
        assert cos > 0.999
        assert abs((peg.predict(X) == y).mean() - ref.score(X, y)) < 0.01
        assert peg.b == pytest.approx(b_ref, abs=0.1)

    def test_more_epochs_gets_closer_to_optimum(self, clf_data):
        X, y = clf_data
        lam = 0.01
        ref = SVC(kernel="linear", C=1 / (lam * len(y)), tol=1e-6).fit(X, y)
        opt = reference_objective(X, y, lam, ref.coef_.ravel(), ref.intercept_[0])
        gaps = [PegasosSVM(lambda_param=lam, epochs=e).fit(X, y).objective_history_[-1] - opt
                for e in (1, 10, 100)]
        assert gaps[0] > gaps[2]
        assert gaps[2] < 0.01

    def test_hyperplane_agreement_on_random_points(self, clf_data):
        X, y = clf_data
        lam = 0.01
        peg = PegasosSVM(lambda_param=lam, epochs=200).fit(X, y)
        ref = SVC(kernel="linear", C=1 / (lam * len(y))).fit(X, y)
        Z = np.random.default_rng(0).normal(size=(2000, X.shape[1]))
        assert (peg.predict(Z) == ref.predict(Z)).mean() > 0.98


# ------------------------------------------------ regression: bias explosion
class TestBiasRegression:
    @pytest.mark.parametrize("lam", [1e-2, 1e-3, 1e-4])
    def test_bias_stays_small_on_centered_data(self, lam):
        """Uncapped bias step 1/(lam*t) used to give |b| ~ 70-700 on centered, separable data."""
        X, y = blobs(n=400, sep=8)
        m = PegasosSVM(lambda_param=lam, epochs=20).fit(X, y)
        assert abs(m.b) < 3.0
        assert (m.predict(X) == y).all()


# ------------------------------------------------ lambda, ROC and RFF
class TestLambdaSensitivity:
    @pytest.mark.parametrize("lam", [0.1, 0.01, 0.001, 0.0001])
    def test_each_swept_lambda_learns(self, clf_data, lam):
        X, y = clf_data
        m = PegasosSVM(lambda_param=lam, epochs=50).fit(X, y)
        assert (m.predict(X) == y).mean() > 0.8

    def test_smaller_lambda_not_worse_on_validation(self, clf_data):
        # Project finding: 0.01 -> 0.001 gained +1.3 pp. Synthetic data can't
        # reproduce the size, only that the smaller lambda doesn't hurt.
        X, y = clf_data
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0)
        acc = {l: (PegasosSVM(lambda_param=l, epochs=50, random_state=0)
                   .fit(Xtr, ytr).predict(Xva) == yva).mean()
               for l in (0.01, 0.001)}
        assert acc[0.001] >= acc[0.01] - 0.01, acc


def test_auc_close_to_sklearn(clf_data):
    X, y = clf_data
    lam = 0.01
    peg = PegasosSVM(lambda_param=lam, epochs=100).fit(X, y)
    ref = SVC(kernel="linear", C=1 / (lam * len(y))).fit(X, y)
    auc_p = roc_auc_score(y, peg.decision_function(X))
    assert auc_p > 0.5 and abs(auc_p - roc_auc_score(y, ref.decision_function(X))) < 0.01


def test_random_fourier_pegasos_learns_nonlinear_pattern():
    rng = np.random.default_rng(10)
    angles = rng.uniform(0, 2*np.pi, 700)
    radius = np.where(np.arange(len(angles)) % 2, 3.0, 1.0) + rng.normal(0, .12, len(angles))
    X = np.c_[radius*np.cos(angles), radius*np.sin(angles)]
    y = (radius > 2).astype(int)
    model = RandomFourierPegasosSVM(lambda_param=.001, epochs=25,
                                    n_components=256, gamma=.5, random_state=4).fit(X, y)
    assert (model.predict(X) == y).mean() > .94
    assert model.margin_width_ > 0
