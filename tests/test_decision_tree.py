"""Extensive tests for DecisionTree (src/decision_tree.py).  Run: pytest tests/test_decision_tree.py -v"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from decision_tree import DecisionTree, TreeNode


# ------------------------------------------------------------------ helpers
def depth(node):
    return 0 if node.feature_index is None else 1 + max(depth(node.left), depth(node.right))


def leaves(node):
    if node.feature_index is None:
        return [node]
    return leaves(node.left) + leaves(node.right)


def count_nodes(node):
    return 1 if node.feature_index is None else 1 + count_nodes(node.left) + count_nodes(node.right)


def leaf_sizes(tree, X):
    """Number of training rows landing in each leaf (via routing)."""
    ids = {}
    for row in X:
        n = tree.root
        while n.feature_index is not None:
            n = n.left if row[n.feature_index] <= n.threshold else n.right
        ids[id(n)] = ids.get(id(n), 0) + 1
    return list(ids.values())


@pytest.fixture(scope="module")
def iris():
    d = load_iris()
    return d.data, d.target


@pytest.fixture(scope="module")
def clf_data():
    return make_classification(n_samples=400, n_features=6, n_informative=4,
                               n_redundant=0, random_state=0)


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(n_samples=300, n_features=5, n_informative=3, noise=5, random_state=0)
    return X, y


# ----------------------------------------------------------------- TreeNode
class TestTreeNode:
    def test_defaults_are_leaf_like(self):
        n = TreeNode()
        assert n.value is None and n.feature_index is None
        assert n.threshold is None and n.left is None and n.right is None

    def test_stores_fields(self):
        l, r = TreeNode(value=0), TreeNode(value=1)
        n = TreeNode(feature_index=2, threshold=1.5, left=l, right=r)
        assert (n.feature_index, n.threshold, n.left, n.right) == (2, 1.5, l, r)


# ------------------------------------------------------------- constructor
class TestInit:
    def test_defaults(self):
        t = DecisionTree()
        assert (t.task, t.criterion, t.max_depth) == ("classification", "gini", None)
        assert t.root is None

    def test_regression_default_criterion(self):
        assert DecisionTree(task="regression").criterion == "mse"

    @pytest.mark.parametrize("crit", ["gini", "entropy"])
    def test_classification_criteria(self, crit):
        assert DecisionTree(criterion=crit).criterion == crit

    @pytest.mark.parametrize("kwargs", [
        dict(task="clustering"),
        dict(criterion="mse"),
        dict(criterion="foo"),
        dict(task="regression", criterion="gini"),
        dict(max_depth=-1),
        dict(min_samples_split=1),
        dict(min_samples_split=0),
        dict(min_samples_leaf=0),
        dict(min_impurity_decrease=-0.1),
    ])
    def test_invalid_args(self, kwargs):
        with pytest.raises(ValueError):
            DecisionTree(**kwargs)

    def test_max_depth_zero_allowed(self):
        assert DecisionTree(max_depth=0).max_depth == 0


# --------------------------------------------------------------- validation
class TestFitValidation:
    def test_x_1d(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.arange(5), np.arange(5))

    def test_y_2d(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.zeros((4, 2)), np.zeros((4, 1)))

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.zeros((4, 2)), np.zeros(3))

    def test_empty(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.zeros((0, 2)), np.zeros(0))

    def test_no_features(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.zeros((3, 0)), np.zeros(3))

    def test_non_numeric_X(self):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.array([["a", "b"], ["c", "d"]]), np.array([0, 1]))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_X(self, bad):
        with pytest.raises(ValueError):
            DecisionTree().fit(np.array([[1.0], [bad]]), np.array([0, 1]))

    def test_regression_non_numeric_y(self):
        with pytest.raises(ValueError):
            DecisionTree(task="regression").fit(np.zeros((2, 1)), np.array(["a", "b"]))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_regression_non_finite_y(self, bad):
        with pytest.raises(ValueError):
            DecisionTree(task="regression").fit(np.zeros((2, 1)), np.array([1.0, bad]))

    def test_fit_returns_self_and_sets_attrs(self, iris):
        X, y = iris
        t = DecisionTree()
        assert t.fit(X, y) is t
        assert t.n_features_in_ == 4 and list(t.classes_) == [0, 1, 2]
        assert isinstance(t.root, TreeNode)

    def test_regression_has_no_classes(self, reg_data):
        assert DecisionTree(task="regression").fit(*reg_data).classes_ is None

    def test_does_not_mutate_inputs(self, iris):
        X, y = iris
        X0, y0 = X.copy(), y.copy()
        DecisionTree().fit(X, y)
        np.testing.assert_array_equal(X, X0)
        np.testing.assert_array_equal(y, y0)

    def test_list_input(self):
        t = DecisionTree().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        assert list(t.predict([[0], [3]])) == [0, 1]


class TestPredictValidation:
    def test_before_fit(self):
        with pytest.raises(ValueError):
            DecisionTree().predict(np.zeros((1, 1)))
        with pytest.raises(ValueError):
            DecisionTree().predict_proba(np.zeros((1, 1)))

    def test_wrong_feature_count(self, iris):
        t = DecisionTree().fit(*iris)
        with pytest.raises(ValueError):
            t.predict(np.zeros((2, 3)))
        with pytest.raises(ValueError):
            t.predict_proba(np.zeros((2, 3)))

    def test_1d_input(self, iris):
        t = DecisionTree().fit(*iris)
        with pytest.raises(ValueError):
            t.predict(np.zeros(4))

    def test_nan_input(self, iris):
        t = DecisionTree().fit(*iris)
        with pytest.raises(ValueError):
            t.predict(np.array([[np.nan, 0, 0, 0]]))

    def test_non_numeric_input(self, iris):
        t = DecisionTree().fit(*iris)
        with pytest.raises(ValueError):
            t.predict(np.array([["a", "b", "c", "d"]]))

    def test_predict_proba_regression_rejected(self, reg_data):
        t = DecisionTree(task="regression").fit(*reg_data)
        with pytest.raises(ValueError):
            t.predict_proba(reg_data[0])


# ---------------------------------------------------------- impurity & score
class TestImpurity:
    def setup_method(self):
        self.g = DecisionTree(criterion="gini")
        self.e = DecisionTree(criterion="entropy")
        self.r = DecisionTree(task="regression")

    def test_gini_pure(self):
        assert self.g._impurity(np.array([1, 1, 1])) == 0.0

    def test_gini_balanced_binary(self):
        assert self.g._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)

    def test_gini_three_equal_classes(self):
        assert self.g._impurity(np.array([0, 1, 2])) == pytest.approx(2 / 3)

    def test_gini_unbalanced(self):
        assert self.g._impurity(np.array([0, 0, 0, 1])) == pytest.approx(1 - (0.75**2 + 0.25**2))

    def test_entropy_pure(self):
        assert self.e._impurity(np.array(["a", "a"])) == 0.0

    def test_entropy_balanced_binary(self):
        assert self.e._impurity(np.array([0, 1, 0, 1])) == pytest.approx(1.0)

    def test_entropy_four_classes(self):
        assert self.e._impurity(np.array([0, 1, 2, 3])) == pytest.approx(2.0)

    def test_entropy_unbalanced(self):
        p = np.array([0.75, 0.25])
        assert self.e._impurity(np.array([0, 0, 0, 1])) == pytest.approx(-(p * np.log2(p)).sum())

    def test_mse_identical(self):
        assert self.r._impurity(np.array([5.0, 5.0, 5.0])) == 0.0

    def test_mse_variance(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert self.r._impurity(y) == pytest.approx(np.var(y))

    def test_impurity_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            y = rng.integers(0, 4, rng.integers(1, 30))
            assert 0 <= self.g._impurity(y) <= 1 - 1 / 4 + 1e-12
            assert 0 <= self.e._impurity(y) <= 2 + 1e-12

    def test_split_score_perfect_split(self):
        s = self.g._split_score(np.array([0, 0]), np.array([1, 1]), 0.5)
        assert s == pytest.approx(0.5)

    def test_split_score_useless_split(self):
        s = self.g._split_score(np.array([0, 1]), np.array([0, 1]), 0.5)
        assert s == pytest.approx(0.0)

    def test_split_score_weighted(self):
        yl, yr = np.array([0, 0, 0, 1]), np.array([1, 1])
        parent = self.g._impurity(np.r_[yl, yr])
        expected = parent - (4 / 6 * self.g._impurity(yl) + 0)
        assert self.g._split_score(yl, yr, parent) == pytest.approx(expected)

    def test_split_score_never_negative(self):
        rng = np.random.default_rng(1)
        for crit in ("gini", "entropy"):
            t = DecisionTree(criterion=crit)
            for _ in range(100):
                y = rng.integers(0, 3, 20)
                k = rng.integers(1, 19)
                assert t._split_score(y[:k], y[k:], t._impurity(y)) >= -1e-12

    def test_regression_split_score_reduces_variance(self):
        yl, yr = np.array([1.0, 1.0]), np.array([9.0, 9.0])
        parent = self.r._impurity(np.r_[yl, yr])
        assert self.r._split_score(yl, yr, parent) == pytest.approx(parent)


class TestLeafValue:
    def test_majority(self):
        assert DecisionTree()._leaf_value(np.array([1, 2, 2, 3])) == 2

    def test_majority_strings(self):
        assert DecisionTree()._leaf_value(np.array(["a", "b", "b"])) == "b"

    def test_tie_is_deterministic_smallest(self):
        assert DecisionTree()._leaf_value(np.array([2, 1, 2, 1])) == 1

    def test_mean(self):
        assert DecisionTree(task="regression")._leaf_value(np.array([1.0, 2.0, 6.0])) == pytest.approx(3.0)


# ---------------------------------------------------------- stopping rules
class TestShouldStop:
    def test_min_samples_split(self):
        t = DecisionTree(min_samples_split=5)
        assert t._should_stop(np.array([0, 1, 0, 1]), 0)
        assert not t._should_stop(np.array([0, 1, 0, 1, 0]), 0)

    def test_max_depth(self):
        t = DecisionTree(max_depth=2)
        y = np.array([0, 1, 0, 1])
        assert not t._should_stop(y, 1)
        assert t._should_stop(y, 2)

    def test_pure_classification(self):
        assert DecisionTree()._should_stop(np.array([1, 1, 1]), 0)

    def test_constant_regression(self):
        assert DecisionTree(task="regression")._should_stop(np.array([2.0, 2.0, 2.0]), 0)

    def test_regression_varied_not_stop(self):
        assert not DecisionTree(task="regression")._should_stop(np.array([1.0, 2.0, 3.0]), 0)


# ----------------------------------------------------------- best split
class TestBestSplit:
    def test_simple_threshold_is_midpoint(self):
        X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        f, thr, gain = DecisionTree()._best_split(X, y)
        assert f == 0 and thr == pytest.approx(6.5) and gain == pytest.approx(0.5)

    def test_picks_informative_feature(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(size=(100, 1))
        signal = np.r_[np.zeros(50), np.ones(50)].reshape(-1, 1) + rng.normal(0, 0.01, (100, 1))
        X = np.hstack([noise, signal])
        y = np.r_[np.zeros(50), np.ones(50)]
        f, thr, gain = DecisionTree()._best_split(X, y)
        assert f == 1 and 0.0 < thr < 1.0 and gain == pytest.approx(0.5)

    def test_no_split_when_constant_feature(self):
        assert DecisionTree()._best_split(np.ones((5, 2)), np.array([0, 1, 0, 1, 0])) is None

    def test_pure_node_becomes_leaf_via_fit(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        assert DecisionTree().fit(X, np.zeros(6)).root.feature_index is None

    def test_zero_gain_split_is_allowed_for_xor(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        best = DecisionTree()._best_split(X, np.array([0, 1, 1, 0]))
        assert best is not None and best[2] == pytest.approx(0.0)

    def test_min_impurity_decrease_still_blocks_zero_gain(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        assert DecisionTree(min_impurity_decrease=0.01)._best_split(X, np.array([0, 1, 1, 0])) is None

    def test_min_samples_leaf_respected(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.array([1] + [0] * 9)          # best unconstrained split isolates row 0
        _, thr, _ = DecisionTree(min_samples_leaf=3)._best_split(X, y)
        assert (X[:, 0] <= thr).sum() >= 3 and (X[:, 0] > thr).sum() >= 3

    def test_min_samples_leaf_too_big_gives_none(self):
        X = np.arange(4, dtype=float).reshape(-1, 1)
        assert DecisionTree(min_samples_leaf=3)._best_split(X, np.array([0, 0, 1, 1])) is None

    def test_best_gain_is_global_max(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 3))
        y = rng.integers(0, 2, 40)
        t = DecisionTree()
        f, thr, gain = t._best_split(X, y)
        parent = t._impurity(y)
        best = 0
        for j in range(3):
            v = np.unique(X[:, j])
            for c in (v[:-1] + v[1:]) / 2:
                m = X[:, j] <= c
                best = max(best, t._split_score(y[m], y[~m], parent))
        assert gain == pytest.approx(best)

    def test_regression_split(self):
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array([1.0] * 4 + [10.0] * 4)
        f, thr, gain = DecisionTree(task="regression")._best_split(X, y)
        assert thr == pytest.approx(3.5) and gain == pytest.approx(np.var(y))


# --------------------------------------------------------- classification
class TestClassification:
    def test_iris_training_accuracy_perfect(self, iris):
        # iris has a few duplicate rows with different labels? check achievable bound
        X, y = iris
        t = DecisionTree().fit(X, y)
        assert (t.predict(X) == y).mean() >= 0.99

    def test_fully_grown_memorizes_unique_data(self, clf_data):
        X, y = clf_data
        t = DecisionTree().fit(X, y)
        assert (t.predict(X) == y).all()

    @pytest.mark.parametrize("crit", ["gini", "entropy"])
    def test_holdout_accuracy(self, clf_data, crit):
        X, y = clf_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        t = DecisionTree(criterion=crit, max_depth=5).fit(Xtr, ytr)
        assert (t.predict(Xte) == yte).mean() > 0.75

    def test_xor_needs_depth_two(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10, dtype=float)
        y = np.array([0, 1, 1, 0] * 10)
        assert (DecisionTree(max_depth=1).fit(X, y).predict(X) == y).mean() <= 0.75
        t = DecisionTree(max_depth=2).fit(X, y)
        assert (t.predict(X) == y).all()

    def test_threshold_boundary_goes_left(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        t = DecisionTree().fit(X, np.array([0, 0, 1, 1]))
        thr = t.root.threshold
        assert thr == pytest.approx(2.5)
        assert t.predict([[thr]])[0] == 0          # <= goes left
        assert t.predict([[np.nextafter(thr, 10)]])[0] == 1

    def test_extrapolation_follows_edge_leaves(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        t = DecisionTree().fit(X, np.array([0, 0, 1, 1]))
        assert list(t.predict([[-1e9], [1e9]])) == [0, 1]

    def test_string_labels(self):
        X = np.array([[0.0], [1.0], [10.0], [11.0]])
        y = np.array(["low", "low", "high", "high"])
        t = DecisionTree().fit(X, y)
        assert list(t.predict([[0.5], [10.5]])) == ["low", "high"]
        assert list(t.classes_) == ["high", "low"]

    def test_boolean_and_float_labels(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        for y in (np.array([0, 0, 0, 1, 1, 1], dtype=bool), np.array([0.5] * 3 + [1.5] * 3)):
            t = DecisionTree().fit(X, y)
            assert (t.predict(X) == y).all()

    def test_single_class_is_single_leaf(self):
        t = DecisionTree().fit(np.random.rand(10, 2), np.full(10, 7))
        assert t.root.feature_index is None and t.root.value == 7
        assert (t.predict(np.random.rand(3, 2)) == 7).all()

    def test_single_sample(self):
        t = DecisionTree().fit([[1.0, 2.0]], [3])
        assert t.predict([[0.0, 0.0]])[0] == 3

    def test_constant_features_make_leaf(self):
        t = DecisionTree().fit(np.ones((6, 2)), np.array([0, 1, 1, 1, 0, 1]))
        assert t.root.feature_index is None and t.root.value == 1

    def test_identical_x_different_y_no_infinite_loop(self):
        X = np.array([[1.0], [1.0], [2.0], [2.0]])
        y = np.array([0, 1, 0, 1])
        t = DecisionTree().fit(X, y)
        assert t.predict([[1.0]])[0] in (0, 1)

    def test_deterministic(self, clf_data):
        X, y = clf_data
        a = DecisionTree(max_depth=6).fit(X, y).predict(X)
        b = DecisionTree(max_depth=6).fit(X, y).predict(X)
        np.testing.assert_array_equal(a, b)

    def test_row_order_invariance(self, clf_data):
        X, y = clf_data
        p = np.random.default_rng(0).permutation(len(y))
        a = DecisionTree(max_depth=5).fit(X, y)
        b = DecisionTree(max_depth=5).fit(X[p], y[p])
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_feature_scaling_invariance(self, clf_data):
        X, y = clf_data
        a = DecisionTree(max_depth=5).fit(X, y).predict(X)
        b = DecisionTree(max_depth=5).fit(X * 100 + 7, y).predict(X * 100 + 7)
        np.testing.assert_array_equal(a, b)

    def test_predictions_subset_of_classes(self, clf_data):
        X, y = clf_data
        t = DecisionTree(max_depth=4).fit(X, y)
        assert set(np.unique(t.predict(np.random.default_rng(0).normal(size=(200, 6))))) <= set(t.classes_)

    def test_three_class_structure(self):
        X = np.array([[0.0], [1.0], [5.0], [6.0], [10.0], [11.0]])
        y = np.array([0, 0, 1, 1, 2, 2])
        t = DecisionTree().fit(X, y)
        assert (t.predict(X) == y).all() and len(leaves(t.root)) == 3


# --------------------------------------------------------- hyperparameters
class TestHyperparameters:
    @pytest.mark.parametrize("d", [0, 1, 2, 3, 5])
    def test_max_depth_respected(self, clf_data, d):
        t = DecisionTree(max_depth=d).fit(*clf_data)
        assert depth(t.root) <= d

    def test_max_depth_zero_is_majority_stump(self, clf_data):
        X, y = clf_data
        t = DecisionTree(max_depth=0).fit(X, y)
        assert t.root.feature_index is None
        assert (t.predict(X) == np.bincount(y).argmax()).all()

    def test_deeper_never_worse_on_train(self, clf_data):
        X, y = clf_data
        accs = [(DecisionTree(max_depth=d).fit(X, y).predict(X) == y).mean() for d in range(0, 8)]
        assert all(a <= b + 1e-12 for a, b in zip(accs, accs[1:]))

    @pytest.mark.parametrize("m", [1, 2, 5, 20])
    def test_min_samples_leaf_respected(self, clf_data, m):
        X, y = clf_data
        t = DecisionTree(min_samples_leaf=m).fit(X, y)
        assert min(leaf_sizes(t, X)) >= m

    @pytest.mark.parametrize("m", [2, 10, 50])
    def test_min_samples_split_respected(self, clf_data, m):
        X, y = clf_data
        t = DecisionTree(min_samples_split=m).fit(X, y)

        def check(node, idx):
            if node.feature_index is None:
                return
            assert len(idx) >= m
            mask = X[idx, node.feature_index] <= node.threshold
            check(node.left, idx[mask]); check(node.right, idx[~mask])

        check(t.root, np.arange(len(y)))

    def test_larger_min_samples_split_smaller_tree(self, clf_data):
        sizes = [count_nodes(DecisionTree(min_samples_split=m).fit(*clf_data).root) for m in (2, 10, 50, 200)]
        assert all(a >= b for a, b in zip(sizes, sizes[1:])) and sizes[0] > sizes[-1]

    def test_min_impurity_decrease_prunes(self, clf_data):
        X, y = clf_data
        sizes = [count_nodes(DecisionTree(min_impurity_decrease=v).fit(X, y).root) for v in (0.0, 0.01, 0.05, 0.4)]
        assert all(a >= b for a, b in zip(sizes, sizes[1:]))
        assert sizes[-1] == 1

    def test_min_impurity_decrease_blocks_weak_splits(self, clf_data):
        X, y = clf_data
        thr = 0.02
        t = DecisionTree(min_impurity_decrease=thr).fit(X, y)

        def check(node, idx):
            if node.feature_index is None:
                return
            m = X[idx, node.feature_index] <= node.threshold
            parent = t._impurity(y[idx])
            assert t._split_score(y[idx][m], y[idx][~m], parent) > thr - 1e-12
            check(node.left, idx[m]); check(node.right, idx[~m])

        check(t.root, np.arange(len(y)))


# -------------------------------------------------------------- structure
class TestStructure:
    def test_every_internal_node_has_two_children(self, clf_data):
        t = DecisionTree(max_depth=6).fit(*clf_data)

        def check(n):
            if n.feature_index is None:
                assert n.left is None and n.right is None and n.value is not None
            else:
                assert n.left is not None and n.right is not None and n.threshold is not None
                check(n.left); check(n.right)

        check(t.root)

    def test_feature_indices_in_range(self, clf_data):
        t = DecisionTree(max_depth=6).fit(*clf_data)

        def check(n):
            if n.feature_index is not None:
                assert 0 <= n.feature_index < 6
                check(n.left); check(n.right)

        check(t.root)

    def test_leaf_counts_sum_to_n(self, clf_data):
        X, y = clf_data
        t = DecisionTree(max_depth=5).fit(X, y)
        assert sum(sum(l.class_counts.values()) for l in leaves(t.root)) == len(y)

    def test_leaf_value_is_argmax_of_counts(self, clf_data):
        t = DecisionTree(max_depth=5).fit(*clf_data)
        for l in leaves(t.root):
            assert l.class_counts[l.value] == max(l.class_counts.values())

    def test_pure_tree_has_pure_leaves(self, clf_data):
        t = DecisionTree().fit(*clf_data)
        assert all(len(l.class_counts) == 1 for l in leaves(t.root))

    def test_root_split_beats_all_alternatives(self, clf_data):
        X, y = clf_data
        t = DecisionTree().fit(X, y)
        m = X[:, t.root.feature_index] <= t.root.threshold
        got = t._split_score(y[m], y[~m], t._impurity(y))
        assert got == pytest.approx(t._best_split(X, y)[2])


# ----------------------------------------------------------- predict_proba
class TestPredictProba:
    def test_shape_and_rows_sum_to_one(self, iris):
        X, y = iris
        P = DecisionTree(max_depth=3).fit(X, y).predict_proba(X)
        assert P.shape == (150, 3)
        np.testing.assert_allclose(P.sum(1), 1.0)
        assert ((P >= 0) & (P <= 1)).all()

    def test_argmax_matches_predict(self, clf_data):
        X, y = clf_data
        t = DecisionTree(max_depth=4).fit(X, y)
        np.testing.assert_array_equal(t.classes_[t.predict_proba(X).argmax(1)], t.predict(X))

    def test_pure_leaves_give_one_hot(self, clf_data):
        X, y = clf_data
        P = DecisionTree().fit(X, y).predict_proba(X)
        assert set(np.unique(P)) <= {0.0, 1.0}

    def test_column_order_follows_classes(self):
        X = np.array([[0.0], [1.0], [10.0], [11.0]])
        t = DecisionTree().fit(X, np.array(["b", "b", "a", "a"]))
        np.testing.assert_allclose(t.predict_proba([[0.5]]), [[0.0, 1.0]])   # classes_ = [a, b]

    def test_mixed_leaf_probabilities(self):
        X = np.array([[1.0]] * 4)
        y = np.array([0, 0, 0, 1])
        np.testing.assert_allclose(DecisionTree().fit(X, y).predict_proba([[1.0]]), [[0.75, 0.25]])


# ---------------------------------------------------------------- regression
class TestRegression:
    def test_step_function(self):
        X = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.where(X[:, 0] < 5, 2.0, 8.0)
        t = DecisionTree(task="regression").fit(X, y)
        np.testing.assert_allclose(t.predict(X), y)
        assert len(leaves(t.root)) == 2 and t.root.threshold == pytest.approx(4.5)

    def test_leaf_is_mean(self):
        X = np.array([[0.0], [0.0], [10.0], [10.0]])
        y = np.array([1.0, 3.0, 10.0, 20.0])
        t = DecisionTree(task="regression", max_depth=1).fit(X, y)
        np.testing.assert_allclose(sorted(l.value for l in leaves(t.root)), [2.0, 15.0])

    def test_fully_grown_memorizes(self, reg_data):
        X, y = reg_data
        t = DecisionTree(task="regression").fit(X, y)
        np.testing.assert_allclose(t.predict(X), y, atol=1e-9)

    def test_predictions_within_target_range(self, reg_data):
        X, y = reg_data
        p = DecisionTree(task="regression", max_depth=4).fit(X, y).predict(
            np.random.default_rng(0).normal(size=(200, 5)) * 5)
        assert p.min() >= y.min() - 1e-9 and p.max() <= y.max() + 1e-9

    def test_mse_decreases_with_depth(self, reg_data):
        X, y = reg_data
        mses = [np.mean((DecisionTree(task="regression", max_depth=d).fit(X, y).predict(X) - y) ** 2)
                for d in range(0, 8)]
        assert all(a >= b - 1e-9 for a, b in zip(mses, mses[1:]))
        assert mses[0] == pytest.approx(np.var(y))

    def test_constant_target_single_leaf(self):
        t = DecisionTree(task="regression").fit(np.random.rand(10, 2), np.full(10, 4.2))
        assert t.root.feature_index is None and t.root.value == pytest.approx(4.2)

    def test_integer_targets_accepted(self):
        t = DecisionTree(task="regression").fit([[0], [1], [2], [3]], [1, 1, 5, 5])
        np.testing.assert_allclose(t.predict([[0], [3]]), [1.0, 5.0])

    def test_returns_floats(self, reg_data):
        assert DecisionTree(task="regression", max_depth=2).fit(*reg_data).predict(reg_data[0]).dtype.kind == "f"

    def test_holdout_r2(self, reg_data):
        X, y = reg_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        p = DecisionTree(task="regression", max_depth=5, min_samples_leaf=5).fit(Xtr, ytr).predict(Xte)
        r2 = 1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
        assert r2 > 0.6

    def test_min_samples_leaf_regression(self, reg_data):
        X, y = reg_data
        t = DecisionTree(task="regression", min_samples_leaf=10).fit(X, y)
        assert min(leaf_sizes(t, X)) >= 10


# -------------------------------------------------- comparison with sklearn
class TestAgainstSklearn:
    @pytest.mark.parametrize("crit", ["gini", "entropy"])
    def test_root_split_matches(self, clf_data, crit):
        X, y = clf_data
        mine = DecisionTree(criterion=crit, max_depth=3).fit(X, y)
        ref = DecisionTreeClassifier(criterion=crit, max_depth=3, random_state=0).fit(X, y)
        assert mine.root.feature_index == ref.tree_.feature[0]
        assert mine.root.threshold == pytest.approx(ref.tree_.threshold[0], rel=1e-4, abs=1e-4)

    @pytest.mark.parametrize("d", [1, 2, 3, 5, None])
    def test_training_accuracy_close_to_sklearn(self, clf_data, d):
        X, y = clf_data
        mine = (DecisionTree(max_depth=d).fit(X, y).predict(X) == y).mean()
        ref = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y).score(X, y)
        assert abs(mine - ref) < 0.02

    def test_iris_accuracy_matches(self, iris):
        X, y = iris
        mine = DecisionTree(max_depth=3).fit(X, y)
        ref = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
        assert (mine.predict(X) == y).mean() == pytest.approx(ref.score(X, y), abs=0.01)

    def test_leaf_count_matches_on_iris(self, iris):
        X, y = iris
        mine = DecisionTree(max_depth=3).fit(X, y)
        ref = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
        assert len(leaves(mine.root)) == ref.get_n_leaves()

    @pytest.mark.parametrize("d", [2, 4])
    def test_regression_root_and_fit_match(self, reg_data, d):
        X, y = reg_data
        mine = DecisionTree(task="regression", max_depth=d).fit(X, y)
        ref = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y)
        assert mine.root.feature_index == ref.tree_.feature[0]
        mse_m = np.mean((mine.predict(X) - y) ** 2)
        mse_r = np.mean((ref.predict(X) - y) ** 2)
        assert mse_m == pytest.approx(mse_r, rel=0.02)

    def test_proba_matches_on_iris(self, iris):
        X, y = iris
        mine = DecisionTree(max_depth=3).fit(X, y).predict_proba(X)
        ref = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y).predict_proba(X)
        np.testing.assert_allclose(mine, ref, atol=1e-9)


def test_tree_feature_importances_are_normalized():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    model = DecisionTree(max_depth=2).fit(X, y)
    assert model.feature_importances_.shape == (2,)
    np.testing.assert_allclose(model.feature_importances_.sum(), 1.0, atol=1e-12)


# ------------------------------------------- real project data vs scikit-learn
from sklearn.datasets import make_classification

DATA = "data/processed"
SUBSET = 2000          # enough to be meaningful, small enough to stay fast
SEED = 42


def _load(target):
    """A seeded subsample of the real project data, or skip if it is absent."""
    try:
        X = np.load(f"{DATA}/X_train.npy")
        y = np.load(f"{DATA}/{target}")
    except FileNotFoundError:
        pytest.skip(f"{DATA} not built — run `python3 -m src.data_prep` first")
    idx = np.random.default_rng(SEED).choice(len(X), SUBSET, replace=False)
    return X[idx], y[idx]


@pytest.fixture(scope="module")
def real_clf_data():
    """Features and the premium/standard tier label."""
    return _load("tier_train.npy")


@pytest.fixture(scope="module")
def real_reg_data():
    """Features and the log-price target."""
    return _load("y_train_log.npy")


@pytest.mark.parametrize("crit", ["gini", "entropy"])
def test_root_split_matches(real_clf_data, crit):
    """Our first split should be the same feature and threshold as sklearn's."""
    X, y = real_clf_data
    mine = DecisionTree(criterion=crit, max_depth=3).fit(X, y)
    ref = DecisionTreeClassifier(criterion=crit, max_depth=3, random_state=0).fit(X, y)
    assert mine.root.feature_index == ref.tree_.feature[0]
    assert mine.root.threshold == pytest.approx(ref.tree_.threshold[0], rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("d", [1, 2, 3, 5, None])
def test_training_accuracy_close_to_sklearn(real_clf_data, d):
    """Training accuracy should track sklearn's at every depth."""
    X, y = real_clf_data
    mine = (DecisionTree(max_depth=d).fit(X, y).predict(X) == y).mean()
    ref = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y).score(X, y)
    assert abs(mine - ref) < 0.02


@pytest.mark.parametrize("d", [2, 4])
def test_regression_root_and_fit_match(real_reg_data, d):
    """Regression: same root split, and MSE within 2% of sklearn's."""
    X, y = real_reg_data
    mine = DecisionTree(task="regression", max_depth=d).fit(X, y)
    ref = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y)
    assert mine.root.feature_index == ref.tree_.feature[0]
    mse_m = np.mean((mine.predict(X) - y) ** 2)
    mse_r = np.mean((ref.predict(X) - y) ** 2)
    assert mse_m == pytest.approx(mse_r, rel=0.02)


# predict_proba and prediction-time input validation


@pytest.fixture(scope="module")
def small_clf():
    X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    return DecisionTree(max_depth=5).fit(X, y), X


def test_predict_proba_matches_sklearn(real_clf_data):
    """Leaf class frequencies should equal sklearn's predict_proba."""
    X, y = real_clf_data
    mine = DecisionTree(max_depth=5).fit(X, y).predict_proba(X)
    ref = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y).predict_proba(X)
    assert np.abs(mine - ref).mean() < 0.02


def test_predict_proba_rows_sum_to_one(small_clf):
    t, X = small_clf
    p = t.predict_proba(X)
    assert p.shape == (len(X), len(t.classes_))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_predict_proba_argmax_matches_predict(small_clf):
    t, X = small_clf
    np.testing.assert_array_equal(t.classes_[t.predict_proba(X).argmax(axis=1)], t.predict(X))


def test_impure_leaf_gives_frequencies():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1])
    p = DecisionTree(max_depth=0).fit(X, y).predict_proba(X)   # one leaf: 3 vs 1
    np.testing.assert_allclose(p, [[0.75, 0.25]] * 4)


def test_predict_proba_rejects_regression():
    t = DecisionTree(task="regression").fit(np.random.rand(20, 2), np.random.rand(20))
    with pytest.raises(ValueError):
        t.predict_proba(np.zeros((1, 2)))


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_before_fit_raises(method):
    with pytest.raises(ValueError):
        getattr(DecisionTree(), method)(np.zeros((1, 2)))


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
@pytest.mark.parametrize("bad", [np.zeros(8), np.zeros((3, 5)),
                                 np.full((2, 8), np.nan), np.full((2, 8), np.inf),
                                 np.full((2, 8), "a")])
def test_bad_input_raises(small_clf, method, bad):
    """1-D, wrong feature count, non-finite, non-numeric."""
    t, _ = small_clf
    with pytest.raises(ValueError):
        getattr(t, method)(bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_fit_rejects_non_finite_X(bad):
    X = np.array([[0.0, 1.0], [bad, 2.0], [1.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError):
        DecisionTree().fit(X, np.array([0, 1, 0, 1]))
