import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression

from decision_tree import DecisionTree
from tree_viz import export_text, plot_tree, to_dot


@pytest.fixture(scope="module")
def iris_tree():
    d = load_iris()
    return DecisionTree(max_depth=3).fit(d.data, d.target), d


def test_export_text_structure(iris_tree):
    t, d = iris_tree
    txt = export_text(t, d.feature_names, d.data, d.target)
    assert "petal length (cm) <= 2.450" in txt
    assert txt.count("class:") == 5                      # 5 leaves for this tree
    assert "samples=50" in txt and "counts=[50, 0, 0]" in txt


def test_export_text_default_names_and_no_data(iris_tree):
    t, _ = iris_tree
    txt = export_text(t)
    assert "x[" in txt and "petal" not in txt


def test_export_text_regression():
    X, y = make_regression(n_samples=50, n_features=3, random_state=0)
    t = DecisionTree(task="regression", max_depth=2).fit(X, y)
    assert "value:" in export_text(t, X=X, y=y)


def test_to_dot_valid_shape(iris_tree):
    t, d = iris_tree
    dot = to_dot(t, d.feature_names, list(d.target_names), d.data, d.target)
    assert dot.startswith("digraph Tree {") and dot.rstrip().endswith("}")
    assert dot.count("->") == 8                          # 9 nodes -> 8 edges
    assert "class = setosa" in dot and 'label="True"' in dot


def test_plot_tree_saves_png(iris_tree, tmp_path):
    t, d = iris_tree
    out = tmp_path / "tree.png"
    ax = plot_tree(t, d.feature_names, list(d.target_names), d.data, d.target, save_path=str(out))
    assert out.exists() and out.stat().st_size > 1000
    assert len(ax.texts) >= 9


def test_plot_tree_regression(tmp_path):
    X, y = make_regression(n_samples=60, n_features=3, random_state=0)
    t = DecisionTree(task="regression", max_depth=3).fit(X, y)
    plot_tree(t, X=X, y=y, save_path=str(tmp_path / "r.png"))


def test_single_leaf_tree_plots(tmp_path):
    t = DecisionTree().fit(np.random.rand(5, 2), np.zeros(5))
    plot_tree(t, save_path=str(tmp_path / "leaf.png"))
    assert "class:" in export_text(t)


def test_name_length_validation(iris_tree):
    t, _ = iris_tree
    with pytest.raises(ValueError):
        export_text(t, feature_names=["a"])
    with pytest.raises(ValueError):
        to_dot(t, class_names=["a"])


def test_unfitted_tree_rejected():
    with pytest.raises(ValueError):
        export_text(DecisionTree())
