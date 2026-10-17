"""NumPy metrics in src/metrics.py against hand-computed values."""
import numpy as np

from src.metrics import classification_summary, regression_summary


def test_numpy_metrics_match_known_binary_and_regression_values():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    metrics = classification_summary(y, pred, scores=np.array([.1, .8, .7, .9]))
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == .75
    assert np.isclose(metrics["precision"], 2 / 3)
    assert metrics["recall"] == 1
    assert np.isclose(metrics["f1"], .8)
    assert np.isclose(metrics["roc_auc"], .75)
    reg = regression_summary(np.array([1., 2., 3.]), np.array([1., 3., 2.]))
    assert np.isclose(reg["mae"], 2 / 3)
    assert np.isclose(reg["rmse"], np.sqrt(2 / 3))
    assert np.isclose(reg["r2"], .0)
