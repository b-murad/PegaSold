import json
from pathlib import Path

from src.site_data import build_site_data, write_site_data


def _results():
    metric_keys = (
        "accuracy", "precision", "recall", "f1", "roc_auc", "rmse", "mae",
        "r2", "rmse_AZN", "mae_AZN", "r2_AZN", "fit_seconds", "predict_seconds",
    )
    metrics = {name: {key: None for key in metric_keys} for name in (
        "decision_tree_regression", "sklearn_tree_regression",
        "decision_tree_classification", "sklearn_tree_classification",
        "pegasos_svm", "linear_svc", "rff_pegasos_svm", "rbf_svc",
    )}
    metrics["decision_tree_classification"].update(accuracy=.92, f1=.91, roc_auc=.96)
    metrics["pegasos_svm"].update(accuracy=.90, f1=.89, roc_auc=.97,
                                  margin_width=.78, support_vector_count=120)
    metrics["decision_tree_regression"].update(rmse=.23, mae=.15, r2=.87,
                                                rmse_AZN=126000, mae_AZN=47000, r2_AZN=.74)
    return {
        "seed": 42,
        "data_audit": {
            "raw_rows": 100, "after_dedup": 80, "after_scope": 75,
            "final_rows": 74, "ambiguous_repost_candidate_rows": 1,
            "domain_filters": {"removed_by_rule": {"price_outside_window": 1}},
        },
        "split": {"train": 50, "validation": 12, "test": 12},
        "tier_threshold_AZN": 200000,
        "feature_count": 8,
        "numeric_feature_count": 5,
        "test_results": metrics,
        "selected": {},
        "cross_validation": {
            task: {"sample_n": 60, "scratch": {"mean": .9, "std": .01},
                  "sklearn": {"mean": .89, "std": .02}}
            for task in ("decision_tree_regression", "decision_tree_classification", "pegasos_svm")
        },
        "feature_importance": {"regression": [{"feature": "area", "importance": .8}]},
        "regression_error_slices": [],
        "exploratory_analysis": {},
        "svm_diagnostics": {"rff_train_sample_count": 30},
    }


def test_site_summary_preserves_test_metrics_and_counts():
    data = build_site_data(_results(), "5 passed in 1.00s")
    assert data["final_rows"] == 74
    assert data["split"]["test"] == 12
    assert data["metrics"]["decision_tree_classification"]["accuracy"] == .92
    assert data["test_summary"] == "5 passed in 1.00s"
    assert "roc_curve" not in data["metrics"]["decision_tree_classification"]


def test_site_summary_falls_back_when_test_log_is_missing():
    assert build_site_data(_results())["test_summary"] == "Test summary unavailable"


def test_site_data_writes_browser_ready_javascript(tmp_path):
    output = write_site_data(_results(), tmp_path)
    contents = output.read_text(encoding="utf-8")
    assert output.name == "site-data.js"
    assert contents.startswith("window.PEGASOLD_DATA = {")
    payload = json.loads(contents.removeprefix("window.PEGASOLD_DATA = ").strip().removesuffix(";"))
    assert payload["threshold_AZN"] == 200000
