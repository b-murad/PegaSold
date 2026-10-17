"""Export the compact, real experiment summary consumed by the static website."""
from __future__ import annotations

import json
import re
from pathlib import Path


def build_site_data(results: dict, test_log: str = "") -> dict:
    """Keep only report-ready fields; omit long ROC coordinate arrays."""
    model_names = (
        "decision_tree_regression", "sklearn_tree_regression",
        "decision_tree_classification", "sklearn_tree_classification",
        "pegasos_svm", "linear_svc", "rff_pegasos_svm", "rbf_svc",
    )
    metrics = {}
    for name in model_names:
        model = results["test_results"][name]
        metrics[name] = {
            key: model.get(key) for key in (
                "accuracy", "precision", "recall", "f1", "roc_auc",
                "rmse", "mae", "r2", "rmse_AZN", "mae_AZN", "r2_AZN",
                "fit_seconds", "predict_seconds",
            )
        }
    audit = results["data_audit"]
    summary = {
        "seed": results["seed"],
        "raw_rows": audit["raw_rows"],
        "after_dedup": audit["after_dedup"],
        "after_scope": audit["after_scope"],
        "final_rows": audit["final_rows"],
        "ambiguous_repost_candidate_rows": audit["ambiguous_repost_candidate_rows"],
        "filters": audit["domain_filters"]["removed_by_rule"],
        "split": results["split"],
        "threshold_AZN": results["tier_threshold_AZN"],
        "feature_count": results["feature_count"],
        "numeric_feature_count": results["numeric_feature_count"],
        "metrics": metrics,
        "selected": results["selected"],
        "cross_validation": results["cross_validation"],
        "feature_importance": {
            task: items[:8] for task, items in results["feature_importance"].items()
        },
        "regression_error_slices": results["regression_error_slices"],
        "exploratory_analysis": results["exploratory_analysis"],
        "svm_diagnostics": {
            "margin_width": results["test_results"]["pegasos_svm"]["margin_width"],
            "support_vector_count": results["test_results"]["pegasos_svm"]["support_vector_count"],
            "rff_train_sample_count": results["svm_diagnostics"]["rff_train_sample_count"],
        },
        "test_summary": next(
            (match.group(0) for line in reversed(test_log.splitlines())
             if (match := re.search(r"\d+ passed(?:, \d+ skipped)?(?: in [\d.]+s)?", line))),
            "Test summary unavailable",
        ),
    }
    return summary


def write_site_data(results: dict, project_root: str | Path,
                    test_log: str = "") -> Path:
    """Write a local JS data file so the website also works without a backend."""
    output = Path(project_root) / "site-data.js"
    payload = json.dumps(build_site_data(results, test_log), ensure_ascii=False,
                         separators=(",", ":"))
    output.write_text(f"window.PEGASOLD_DATA = {payload};\n", encoding="utf-8")
    return output
