#   metrics + comparison helpers
"""Evaluation metrics (NumPy, from scratch) and model comparison utilities.

Works with any object that has fit(X, y) and predict(X): the from-scratch
DecisionTree, PegasosSVM, and sklearn baselines.

Main entry points
-----------------
classification_report_dict / regression_report_dict : metric dictionaries
compare_models      : fit + evaluate several models, return a results table
cross_validate      : k-fold CV for one model
svm_lambda_study    : sweep PegasosSVM lambda on a validation set
plot_*              : figures for the report
save_results        : write CSV / JSON tables
"""
import copy
import json
import os
import time

import numpy as np
import pandas as pd

try:  # support both `python -m src.run_all` and direct module use
    from .decision_tree import DecisionTree
except ImportError:  # pragma: no cover - direct execution compatibility
    from decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# Classification metrics:

def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, labels=None):
    """Rows - true classes, columns - predicted classes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {lab: i for i, lab in enumerate(labels.tolist())}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm, labels

def per_class_prf(y_true, y_pred, labels=None):
    """Per class precision, recall, F1-score and support (zero-division -> 0)"""
    cm, labels = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    pred_count = cm.sum(axis=0).astype(float)
    true_count = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return precision, recall, f1, true_count, labels

def roc_auc(y_true, scores, pos_label):
    """Binary ROC-AUC"""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos = y_true == pos_label
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    _, inv, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    mid_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = np.empty(len(scores))
    ranks[order] = mid_rank[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def roc_curve(y_true, scores, pos_label):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pos = (y_true == pos_label).astype(float)
    order = np.argsort(-scores, kind="mergesort")   # highest score first
    pos = pos[order]
    scores = scores[order]
    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, len(scores) - 1]
    tps = np.cumsum(pos)[idx]
    fps = 1 + idx - tps
    tpr = np.r_[0, tps / max(tps[-1], 1)]
    fpr = np.r_[0, fps / max(fps[-1], 1)]
    return fpr, tpr

def classification_report_dict(y_true, y_pred, scores=None, pos_label=None):
    """Accuracy, macro/weighted precision-recall-F1, optional ROC-AUC"""
    p, r, f1, support, labels = per_class_prf(y_true, y_pred)
    w = support / support.sum()
    report = {
        "accuracy": accuracy(y_true, y_pred),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float((f1 * w).sum()),
        "roc_auc": float("nan"),
    }
    if scores is not None and pos_label is not None and len(labels) == 2:
        report["roc_auc"] = roc_auc(y_true, scores, pos_label)
    return report

# Regression metrics:

def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nz = y_true != 0
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[nz] - y_pred[nz]) / np.abs(y_true[nz])) * 100)

def regression_report_dict(y_true, y_pred):
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }


# Summary dicts used by src/experiments.py. Named differently from the
# metrics in src/evaluate.py, which return other keys ("acc" vs "accuracy").
def regression_summary(y_true, y_pred):
    return regression_report_dict(y_true, y_pred)


def classification_summary(y_true, y_pred, scores=None, pos_label=1):
    """Binary metrics with positive-class fields and a labeled confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    base = classification_report_dict(y_true, y_pred, scores, pos_label)
    p, r, f, _, used_labels = per_class_prf(y_true, y_pred, labels)
    pos = np.flatnonzero(used_labels == pos_label)
    idx = int(pos[0]) if len(pos) else 0
    cm, _ = confusion_matrix(y_true, y_pred, labels=[0, 1])
    base.update({"precision": float(p[idx]), "recall": float(r[idx]),
                 "f1": float(f[idx]), "confusion_matrix": cm.tolist()})
    return base

# Evaluating a fitted model:

def _get_scores(model, X, classes):
    """Decision scores for the positive class if available"""
    if len(classes) == 2 and hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return scores if scores.ndim == 1 else None
    return None

def evaluate_model(model, X, y, task="classification", classes=None):
    """Evaluate an already fitted model"""
    y = np.asarray(y)
    y_pred = np.asarray(model.predict(X))
    if task == "regression":
        return regression_report_dict(y, y_pred), y_pred, None
    if classes is None:
        classes = np.unique(y)
    scores = _get_scores(model, X, classes)
    pos_label = classes[1] if len(classes) == 2 else None
    return classification_report_dict(y, y_pred, scores, pos_label), y_pred, scores

def compare_models(models, X_train, y_train, X_test, y_test, task="classification", out_dir=None):
    """Fit every model on the training data and evaluate on the test data.

    Args:
        models: dict name -> unfitted model (fit/predict). Models are deep-
            copied, so the originals are untouched.
        task: "classification" or "regression".
        out_dir: if given, results table/JSON are saved there.

    Returns:
        results: DataFrame (one row per model, metrics + timing)
        details: dict name -> {"model", "y_pred", "scores"} for plotting.
                 Models that failed to fit (e.g. the binary-only SVM on a
                 multi-class target) are listed with a "skipped" reason.
    """
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    classes = np.unique(y_train) if task == "classification" else None

    rows = []
    details = {}
    for name, proto in models.items():
        model = copy.deepcopy(proto)
        try:
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            metrics, y_pred, scores = evaluate_model(model, X_test, y_test, task, classes)
            predict_time = time.perf_counter() - t0
            if task == "classification":
                metrics["train_accuracy"] = accuracy(y_train, model.predict(X_train))
            else:
                metrics["train_r2"] = r2_score(y_train, model.predict(X_train))
        except ValueError as exc:
            # e.g. the binary-only SVM on a multi-class target
            rows.append({"model": name, "status": f"skipped: {exc}"})
            details[name] = {"skipped": str(exc)}
            continue
        rows.append({"model": name, "status": "ok", **metrics,
                     "fit_time_s": fit_time, "predict_time_s": predict_time})
        details[name] = {"model": model, "y_pred": y_pred, "scores": scores}

    results = pd.DataFrame(rows)
    if out_dir is not None:
        save_results(results, out_dir, name=f"comparison_{task}")
    return results, details

# Cross-validation:

def kfold_indices(y, k=5, seed=42, stratified=True):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    if stratified:
        for cls in np.unique(y):
            idx = rng.permutation(np.where(y == cls)[0])
            for i, j in enumerate(idx):
                folds[i % k].append(j)
    else:
        for i, j in enumerate(rng.permutation(len(y))):
            folds[i % k].append(j)
    for i in range(k):
        val = np.array(sorted(folds[i]))
        train = np.array(sorted(j for f in range(k) if f != i for j in folds[f]))
        yield train, val

def cross_validate(model, X, y, task="classification", k=5, seed=42):
    X = np.asarray(X)
    y = np.asarray(y)
    rows = []
    for fold, (tr, va) in enumerate(kfold_indices(y, k, seed, stratified=(task == "classification"))):
        m = copy.deepcopy(model).fit(X[tr], y[tr])
        classes = np.unique(y[tr]) if task == "classification" else None
        metrics, _, _ = evaluate_model(m, X[va], y[va], task, classes)
        rows.append({"fold": fold, **metrics})
    df = pd.DataFrame(rows)
    summary = pd.DataFrame({"fold": ["mean", "std"]})
    for col in df.columns[1:]:
        summary[col] = [df[col].mean(), df[col].std(ddof=1)]
    return pd.concat([df, summary], ignore_index=True)

# SVM lambda study:

def svm_lambda_study(svm_class, X_train, y_train, X_val, y_val, lambdas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), epochs=50, random_state=42):
    """Train PegasosSVM per lambda; report accuracy, objective, margin"""
    rows = []
    models = {}
    for lam in lambdas:
        svm = svm_class(lambda_param=lam, epochs=epochs, random_state=random_state)
        svm.fit(X_train, y_train)
        classes = np.unique(y_train)
        metrics, _, _ = evaluate_model(svm, X_val, y_val, "classification", classes)
        rows.append({
            "lambda": lam,
            "train_accuracy": accuracy(y_train, svm.predict(X_train)),
            "val_accuracy": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
            "val_roc_auc": metrics["roc_auc"],
            "final_objective": svm.objective_history_[-1],
            "n_support_vectors": int(svm.support_vector_mask_.sum()),
            "margin_width": svm.margin_width_,
        })
        models[lam] = svm
    return pd.DataFrame(rows), models

# Saving:

def save_results(results, out_dir, name="results"):
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(json.loads(results.to_json(orient="records")), f, indent=2)

# Plots:

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def _finish(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig

def plot_metric_comparison(results, metrics=("accuracy", "f1_macro", "roc_auc"), path=None, title="Model comparison"):
    """Grouped bar chart of selected metrics across models"""
    plt = _plt()
    df = results[results["status"] == "ok"]
    metrics = [m for m in metrics if m in df.columns and df[m].notna().any()]
    x = np.arange(len(df))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.6), 4))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m].fillna(0), width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, path)

plot_metrics_comparison = plot_metric_comparison   # alias for the other spelling

def plot_confusion_matrices(details, y_test, path=None, normalize=True):
    plt = _plt()
    ok = {k: v for k, v in details.items() if "y_pred" in v}
    labels = np.unique(y_test)
    fig, axes = plt.subplots(1, len(ok), figsize=(4 * len(ok), 3.8), squeeze=False)
    for ax, (name, d) in zip(axes[0], ok.items()):
        cm, _ = confusion_matrix(y_test, d["y_pred"], labels)
        shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
        ax.imshow(shown, cmap="Blues", vmin=0)
        ax.set_title(name)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if shown[i, j] > shown.max() / 2.0 else "black")
    fig.tight_layout()
    return _finish(fig, path)

def plot_roc_curves(details, y_test, pos_label, path=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, d in details.items():
        if d.get("scores") is None:
            continue
        fpr, tpr = roc_curve(y_test, d["scores"], pos_label)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc(y_test, d['scores'], pos_label):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    return _finish(fig, path)

def plot_objective(svm_models, path=None):
    plt = _plt()
    if not isinstance(svm_models, dict):
        svm_models = {"PegasosSVM": svm_models}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, m in svm_models.items():
        ax.plot(range(1, len(m.objective_history_) + 1), m.objective_history_, label=f"lambda={label}" if not isinstance(label, str) else label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective")
    ax.set_yscale("log")
    ax.set_title("SVM objective during training")
    ax.legend()
    return _finish(fig, path)

def plot_lambda_study(study, path=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(study["lambda"], study["train_accuracy"], "o-", label="train")
    ax.plot(study["lambda"], study["val_accuracy"], "s-", label="validation")
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Accuracy")
    ax.legend()
    return _finish(fig, path)

def plot_regression_fit(details, y_test, path=None):
    plt = _plt()
    ok = {k: v for k, v in details.items() if "y_pred" in v}
    fig, axes = plt.subplots(1, len(ok), figsize=(4 * len(ok), 4), squeeze=False)
    lo, hi = float(np.min(y_test)), float(np.max(y_test))
    for ax, (name, d) in zip(axes[0], ok.items()):
        ax.scatter(y_test, d["y_pred"], s=6, alpha=0.4)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    return _finish(fig, path)

# Default model sets:

def default_models(task="classification", binary=True, svm_params=None, tree_params=None):
    """Our from-scratch models plus sklearn baselines, ready for compare_models."""
    tree_params = {"max_depth": 8, **(tree_params or {})}
    if task == "regression":
        from sklearn.linear_model import Ridge
        return {
            "DecisionTree (ours)": DecisionTree(task="regression", **tree_params),
            "DecisionTree (sklearn)": DecisionTreeRegressor(max_depth=tree_params["max_depth"], random_state=42),
            "Ridge (sklearn)": Ridge(alpha=1.0),
        }

    models = {
        "DecisionTree gini (ours)": DecisionTree(task="classification", criterion="gini", **tree_params),
        "DecisionTree entropy (ours)": DecisionTree(task="classification", criterion="entropy", **tree_params),
        "DecisionTree (sklearn)": DecisionTreeClassifier(max_depth=tree_params["max_depth"], random_state=42),
    }
    if binary:   # PegasosSVM is binary only
        try:
            from .svm import PegasosSVM
        except ImportError:  # pragma: no cover
            from svm import PegasosSVM
        from sklearn.linear_model import SGDClassifier
        from sklearn.svm import LinearSVC
        params = {"lambda_param": 0.01, "epochs": 50, **(svm_params or {})}
        models["PegasosSVM (ours)"] = PegasosSVM(**params)
        models["SGDClassifier hinge (sklearn)"] = SGDClassifier(
            loss="hinge", alpha=params["lambda_param"], random_state=42)
        models["LinearSVC (sklearn)"] = LinearSVC(random_state=42)
    return models
