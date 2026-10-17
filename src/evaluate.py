import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from src.ensemble import BaggedTrees
from src.decision_tree import DecisionTree
from src.svm import PegasosSVM


RANDOM_SEED = 42

# Bagging settings. Each member trains on a bootstrap sample rather than all
# 39,902 rows, which is what keeps this affordable with our tree. Measured on
# validation: 30 trees x 6,000 rows cuts regression MSE by about 19% against a
# single tree, for roughly 3-4 minutes of fitting per task.
N_ESTIMATORS = 30
BAG_MAX_SAMPLES = 6000
BAG_MIN_SAMPLES_LEAF = 2   # members may fit harder; averaging controls variance

# Members are grown DEEPER than the single tree, and deliberately not at the
# depth the sweep chose. The sweep answers "how deep before one tree starts
# overfitting?", which is the wrong question here: averaging already handles
# the variance, so each member should fit hard and let the ensemble cancel the
# noise. This is why random forests grow their trees out.
#
# Measured on validation (30 trees x 6,000 rows, classification):
#   members at the swept depth 8 -> acc 0.9145, AUC 0.9757, OOB 0.9165
#   members at depth 16          -> acc 0.9284, AUC 0.9808, OOB 0.9291
BAG_MAX_DEPTH = 16

# Depth grid for model selection. Coarse on purpose: each fit on the tuning
# subsample costs seconds, and the validation curve is flat enough near the
# optimum that a finer grid does not change the choice.
DEPTH_GRID = [2, 4, 6, 8, 10, 12, 14, 16, 20]
LAMBDA_GRID = [0.0001, 0.001, 0.01, 0.1]

# Leaves below this size are not worth splitting and the smoothing they give
# `predict_proba` is what makes ROC-AUC meaningful (pure leaves score 0/1 and
# rank badly). Matched into the sklearn baseline so the comparison is fair.
MIN_SAMPLES_LEAF = 5

# Depth selection runs on a seeded subsample because our DecisionTree is exact
# but slow; only the chosen depth is refit on the full training set.
TUNE_SUBSAMPLE = 6000
SVM_EPOCHS = 30


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MSE/RMSE/MAE/R2 on the log target, plus the human-readable error.

    The target is natural-log price, so a residual is a ratio: exp(MAE) - 1 is
    the typical miss as a percentage, which is the only one of these numbers a
    reader can interpret without knowing the units.
    """
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    mae = float(np.abs(err).mean())
    return {
        "mse": float((err ** 2).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": mae,
        "r2": float(1.0 - ss_res / ss_tot),
        "typical_error_pct": float((np.exp(mae) - 1.0) * 100.0),
    }

def classification_metrics(y_true, y_pred, scores=None) -> dict:
    out = {"acc": float((np.asarray(y_pred) == np.asarray(y_true)).mean())}
    if scores is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    return out


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out / name}")

# -----------------------------------------------------------------------------
# MODEL SELECTION (our models)
# -----------------------------------------------------------------------------

def tune_svm_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Pick the decision threshold maximising validation accuracy.

    Pegasos puts the boundary at 0, which maximises the hinge objective rather
    than accuracy. The ranking is good (high ROC-AUC) even when the default cut
    is poor, so moving the threshold is close to free accuracy.
    """
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    accs = [((scores >= c).astype(int) == y_true).mean() for c in candidates]
    return float(candidates[int(np.argmax(accs))])


def fig_bias_variance(sweep_r: dict, sweep_c: dict, out: Path) -> None:
    """Plot training and validation performance across decision-tree depths.

    Creates two subplots from the results of ``sweep_depth``:
    one for log-price regression and one for binary classification.
    Regression uses RMSE, where lower values are better; classification
    uses accuracy, where higher values are better. A vertical dashed line
    marks the best depth selected by the validation sweep.

    Args:
        sweep_r: Regression sweep results containing ``depths``, ``train``,
            ``val``, and ``best``.
        sweep_c: Classification sweep results containing the same keys.
        out: Directory in which to save the figure.

    Returns:
        None. Saves ``fig1_bias_variance.png`` and closes the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Bias–Variance — our DecisionTree",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(
        sweep_r["depths"], sweep_r["train"],
        "o-", label="Train RMSE",
    )
    ax.plot(
        sweep_r["depths"], sweep_r["val"],
        "s-", label="Validation RMSE",
    )
    ax.axvline(
        sweep_r["best"], linestyle="--",
        label=f"Best depth = {sweep_r['best']}",
    )
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("RMSE (log-price)")
    ax.set_title("Regression")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(
        sweep_c["depths"], sweep_c["train"],
        "o-", label="Train accuracy",
    )
    ax.plot(
        sweep_c["depths"], sweep_c["val"],
        "s-", label="Validation accuracy",
    )
    ax.axvline(
        sweep_c["best"], linestyle="--",
        label=f"Best depth = {sweep_c['best']}",
    )
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    save(fig, out, "fig1_bias_variance.png")


def fig_classification(
    t_true, tree_scores, svm_scores, out: Path
) -> None:
    """Plot ROC curves and a confusion matrix for binary classification.

    The left subplot compares the ROC curves of our DecisionTree and
    PegasosSVM using their respective positive-class probabilities or
    decision scores. Each curve includes its ROC-AUC value. The right
    subplot displays the DecisionTree confusion matrix using a threshold
    of 0.5 on its positive-class probabilities.

    Args:
        t_true: One-dimensional array of true binary labels encoded as 0
            and 1.
        tree_scores: Positive-class probabilities from our DecisionTree.
        svm_scores: Decision scores from our PegasosSVM. Larger values
            should indicate stronger evidence for class 1.
        out: Directory in which to save the figure.

    Returns:
        None. Saves ``fig3_classification.png`` and closes the figure.

    Notes:
        The confusion matrix uses a fixed 0.5 threshold for the tree.
        The SVM decision scores are used for ranking in ROC analysis,
        not as probabilities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Classification Quality",
        fontsize=13,
        fontweight="bold",
    )

    # ROC curves
    ax = axes[0]
    for name, scores in (
        ("Our DecisionTree", tree_scores),
        ("Our PegasosSVM", svm_scores),
    ):
        fpr, tpr, _ = roc_curve(t_true, scores)
        auc = roc_auc_score(t_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Confusion matrix for the decision tree
    pred = (np.asarray(tree_scores) >= 0.5).astype(int)
    true = np.asarray(t_true)

    cm = np.array([
        [int(((true == i) & (pred == j)).sum()) for j in (0, 1)]
        for i in (0, 1)
    ])

    ax = axes[1]
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                ),
            )

    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("DecisionTree confusion matrix")

    save(fig, out, "fig3_classification.png")


def fig_model_comparison(
    reg_bars: dict, clf_bars: dict, out: Path
) -> None:
    """Compare our models with scikit-learn baselines on validation data.

    Creates two horizontal bar charts. The regression chart compares
    validation RMSE for log-price prediction, where lower values are
    better. The classification chart compares validation accuracy,
    where higher values are better. Model names beginning with ``OUR``
    are highlighted to distinguish our implementations from baselines.

    Args:
        reg_bars: Mapping of regression model names to validation RMSE.
        clf_bars: Mapping of classification model names to validation
            accuracy.
        out: Directory in which to save the figure.

    Returns:
        None. Saves ``fig4_model_comparison.png`` and closes the figure.

    Notes:
        The dictionaries should contain already-computed validation
        metrics. This function only visualizes results; it does not
        train or evaluate models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Our Models vs Scikit-learn Baselines",
        fontsize=13,
        fontweight="bold",
    )

    # Regression comparison: lower RMSE is better
    ax = axes[0]
    names = list(reg_bars)
    values = [reg_bars[name] for name in names]
    colors = [
        "seagreen" if name.startswith("OUR") else "lightsteelblue"
        for name in names
    ]

    ax.barh(names, values, color=colors)
    for i, value in enumerate(values):
        ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)

    ax.set_xlabel("RMSE (log-price) — lower is better")
    ax.set_title("Regression")
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Classification comparison: higher accuracy is better
    ax = axes[1]
    names = list(clf_bars)
    values = [clf_bars[name] for name in names]
    colors = [
        "seagreen" if name.startswith("OUR") else "lightsteelblue"
        for name in names
    ]

    ax.barh(names, values, color=colors)
    for i, value in enumerate(values):
        ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)

    ax.set_xlabel("Accuracy — higher is better")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Classification")
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    save(fig, out, "fig4_model_comparison.png")



def fig_predictions(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    """Show how closely predicted log-prices track the real ones.

    Draws two panels. The left is a scatter of predicted against actual
    log-price with the perfect-prediction diagonal drawn on top, so any
    systematic bend away from the line is visible. The right is a
    histogram of the residuals, which should sit symmetrically around
    zero if the log transform has done its job.

    Args:
        y_true: Actual log-price values for the validation split.
        y_pred: Predicted log-price values for the same rows.
        out: Directory in which to save the figure.

    Returns:
        None. Saves ``fig2_predictions.png`` and closes the figure.

    Notes:
        Symmetry in the residual histogram is what justifies reporting
        ``exp(MAE) - 1`` as a percentage error: it means the model is
        equally likely to over- and under-predict by a given ratio.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Task A — Predicted vs Actual (our DecisionTree)",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    ax.scatter(y_true, y_pred, s=4, alpha=0.2, color="steelblue")
    lo, hi = y_true.min(), y_true.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect prediction")
    ax.set_xlabel("actual log(price)")
    ax.set_ylabel("predicted log(price)")
    ax.set_title("Validation set")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    resid = y_pred - y_true
    ax.hist(resid, bins=60, color="steelblue", edgecolor="white")
    ax.axvline(0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("residual (log units)")
    ax.set_ylabel("count")
    ax.set_title(f"Residuals — std = {resid.std():.3f}")
    ax.grid(alpha=0.3)

    save(fig, out, "fig2_predictions.png")


def fig_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    """Show where the model fails, in manat rather than log units.

    The left panel splits the validation set into price deciles and
    plots the median absolute percentage error of each, which exposes
    the bands where the model is weak. The right panel is the overall
    distribution of relative error.

    Args:
        y_true: Actual log-price values for the validation split.
        y_pred: Predicted log-price values for the same rows.
        out: Directory in which to save the figure.

    Returns:
        None. Saves ``fig5_error_analysis.png`` and closes the figure.

    Notes:
        Both inputs are log-prices, so ``exp(y_pred - y_true) - 1`` is the
        relative error directly. Reporting by decile matters because a
        single headline percentage hides that cheap flats and
        million-manat properties are predicted far worse than the bulk
        of the market.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Task A — Error Analysis", fontsize=13, fontweight="bold")

    price = np.exp(y_true)
    pct = (np.exp(y_pred - y_true) - 1.0) * 100.0

    ax = axes[0]
    edges = np.quantile(price, np.linspace(0, 1, 11))
    centres, medians = [], []
    for i in range(10):
        upper = price < edges[i + 1] if i < 9 else price <= edges[i + 1]
        band = (price >= edges[i]) & upper
        if band.sum():
            centres.append(np.median(price[band]))
            medians.append(np.median(np.abs(pct[band])))
    ax.plot(centres, medians, "o-", color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel("actual price (AZN, log scale)")
    ax.set_ylabel("median |error| (%)")
    ax.set_title("Error by price decile")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(np.clip(pct, -100, 200), bins=60, color="steelblue", edgecolor="white")
    ax.axvline(0, color="r", ls="--", lw=1.5)
    ax.set_xlabel("relative error (%)")
    ax.set_ylabel("count")
    ax.set_title(f"median |error| = {np.median(np.abs(pct)):.1f}%")
    ax.grid(alpha=0.3)

    save(fig, out, "fig5_error_analysis.png")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def run(processed_dir: str | Path, out_dir: str | Path, fast: bool = False, ensemble: bool = True) -> dict:
    out = Path(out_dir)
    data = load_data(processed_dir)
    n_tune = 1500 if fast else TUNE_SUBSAMPLE
    n_train = 4000 if fast else len(data["X_train"])
    depths = [2, 4, 6, 8] if fast else DEPTH_GRID
    epochs = 5 if fast else SVM_EPOCHS
    n_estimators = 5 if fast else N_ESTIMATORS
    bag_samples = 1000 if fast else BAG_MAX_SAMPLES

    print(f"\nevaluate: train={len(data['X_train']):,} rows x "
          f"{data['X_train'].shape[1]} features"
          f"{'   [FAST MODE — numbers are not final]' if fast else ''}")

    # --- 1. model selection on our models -------------------------------------
    print("\n[1/5] selecting hyper-parameters (our models) …")
    sweep_r = sweep_depth(data, "regression", depths, n_tune)
    sweep_c = sweep_depth(data, "classification", depths, n_tune)
    sweep_s = sweep_lambda(data, LAMBDA_GRID, epochs, n_tune)

    # --- 2. final fits -------------------------------------------------------
    print(f"\n[2/5] final fits on {n_train:,} training rows …")
    fit_idx = subsample(len(data["X_train"]), n_train)
    X_tr = data["X_train"][fit_idx]
    y_tr = data["y_train"][fit_idx]
    t_tr = data["tier_train"][fit_idx]
    timings = {}

    t0 = time.time()
    tree_r = DecisionTree(task="regression", max_depth=sweep_r["best"],
                          min_samples_leaf=MIN_SAMPLES_LEAF).fit(X_tr, y_tr)
    timings["our_tree_regression_s"] = time.time() - t0
    print(f"  our DecisionTree (regression)     {timings['our_tree_regression_s']:7.1f}s",
          flush=True)

    t0 = time.time()
    tree_c = DecisionTree(task="classification", max_depth=sweep_c["best"],
                          min_samples_leaf=MIN_SAMPLES_LEAF).fit(X_tr, t_tr)
    timings["our_tree_classification_s"] = time.time() - t0
    print(f"  our DecisionTree (classification) {timings['our_tree_classification_s']:7.1f}s",
          flush=True)

    t0 = time.time()
    svm = PegasosSVM(lambda_param=sweep_s["best"], epochs=epochs,
                     random_state=RANDOM_SEED).fit(X_tr, t_tr)
    timings["our_svm_s"] = time.time() - t0
    print(f"  our PegasosSVM                    {timings['our_svm_s']:7.1f}s", flush=True)

    # Pegasos optimises hinge loss, not accuracy; move the cut to where
    # validation accuracy peaks and report both.
    svm_val_scores = svm.decision_function(data["X_valid"])
    threshold = tune_svm_threshold(svm_val_scores, data["tier_valid"])

    # Bagging over our own tree. Members see bootstrap samples drawn from the
    # FULL training set, not from the subsample used for the single tree, so
    # the ensemble still learns from every row.
    bag_r = bag_c = None
    if ensemble:
        print(f"\n  bagging: {n_estimators} trees x {bag_samples:,} rows per task "
              f"(this is the slow part) …", flush=True)
        t0 = time.time()
        bag_depth = min(BAG_MAX_DEPTH, max(depths)) if fast else BAG_MAX_DEPTH
        bag_r = BaggedTrees(task="regression", n_estimators=n_estimators,
                            max_samples=bag_samples, max_depth=bag_depth,
                            min_samples_leaf=BAG_MIN_SAMPLES_LEAF,
                            random_state=RANDOM_SEED, oob_score=True).fit(
            data["X_train"], data["y_train"])
        timings["our_bagged_regression_s"] = time.time() - t0
        print(f"  BaggedTrees (regression)          "
              f"{timings['our_bagged_regression_s']:7.1f}s  "
              f"OOB R2={bag_r.oob_score_:.4f}", flush=True)

        t0 = time.time()
        bag_c = BaggedTrees(task="classification", n_estimators=n_estimators,
                            max_samples=bag_samples, max_depth=bag_depth,
                            min_samples_leaf=BAG_MIN_SAMPLES_LEAF,
                            random_state=RANDOM_SEED, oob_score=True).fit(
            data["X_train"], data["tier_train"])
        timings["our_bagged_classification_s"] = time.time() - t0
        print(f"  BaggedTrees (classification)      "
              f"{timings['our_bagged_classification_s']:7.1f}s  "
              f"OOB acc={bag_c.oob_score_:.4f}", flush=True)

    # --- 3. metrics on validation and test ----------------------------
    print("\n[3/5] scoring …")
    metrics: dict = {
        "fast_mode": fast,
        "random_seed": RANDOM_SEED,
        "n_train_used": int(n_train),
        "n_tune_subsample": int(n_tune),
        "best_depth_regression": sweep_r["best"],
        "best_depth_classification": sweep_c["best"],
        "best_lambda_svm": sweep_s["best"],
        "svm_threshold": threshold,
        "svm_n_support_vectors": int(svm.support_vector_mask_.sum()),
        "svm_margin_width": float(svm.margin_width_),
        "fit_seconds": timings,
        "ensemble": bool(ensemble),
    }
    if ensemble:
        metrics["n_estimators"] = int(n_estimators)
        metrics["bag_max_samples"] = int(bag_samples)
        metrics["bag_max_depth"] = int(bag_depth)
        metrics["oob_r2_regression"] = float(bag_r.oob_score_)
        metrics["oob_acc_classification"] = float(bag_c.oob_score_)

    for split in ("valid", "test"):
        X, y, t = data[f"X_{split}"], data[f"y_{split}"], data[f"tier_{split}"]
        metrics[f"our_tree_regression_{split}"] = regression_metrics(y, tree_r.predict(X))
        proba = tree_c.predict_proba(X)[:, 1]
        metrics[f"our_tree_classification_{split}"] = classification_metrics(
            t, tree_c.predict(X), proba)
        scores = svm.decision_function(X)
        m = classification_metrics(t, svm.predict(X), scores)
        m["acc_tuned_threshold"] = float(((scores >= threshold).astype(int) == t).mean())
        metrics[f"our_svm_{split}"] = m
        if ensemble:
            metrics[f"our_bagged_regression_{split}"] = regression_metrics(
                y, bag_r.predict(X))
            metrics[f"our_bagged_classification_{split}"] = classification_metrics(
                t, bag_c.predict(X), bag_c.predict_proba(X)[:, 1])

    # --- 4. sklearn baselines, hyper-parameters matched ------------------------
    print("\n[4/5] scikit-learn baselines (matched hyper-parameters) …")
    X_va, y_va, t_va = data["X_valid"], data["y_valid"], data["tier_valid"]

    sk_r = DecisionTreeRegressor(max_depth=sweep_r["best"],
                                 min_samples_leaf=MIN_SAMPLES_LEAF,
                                 random_state=RANDOM_SEED).fit(X_tr, y_tr)
    ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED).fit(X_tr, y_tr)
    dummy_r = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    metrics["sklearn_tree_regression_valid"] = regression_metrics(y_va, sk_r.predict(X_va))
    metrics["ridge_valid"] = regression_metrics(y_va, ridge.predict(X_va))
    metrics["dummy_regression_valid"] = regression_metrics(y_va, dummy_r.predict(X_va))

    sk_c = DecisionTreeClassifier(max_depth=sweep_c["best"],
                                  min_samples_leaf=MIN_SAMPLES_LEAF,
                                  random_state=RANDOM_SEED).fit(X_tr, t_tr)
    lsvc = LinearSVC(C=0.1, random_state=RANDOM_SEED).fit(X_tr, t_tr)
    dummy_c = DummyClassifier(strategy="stratified",
                              random_state=RANDOM_SEED).fit(X_tr, t_tr)
    metrics["sklearn_tree_classification_valid"] = classification_metrics(
        t_va, sk_c.predict(X_va), sk_c.predict_proba(X_va)[:, 1])
    metrics["linear_svc_valid"] = classification_metrics(
        t_va, lsvc.predict(X_va), lsvc.decision_function(X_va))
    metrics["dummy_classification_valid"] = classification_metrics(t_va, dummy_c.predict(X_va))

    # --- 5. figures + written metrics -----------------------------------------
    print("\n[5/5] figures …")
    fig_bias_variance(sweep_r, sweep_c, out)
    fig_predictions(y_va, tree_r.predict(X_va), out)
    fig_classification(t_va, tree_c.predict_proba(X_va)[:, 1], svm_val_scores, out)
    reg_bars = {}
    clf_bars = {}
    if ensemble:
        reg_bars["OUR BaggedTrees"] = metrics["our_bagged_regression_valid"]["rmse"]
        clf_bars["OUR BaggedTrees"] = metrics["our_bagged_classification_valid"]["acc"]
    reg_bars.update({
        "OUR DecisionTree": metrics["our_tree_regression_valid"]["rmse"],
        "sklearn tree": metrics["sklearn_tree_regression_valid"]["rmse"],
        "Ridge (a=1)": metrics["ridge_valid"]["rmse"],
        "Dummy (mean)": metrics["dummy_regression_valid"]["rmse"],
    })
    clf_bars.update({
        "OUR DecisionTree": metrics["our_tree_classification_valid"]["acc"],
        "OUR PegasosSVM": metrics["our_svm_valid"]["acc"],
        "sklearn tree": metrics["sklearn_tree_classification_valid"]["acc"],
        "LinearSVC (C=0.1)": metrics["linear_svc_valid"]["acc"],
        "Dummy (stratified)": metrics["dummy_classification_valid"]["acc"],
    })
    fig_model_comparison(reg_bars, clf_bars, out)
    fig_error_analysis(y_va, tree_r.predict(X_va), out)

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "metrics.md").write_text(render_markdown(metrics), encoding="utf-8")
    print(f"\n  wrote -> {out / 'metrics.json'}")
    print(f"  wrote -> {out / 'metrics.md'}")
    return metrics

def render_markdown(m: dict) -> str:
    a_v, a_t = m["our_tree_regression_valid"], m["our_tree_regression_test"]
    c_v, c_t = m["our_tree_classification_valid"], m["our_tree_classification_test"]
    s_v, s_t = m["our_svm_valid"], m["our_svm_test"]
    return (
        "# Model metrics — our NumPy implementations\n\n"
        f"- fast_mode: {m['fast_mode']} · seed {m['random_seed']} · "
        f"trained on {m['n_train_used']:,} rows\n"
        f"- chosen: tree depth {m['best_depth_regression']} (regression), "
        f"{m['best_depth_classification']} (classification), "
        f"SVM lambda={m['best_lambda_svm']}\n\n"
        "## Task A — regression (natural-log price)\n\n"
        "| split | MSE | RMSE | MAE | R2 | typical error |\n"
        "|---|---|---|---|---|---|\n"
        f"| valid | {a_v['mse']:.5f} | {a_v['rmse']:.4f} | {a_v['mae']:.4f} | "
        f"{a_v['r2']:.4f} | {a_v['typical_error_pct']:.1f}% |\n"
        f"| test | {a_t['mse']:.5f} | {a_t['rmse']:.4f} | {a_t['mae']:.4f} | "
        f"{a_t['r2']:.4f} | {a_t['typical_error_pct']:.1f}% |\n\n"
        "## Task B — classification (price tier)\n\n"
        "| model | split | accuracy | ROC-AUC |\n"
        "|---|---|---|---|\n"
        f"| our DecisionTree | valid | {c_v['acc']:.4f} | {c_v['roc_auc']:.4f} |\n"
        f"| our DecisionTree | test | {c_t['acc']:.4f} | {c_t['roc_auc']:.4f} |\n"
        f"| our PegasosSVM | valid | {s_v['acc']:.4f} | {s_v['roc_auc']:.4f} |\n"
        f"| our PegasosSVM | test | {s_t['acc']:.4f} | {s_t['roc_auc']:.4f} |\n\n"
        f"PegasosSVM at a tuned threshold ({m['svm_threshold']:.4f}): "
        f"valid acc {s_v['acc_tuned_threshold']:.4f}, "
        f"test acc {s_t['acc_tuned_threshold']:.4f}. "
        f"{m['svm_n_support_vectors']:,} support vectors, "
        f"margin width {m['svm_margin_width']:.4f}.\n\n"
        + (_render_ensemble(m) if m.get("ensemble") else "")
        + "## Baselines (validation, hyper-parameters matched to ours)\n\n"
        "| model | RMSE | accuracy |\n"
        "|---|---|---|\n"
        f"| sklearn tree | {m['sklearn_tree_regression_valid']['rmse']:.4f} | "
        f"{m['sklearn_tree_classification_valid']['acc']:.4f} |\n"
        f"| Ridge / LinearSVC | {m['ridge_valid']['rmse']:.4f} | "
        f"{m['linear_svc_valid']['acc']:.4f} |\n"
        f"| Dummy | {m['dummy_regression_valid']['rmse']:.4f} | "
        f"{m['dummy_classification_valid']['acc']:.4f} |\n"
    )

def load_data(processed_dir: str | Path) -> dict:
    """Load the arrays written by src.data_prep."""
    d = Path(processed_dir)
    missing = [
        f for f in ("X_train.npy", "y_train_log.npy", "tier_train.npy")
        if not (d / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{d} is missing {missing}. Run `python -m src.data_prep` first, "
            f"or just `python -m src.run_all`."
        )
    data = {}
    for split in ("train", "valid", "test"):
        data[f"X_{split}"] = np.load(d / f"X_{split}.npy")
        data[f"y_{split}"] = np.load(d / f"y_{split}_log.npy")
        data[f"tier_{split}"] = np.load(d / f"tier_{split}.npy")
    return data


def subsample(n: int, k: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """A fixed, seeded index subset; returns everything when k >= n."""
    if k >= n:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, k, replace=False)


def sweep_depth(data: dict, task: str, depths: list[int], n_tune: int) -> dict:
    """Fit our tree at each depth on a subsample; score on full validation.

    Training is subsampled because our tree is slow; validation is not, so the
    selected depth is chosen against the same validation set the final numbers
    are reported on.
    """
    X, y = data["X_train"], data[("y_train" if task == "regression" else "tier_train")]
    idx = subsample(len(X), n_tune)
    Xs, ys = X[idx], y[idx]
    X_va = data["X_valid"]
    y_va = data["y_valid" if task == "regression" else "tier_valid"]

    train_scores, val_scores = [], []
    print(f"  depth sweep ({task}) on {len(idx):,} rows:", flush=True)
    for d in depths:
        t0 = time.time()
        m = DecisionTree(task=task, max_depth=d,
                         min_samples_leaf=MIN_SAMPLES_LEAF).fit(Xs, ys)
        if task == "regression":
            tr = regression_metrics(ys, m.predict(Xs))["rmse"]
            va = regression_metrics(y_va, m.predict(X_va))["rmse"]
        else:
            tr = classification_metrics(ys, m.predict(Xs))["acc"]
            va = classification_metrics(y_va, m.predict(X_va))["acc"]
        train_scores.append(tr)
        val_scores.append(va)
        print(f"    depth {d:3d}: train={tr:.4f} val={va:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)

    best = depths[int(np.argmin(val_scores) if task == "regression"
                      else np.argmax(val_scores))]
    print(f"    -> best depth = {best}")
    return {"depths": depths, "train": train_scores, "val": val_scores, "best": best}

def sweep_lambda(data: dict, lambdas: list[float], epochs: int, n_tune: int) -> dict:
    """Pick the Pegasos regularisation strength on validation accuracy."""
    X, t = data["X_train"], data["tier_train"]
    idx = subsample(len(X), n_tune)
    accs = []
    print(f"  lambda sweep on {len(idx):,} rows:", flush=True)
    for lam in lambdas:
        m = PegasosSVM(lambda_param=lam, epochs=epochs,
                       random_state=RANDOM_SEED).fit(X[idx], t[idx])
        a = classification_metrics(data["tier_valid"], m.predict(data["X_valid"]))["acc"]
        accs.append(a)
        print(f"    lambda={lam:<8g} val acc={a:.4f}", flush=True)
    best = lambdas[int(np.argmax(accs))]
    print(f"    -> best lambda = {best}")
    return {"lambdas": lambdas, "val_acc": accs, "best": best}

def _render_ensemble(m: dict) -> str:
    """The bagging section, including the gain over a single tree."""
    b_v, b_t = m["our_bagged_regression_valid"], m["our_bagged_regression_test"]
    bc_v, bc_t = (m["our_bagged_classification_valid"],
                  m["our_bagged_classification_test"])
    single = m["our_tree_regression_valid"]["mse"]
    gain = (1.0 - b_v["mse"] / single) * 100.0
    return (
        f"## Bagging — {m['n_estimators']} of our trees, "
        f"{m['bag_max_samples']:,} bootstrap rows each, depth {m['bag_max_depth']}\n\n"
        "| task | split | headline |\n"
        "|---|---|---|\n"
        f"| A | valid | MSE={b_v['mse']:.5f} RMSE={b_v['rmse']:.4f} "
        f"R2={b_v['r2']:.4f} ({b_v['typical_error_pct']:.1f}%) |\n"
        f"| A | test | MSE={b_t['mse']:.5f} RMSE={b_t['rmse']:.4f} "
        f"R2={b_t['r2']:.4f} |\n"
        f"| B | valid | acc={bc_v['acc']:.4f} AUC={bc_v['roc_auc']:.4f} |\n"
        f"| B | test | acc={bc_t['acc']:.4f} AUC={bc_t['roc_auc']:.4f} |\n\n"
        f"Against one tree: MSE {single:.5f} -> {b_v['mse']:.5f} "
        f"(**{gain:+.1f}%**). Out-of-bag estimates, which use no validation "
        f"data at all: R2={m['oob_r2_regression']:.4f} (regression), "
        f"accuracy={m['oob_acc_classification']:.4f} (classification).\n\n"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="data/processed")
    p.add_argument("--out", default="reports/figures")
    p.add_argument("--fast", action="store_true",
                   help="subsample everything for a quick smoke test")
    p.add_argument("--no-ensemble", action="store_true",
                   help="skip bagging (the slowest step)")
    a = p.parse_args()
    run(a.dir, a.out, fast=a.fast, ensemble=not a.no_ensemble)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
