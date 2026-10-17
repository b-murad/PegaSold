"""Extra experiments ported from the decision-tree branch, run on OUR pipeline.

    python -m src.experiments          # after src.evaluate has written metrics.json

Adds what src/evaluate.py does not cover: the nonlinear random-Fourier-feature
Pegasos SVM and its RBF SVC baseline, 5-fold cross-validation against
scikit-learn, learning curves, regression error slices, tree feature
importances, a readable tree diagram, and the figures used by the web report
page. Hyper-parameters are read from reports/figures/metrics.json so these
numbers describe the same models as the main evaluation. Writes
reports/results.json and src/web/site-data.js.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier as SKTreeC
from sklearn.tree import DecisionTreeRegressor as SKTreeR

from src.data_prep import RANDOM_SEED as SEED
from src.decision_tree import DecisionTree
from src.evaluate import MIN_SAMPLES_LEAF
from src.metrics import classification_summary, regression_summary
from src.site_data import write_site_data
from src.svm import PegasosSVM, RandomFourierPegasosSVM
from src.tree_viz import plot_tree

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
WEB = ROOT / "src" / "web"


# ------------------------------------------------------------------ data
def load(processed: Path = PROCESSED) -> dict:
    """Everything the experiments need, in the shape the branch code expects."""
    rep = json.loads((processed / "preprocessing_report.json").read_text(encoding="utf-8"))
    d = {"feature_names": pd.read_csv(processed / "feature_names.csv")["feature"].tolist(),
         "numeric_feature_count": len(rep["numeric_features"])}
    for ours, theirs in (("train", "train"), ("valid", "validation"), ("test", "test")):
        d[f"X_{theirs}"] = np.load(processed / f"X_{ours}.npy")
        d[f"y_{theirs}_log"] = np.load(processed / f"y_{ours}_log.npy")
        d[f"tier_{theirs}"] = np.load(processed / f"tier_{ours}.npy")
        meta = pd.read_csv(processed / f"{ours}_clean.csv", low_memory=False)
        d[f"metadata_{theirs}"] = meta
        d[f"price_{theirs}"] = meta["price"].to_numpy(dtype=float)
    d["area_train_m2"] = d["metadata_train"]["area_m2"].to_numpy(dtype=float)
    d["rooms_train"] = d["metadata_train"]["rooms"].to_numpy(dtype=float)
    d["missing_feature_percent"] = {
        c: float(d["metadata_train"][c].isna().mean() * 100) for c in ("rooms", "lat", "lng")}
    d["tier_threshold"] = float(rep["tier_threshold_AZN"])
    d["audit"] = {"raw_rows": rep["raw_rows"], "after_dedup": rep["rows_after_deduplication"],
                  "after_scope": rep["scope"]["rows_after_scope"], "final_rows": rep["final_rows"],
                  "ambiguous_repost_candidate_rows": 0,   # our dedup removes reposts; none are kept as ambiguous
                  "domain_filters": rep["domain_filters"],
                  "leakage_columns_excluded": rep["leakage_columns_removed"]}
    return d


# --------------------------------------------- helpers (decision-tree branch)
def _tree(task, criterion, depth, leaf):
    # Same settings as the final trees in src/evaluate.py.
    return DecisionTree(task=task, criterion=criterion, max_depth=depth, min_samples_leaf=leaf)


def _sk_tree(task, criterion, depth, leaf):
    if task == "regression":
        return SKTreeR(criterion="squared_error", max_depth=depth,
                       min_samples_leaf=leaf, random_state=SEED)
    return SKTreeC(criterion=criterion or "gini", max_depth=depth,
                   min_samples_leaf=leaf, random_state=SEED)


def _fit_time(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def _cv_tree(X, y, task, criterion, depth, leaf, seed=SEED):
    # Use every available training row for the uncertainty estimate. The
    # separately held validation set still selects hyperparameters; test is
    # never included in CV.
    n = len(y)
    xx, yy = np.asarray(X), np.asarray(y)
    splitter = (StratifiedKFold(5, shuffle=True, random_state=seed) if task == "classification"
                else KFold(5, shuffle=True, random_state=seed))
    ours, reference = [], []
    ours_mae_azn, ref_mae_azn = [], []
    ours_rmse_azn, ref_rmse_azn = [], []
    ours_r2_azn, ref_r2_azn = [], []
    for tr, va in splitter.split(xx, yy):
        m = _tree(task, criterion, depth, leaf).fit(xx[tr], yy[tr])
        sk = _sk_tree(task, criterion, depth, leaf).fit(xx[tr], yy[tr])
        po, ps = m.predict(xx[va]), sk.predict(xx[va])
        if task == "classification":
            ours.append(float(accuracy_score(yy[va], po)))
            reference.append(float(accuracy_score(yy[va], ps)))
        else:
            ours.append(float(np.sqrt(mean_squared_error(yy[va], po))))
            reference.append(float(np.sqrt(mean_squared_error(yy[va], ps))))
            actual_azn = np.exp(yy[va])
            ours_azn = np.exp(np.clip(po, 0, 25))
            ref_azn = np.exp(np.clip(ps, 0, 25))
            ours_metrics = regression_summary(actual_azn, ours_azn)
            ref_metrics = regression_summary(actual_azn, ref_azn)
            ours_mae_azn.append(ours_metrics["mae"])
            ref_mae_azn.append(ref_metrics["mae"])
            ours_rmse_azn.append(ours_metrics["rmse"])
            ref_rmse_azn.append(ref_metrics["rmse"])
            ours_r2_azn.append(ours_metrics["r2"])
            ref_r2_azn.append(ref_metrics["r2"])
    metric = "accuracy" if task == "classification" else "rmse_log"
    result = {"metric": metric, "sample_n": n, "fold_count": 5,
              "scratch": {"fold_scores": ours, "mean": float(np.mean(ours)), "std": float(np.std(ours))},
              "sklearn": {"fold_scores": reference, "mean": float(np.mean(reference)), "std": float(np.std(reference))}}
    if task == "regression":
        result["azn_metrics"] = {
            "scratch_mae": {"fold_scores": ours_mae_azn, "mean": float(np.mean(ours_mae_azn)), "std": float(np.std(ours_mae_azn))},
            "sklearn_mae": {"fold_scores": ref_mae_azn, "mean": float(np.mean(ref_mae_azn)), "std": float(np.std(ref_mae_azn))},
            "scratch_rmse": {"fold_scores": ours_rmse_azn, "mean": float(np.mean(ours_rmse_azn)), "std": float(np.std(ours_rmse_azn))},
            "sklearn_rmse": {"fold_scores": ref_rmse_azn, "mean": float(np.mean(ref_rmse_azn)), "std": float(np.std(ref_rmse_azn))},
            "scratch_r2": {"fold_scores": ours_r2_azn, "mean": float(np.mean(ours_r2_azn)), "std": float(np.std(ours_r2_azn))},
            "sklearn_r2": {"fold_scores": ref_r2_azn, "mean": float(np.mean(ref_r2_azn)), "std": float(np.std(ref_r2_azn))},
        }
    return result


def _cv_svm(X, y, lam, epochs, seed=SEED):
    # Use the full training partition; validation/test remain separate.
    xx, yy = np.asarray(X), np.asarray(y)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    ours, refs = [], []
    for tr, va in folds.split(xx, yy):
        model = PegasosSVM(lambda_param=lam, epochs=epochs, random_state=seed).fit(xx[tr], yy[tr])
        svc = LinearSVC(loss="hinge", C=1/(lam*len(tr)), dual="auto",
                        max_iter=10000, tol=1e-3, random_state=seed).fit(xx[tr], yy[tr])
        ours.append(float(accuracy_score(yy[va], model.predict(xx[va]))))
        refs.append(float(accuracy_score(yy[va], svc.predict(xx[va]))))
    return {"metric": "accuracy", "sample_n": len(yy), "fold_count": 5, "epochs": epochs,
            "scratch": {"fold_scores": ours, "mean": float(np.mean(ours)), "std": float(np.std(ours))},
            "sklearn": {"fold_scores": refs, "mean": float(np.mean(refs)), "std": float(np.std(refs))}}


def _learning_curves(d, best_reg, best_cls):
    rng = np.random.default_rng(SEED + 7)
    sizes = sorted(set(min(n, len(d["X_train"])) for n in (1000, 4000, 12000, 24000, len(d["X_train"]))))
    curves = {"regression": [], "classification": []}
    for size in sizes:
        ix = rng.choice(len(d["X_train"]), size, replace=False)
        for task, best, target_key, valid_key, metric in (
            ("regression", best_reg, "y_train_log", "y_validation_log", "rmse"),
            ("classification", best_cls, "tier_train", "tier_validation", "accuracy")):
            m = _tree(task, best["criterion"], best["max_depth"], best["min_samples_leaf"]).fit(d["X_train"][ix], d[target_key][ix])
            pt, pv = m.predict(d["X_train"][ix]), m.predict(d["X_validation"])
            if task == "regression":
                train_score = float(np.sqrt(mean_squared_error(d[target_key][ix], pt)))
                val_score = float(np.sqrt(mean_squared_error(d[valid_key], pv)))
            else:
                train_score = float(accuracy_score(d[target_key][ix], pt))
                val_score = float(accuracy_score(d[valid_key], pv))
            curves[task].append({"n": size, "train": train_score, "validation": val_score, "metric": metric})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task, title, ylabel in ((axes[0], "regression", "Tree learning curve — regression", "RMSE in log-price space"),
                                    (axes[1], "classification", "Tree learning curve — classification", "Accuracy")):
        c = curves[task]; ax.plot([x["n"] for x in c], [x["train"] for x in c], "o-", label="training")
        ax.plot([x["n"] for x in c], [x["validation"] for x in c], "o-", label="validation")
        ax.set(title=title, xlabel="Training rows", ylabel=ylabel); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "tree_learning_curves.png", dpi=160); plt.close(fig)
    return curves


def _with_roc(y, pred, scores):
    from sklearn.metrics import roc_curve
    out = classification_summary(y, pred, scores)
    fpr, tpr, _ = roc_curve(y, scores)
    step = max(1, len(fpr) // 200)
    indices = list(range(0, len(fpr), step))
    if indices[-1] != len(fpr) - 1:
        indices.append(len(fpr) - 1)
    out["roc_curve"] = {"fpr": fpr[indices].tolist(), "tpr": tpr[indices].tolist()}
    return out


def _make_figures(d, results, final_models):
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Price distribution and cleaning attrition.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(d["price_train"], bins=60, color="#3977a8"); ax[0].set(xscale="log", title="Sale prices, AZN (log axis)", xlabel="Price (AZN)", ylabel="Listings")
    stages = results["data_audit"]; labels = ["Raw", "Unique listing", "Residential", "Valid"]
    vals = [stages["raw_rows"], stages["after_dedup"], stages["after_scope"], stages["final_rows"]]
    ax[1].bar(labels, vals, color=["#8fa4b5", "#3977a8", "#7c9c59", "#bd4f3a"]); ax[1].set(title="Data cleaning attrition", ylabel="Rows")
    fig.tight_layout(); fig.savefig(FIGURES / "data_and_prices.png", dpi=160); plt.close(fig)
    # Measured values vs target with property category colors.
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(d["price_train"]), min(3500, len(d["price_train"])), replace=False)
    categories = d["metadata_train"].iloc[take]["category"].astype("category").cat.codes.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5)); scat = ax.scatter(d["area_train_m2"][take], d["price_train"][take], c=categories, s=10, alpha=.45, cmap="viridis")
    ax.set(xscale="log", yscale="log", xlabel="Area (m², log scale)", ylabel="Sale price (AZN, log scale)", title="Price vs area, colored by property category"); fig.colorbar(scat, ax=ax, label="Category code"); fig.tight_layout(); fig.savefig(FIGURES / "price_vs_area.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    balance_groups = [d["tier_train"], d["tier_validation"], d["tier_test"]]
    standard = [float(np.mean(y == 0)) for y in balance_groups]
    premium = [float(np.mean(y == 1)) for y in balance_groups]
    ax.bar(["Train", "Validation", "Test"], standard, label="Standard", color="#3977a8")
    ax.bar(["Train", "Validation", "Test"], premium, bottom=standard, label="Premium", color="#bd4f3a")
    ax.set(ylim=(0, 1), ylabel="Share", title="Price-tier balance by split"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "class_balance.png", dpi=160); plt.close(fig)
    missing = d["missing_feature_percent"]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(list(missing), list(missing.values()), color="#7c9c59")
    ax.set(xlabel="Missing before imputation (%)", title="Numeric feature missingness"); fig.tight_layout(); fig.savefig(FIGURES / "missingness.png", dpi=160); plt.close(fig)
    room_values = d["rooms_train"]
    room_numbers = sorted(set(int(x) for x in room_values if 1 <= x <= 8))
    keep = [(n, d["price_train"][room_values == n]) for n in room_numbers]
    keep = [(n, values) for n, values in keep if len(values)]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.boxplot([x[1] for x in keep], tick_labels=[str(x[0]) for x in keep], showfliers=False)
    ax.set(yscale="log", xlabel="Number of rooms", ylabel="Sale price (AZN, log scale)", title="Training price by room count"); fig.tight_layout(); fig.savefig(FIGURES / "price_vs_rooms.png", dpi=160); plt.close(fig)
    location_values = d["metadata_train"]["location"].astype(str)
    locations = []
    for name, positions in location_values.groupby(location_values).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        if len(positions) >= 100:
            locations.append((name, len(positions), float(np.median(d["price_train"][positions]))))
    locations = sorted(locations, key=lambda x: x[2], reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh([x[0] for x in locations][::-1], [x[2] for x in locations][::-1], color="#bd4f3a")
    ax.set(xlabel="Median advertised price (AZN)", title="Highest-median training locations (n ≥ 100)"); fig.tight_layout(); fig.savefig(FIGURES / "location_prices.png", dpi=160); plt.close(fig)
    # Best tree/SVM confusion matrices.
    for key, title, filename in (("decision_tree_classification", "Scratch decision tree", "tree_confusion_matrix.png"),
                                 ("pegasos_svm", "Scratch Pegasos SVM", "pegasos_confusion_matrix.png"),
                                 ("rff_pegasos_svm", "Random Fourier Pegasos", "rff_confusion_matrix.png")):
        cm = results["test_results"][key]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4.4, 3.8)); ax.imshow(cm, cmap="Blues"); ax.set(title=title + " — test", xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1], xticklabels=["Standard", "Premium"], yticklabels=["Standard", "Premium"])
        for (i, j), value in np.ndenumerate(cm): ax.text(j, i, f"{value:,}", ha="center", va="center")
        fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=160); plt.close(fig)
    # Test ROC comparisons.
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ("decision_tree_classification", "sklearn_tree_classification", "pegasos_svm", "linear_svc", "rff_pegasos_svm", "rbf_svc"):
        roc = results["test_results"][name].get("roc_curve")
        if roc:
            ax.plot(roc["fpr"], roc["tpr"], label=f"{name} AUC={results['test_results'][name]['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Price-tier ROC curves on the held-out test set"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "roc_curves.png", dpi=160); plt.close(fig)
    # Test accuracy and F1, matched two tasks where applicable.
    names = ["Scratch tree", "sklearn tree", "Pegasos", "Linear SVC", "RFF Pegasos", "RBF SVC"]
    keys = ["decision_tree_classification", "sklearn_tree_classification", "pegasos_svm", "linear_svc", "rff_pegasos_svm", "rbf_svc"]
    xx = np.arange(len(keys)); width = .36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xx-width/2, [results["test_results"][k]["accuracy"] for k in keys], width, label="Accuracy")
    ax.bar(xx+width/2, [results["test_results"][k]["f1"] for k in keys], width, label="F1")
    ax.set(ylim=(0, 1), ylabel="Score", title="Classification test scores"); ax.set_xticks(xx, names, rotation=20, ha="right"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "classification_comparison.png", dpi=160); plt.close(fig)
    # Pegasos objective history and margin score distribution.
    svm = final_models["pegasos_svm"]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(np.arange(1, len(svm.objective_history_)+1), svm.objective_history_, "o-", ms=3)
    ax.set(xlabel="Epoch", ylabel="Regularized hinge objective", title="Pegasos objective by epoch"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "pegasos_objective.png", dpi=160); plt.close(fig)
    scores = svm.decision_function(final_models["X_test_scaled"]); labels = d["tier_test"]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(scores[labels == 0], bins=45, alpha=.65, label="Standard"); ax.hist(scores[labels == 1], bins=45, alpha=.65, label="Premium"); ax.axvline(0, color="black", ls="--"); ax.set(title="Pegasos decision scores and margin boundary", xlabel="w·x + b", ylabel="Listings"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "svm_margin_scores.png", dpi=160); plt.close(fig)
    # Error diagnosis for the final regression model.
    pred_log = final_models["tree_regression"].predict(d["X_test"])
    true_price = d["price_test"]; pred_price = np.exp(np.clip(pred_log, 0, 25))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    take = rng.choice(len(true_price), min(3000, len(true_price)), replace=False)
    ax[0].scatter(true_price[take], pred_price[take], s=9, alpha=.35); lo, hi = np.quantile(true_price[take], [.01, .99]); ax[0].plot([lo, hi], [lo, hi], "r--"); ax[0].set(xscale="log", yscale="log", xlabel="Actual price (AZN)", ylabel="Predicted price (AZN)", title="Tree price predictions")
    err = np.log(true_price) - pred_log; ax[1].scatter(pred_log[take], err[take], s=9, alpha=.35); ax[1].axhline(0, color="red", ls="--"); ax[1].set(xlabel="Predicted ln(price)", ylabel="Actual − predicted ln(price)", title="Regression residuals")
    fig.tight_layout(); fig.savefig(FIGURES / "regression_error_analysis.png", dpi=160); plt.close(fig)
    # Feature importance and readable top-three-level tree.
    for key, model in (("regression", final_models["tree_regression"]), ("classification", final_models["tree_classification"])):
        imp = model.feature_importances_; ix = np.argsort(imp)[-15:]
        fig, ax = plt.subplots(figsize=(7.5, 5)); ax.barh([d["feature_names"][i] for i in ix], imp[ix], color="#3977a8"); ax.set(title=f"Decision-tree {key} feature importance", xlabel="Normalized weighted impurity decrease"); fig.tight_layout(); fig.savefig(FIGURES / f"feature_importance_{key}.png", dpi=160); plt.close(fig)
    small = _tree("classification", "gini", 3, 30).fit(d["X_train"], d["tier_train"])
    plot_tree(small, feature_names=d["feature_names"], X=d["X_train"], y=d["tier_train"], save_path=str(FIGURES / "decision_tree.png"), fontsize=7)


# ------------------------------------------------------------------- run
def run(test_log: str = "") -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((FIGURES / "metrics.json").read_text(encoding="utf-8"))
    d = load()
    Xtr, Xv, Xtest = d["X_train"], d["X_validation"], d["X_test"]
    lam = float(metrics["best_lambda_svm"])
    svm_epochs = 30   # SVM_EPOCHS in src/evaluate.py
    best_reg = {"criterion": "mse", "max_depth": metrics["best_depth_regression"], "min_samples_leaf": MIN_SAMPLES_LEAF}
    best_cls = {"criterion": "gini", "max_depth": metrics["best_depth_classification"], "min_samples_leaf": MIN_SAMPLES_LEAF}
    best_linear = {"lambda": lam, "epochs": svm_epochs}

    # Statement's nonlinear choice: RBF random Fourier features + our same Pegasos.
    rng = np.random.default_rng(SEED)
    rff_idx = rng.choice(len(Xtr), min(8000, len(Xtr)), replace=False)
    rff_scaler = StandardScaler().fit(Xtr[rff_idx])
    Xrff_train, Xrff_valid, Xrff_test = (rff_scaler.transform(x) for x in (Xtr[rff_idx], Xv, Xtest))
    rff_candidates, rff_curve = [], []
    for gamma in (.01, .1, 1.0):
        model, fit_s = _fit_time(RandomFourierPegasosSVM(lambda_param=lam, epochs=10,
                                    n_components=128, gamma=gamma, random_state=SEED), Xrff_train, d["tier_train"][rff_idx])
        pred, scores = model.predict(Xrff_valid), model.decision_function(Xrff_valid)
        met = classification_summary(d["tier_validation"], pred, scores)
        rec = {"gamma": gamma, "n_components": 128, "epochs": 10, "lambda": lam,
               "validation_accuracy": met["accuracy"], "validation_f1": met["f1"], "validation_roc_auc": met["roc_auc"], "fit_seconds": fit_s, "tuning_train_n": len(rff_idx)}
        rff_curve.append(rec); rff_candidates.append((met["f1"], rec))
    best_rff = max(rff_candidates, key=lambda item: item[0])[1]
    print(f"  RFF sweep: best gamma={best_rff['gamma']}", flush=True)

    # Final model refits on train only; test data is used once from this point on.
    reg, reg_fit = _fit_time(_tree("regression", "mse", best_reg["max_depth"], best_reg["min_samples_leaf"]), Xtr, d["y_train_log"])
    cls, cls_fit = _fit_time(_tree("classification", "gini", best_cls["max_depth"], best_cls["min_samples_leaf"]), Xtr, d["tier_train"])
    sk_reg, sk_reg_fit = _fit_time(_sk_tree("regression", "mse", best_reg["max_depth"], best_reg["min_samples_leaf"]), Xtr, d["y_train_log"])
    sk_cls, sk_cls_fit = _fit_time(_sk_tree("classification", "gini", best_cls["max_depth"], best_cls["min_samples_leaf"]), Xtr, d["tier_train"])
    svm, svm_fit = _fit_time(PegasosSVM(lambda_param=lam, epochs=svm_epochs, random_state=SEED), Xtr, d["tier_train"])
    svc, svc_fit = _fit_time(LinearSVC(loss="hinge", C=1/(lam*len(d["tier_train"])), dual="auto",
                                       max_iter=10000, tol=1e-3, random_state=SEED), Xtr, d["tier_train"])
    rff, rff_fit = _fit_time(RandomFourierPegasosSVM(lambda_param=lam, epochs=12, n_components=128,
                             gamma=best_rff["gamma"], random_state=SEED), Xrff_train, d["tier_train"][rff_idx])
    rbf, rbf_fit = _fit_time(Pipeline([("scale", StandardScaler()),
                                      ("svc", SVC(kernel="rbf", C=1/(lam*len(rff_idx)), gamma=best_rff["gamma"]))]), Xtr[rff_idx], d["tier_train"][rff_idx])
    print("  final models fitted; scoring test set...", flush=True)

    test_metrics = {}
    regression_preds = {}
    for name, model, fit_s in (("decision_tree_regression", reg, reg_fit), ("sklearn_tree_regression", sk_reg, sk_reg_fit)):
        start = time.perf_counter(); pred_log = model.predict(Xtest); predict_s = time.perf_counter() - start
        regression_preds[name] = pred_log
        met = regression_summary(d["y_test_log"], pred_log)
        pred_price = np.exp(np.clip(pred_log, 0, 25))
        raw = regression_summary(d["price_test"], pred_price)
        absolute_error = np.abs(d["price_test"] - pred_price)
        worst_index = int(np.argmax(absolute_error))
        worst_row = d["metadata_test"].iloc[worst_index]
        met.update({"rmse_AZN": raw["rmse"], "mae_AZN": raw["mae"], "r2_AZN": raw["r2"],
                    "median_absolute_error_AZN": float(np.median(absolute_error)),
                    "max_absolute_error_AZN": float(absolute_error[worst_index]),
                    "max_error_case": {"actual_AZN": float(d["price_test"][worst_index]),
                                       "predicted_AZN": float(pred_price[worst_index]),
                                       "category": str(worst_row["category"]),
                                       "area_m2": float(worst_row["area_m2"]),
                                       "rooms": float(worst_row["rooms"]) if pd.notna(worst_row["rooms"]) else None,
                                       "location": str(worst_row["location"])},
                    "fit_seconds": fit_s, "predict_seconds": predict_s})
        test_metrics[name] = met
    classification_models = {"decision_tree_classification": (cls, Xtest, cls_fit),
                             "sklearn_tree_classification": (sk_cls, Xtest, sk_cls_fit),
                             "pegasos_svm": (svm, Xtest, svm_fit), "linear_svc": (svc, Xtest, svc_fit),
                             "rff_pegasos_svm": (rff, Xrff_test, rff_fit), "rbf_svc": (rbf, Xtest, rbf_fit)}
    for name, (model, matrix, fit_s) in classification_models.items():
        start = time.perf_counter()
        if "tree" in name:
            scores = model.predict_proba(matrix)[:, 1]
        else:
            scores = model.decision_function(matrix)
        pred = model.predict(matrix)
        pred_s = time.perf_counter() - start
        test_metrics[name] = _with_roc(d["tier_test"], pred, scores)
        test_metrics[name].update({"fit_seconds": fit_s, "predict_seconds": pred_s})
    test_metrics["pegasos_svm"].update({"weight_norm": float(np.linalg.norm(svm.w)), "margin_width": svm.margin_width_,
                                        "support_vector_count": int(svm.support_vector_mask_.sum()), "train_n": len(d["tier_train"])})
    test_metrics["rff_pegasos_svm"].update({"weight_norm": float(np.linalg.norm(rff.model_.w)), "margin_width": rff.margin_width_,
                                            "support_vector_count": rff.n_support_vectors_, "train_n": len(rff_idx),
                                            "rff_components": rff.n_components, "gamma": rff.gamma})
    test_metrics["rbf_svc"].update({"support_vector_count": int(rbf.named_steps["svc"].n_support_.sum()), "train_n": len(rff_idx)})

    print("  5-fold cross-validation...", flush=True)
    cv_results = {
        "decision_tree_regression": _cv_tree(Xtr, d["y_train_log"], "regression", "mse", best_reg["max_depth"], best_reg["min_samples_leaf"]),
        "decision_tree_classification": _cv_tree(Xtr, d["tier_train"], "classification", "gini", best_cls["max_depth"], best_cls["min_samples_leaf"]),
        "pegasos_svm": _cv_svm(Xtr, d["tier_train"], lam, svm_epochs)}

    # Error slices and learned feature rankings.
    err_log = np.log(d["price_test"]) - regression_preds["decision_tree_regression"]
    quartile = np.digitize(d["price_test"], np.quantile(d["price_train"], [.25, .5, .75]))
    pred_price = np.exp(np.clip(regression_preds["decision_tree_regression"], 0, 25))
    subgroup = []
    for q in range(4):
        mask = quartile == q
        subgroup.append({"slice": f"training-price-quartile-{q+1}", "n": int(mask.sum()),
                         "median_actual_AZN": float(np.median(d["price_test"][mask])),
                         "mae_AZN": float(np.mean(np.abs(d["price_test"][mask] - pred_price[mask]))),
                         "mae_log": float(np.mean(np.abs(err_log[mask])))})
    for category in sorted(d["metadata_test"]["category"].dropna().unique()):
        mask = d["metadata_test"]["category"].to_numpy() == category
        subgroup.append({"slice": f"property:{category}", "n": int(mask.sum()),
                         "mae_AZN": float(np.mean(np.abs(d["price_test"][mask] - pred_price[mask]))),
                         "mae_log": float(np.mean(np.abs(err_log[mask])))})
    importance = {}
    for key, model in (("regression", reg), ("classification", cls)):
        indices = np.argsort(model.feature_importances_)[::-1][:20]
        importance[key] = [{"feature": d["feature_names"][i], "importance": float(model.feature_importances_[i])} for i in indices]
    numeric_corr = []
    for i, name in enumerate(d["feature_names"][:d["numeric_feature_count"]]):
        corr = float(np.corrcoef(Xtr[:, i], d["y_train_log"])[0, 1])
        numeric_corr.append({"feature": name, "pearson_r_with_log_price": corr if np.isfinite(corr) else 0.0})
    numeric_corr.sort(key=lambda x: abs(x["pearson_r_with_log_price"]), reverse=True)

    results = {"seed": SEED, "data_audit": d["audit"],
               "split": {"train": len(Xtr), "validation": len(Xv), "test": len(Xtest)},
               "tier_threshold_AZN": d["tier_threshold"],
               "feature_count": len(d["feature_names"]), "numeric_feature_count": d["numeric_feature_count"],
               "feature_names": d["feature_names"], "leakage_columns_excluded": d["audit"]["leakage_columns_excluded"],
               "tuning": {"svm_rff": rff_curve},
               "selected": {"tree_regression": best_reg, "tree_classification": best_cls,
                            "linear_svm": best_linear, "rff_svm": best_rff},
               "cross_validation": cv_results, "test_results": test_metrics,
               "feature_importance": importance, "regression_error_slices": subgroup,
               "exploratory_analysis": {"missing_numeric_percent": d["missing_feature_percent"],
                                        "tier_prevalence": {"train": float(d["tier_train"].mean()), "validation": float(d["tier_validation"].mean()), "test": float(d["tier_test"].mean())},
                                        "numeric_correlations_with_log_price": numeric_corr},
               "svm_diagnostics": {"pegasos_objective_by_epoch": [float(x) for x in svm.objective_history_],
                                   "rff_objective_by_epoch": [float(x) for x in rff.objective_history_],
                                   "pegasos_train_sample_count": len(d["tier_train"]),
                                   "rff_train_sample_count": len(rff_idx)}}
    print("  learning curves and figures...", flush=True)
    results["learning_curves"] = _learning_curves(d, best_reg, best_cls)
    _make_figures(d, results, {"pegasos_svm": svm, "X_test_scaled": Xtest,
                               "tree_regression": reg, "tree_classification": cls})
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if not test_log:   # the results page shows the latest test count
        test_log = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"],
                                  cwd=ROOT, capture_output=True, text=True).stdout
    write_site_data(results, WEB, test_log)
    for name, met in test_metrics.items():
        if "accuracy" in met:
            print(f"  {name:30s} test acc={met['accuracy']:.4f} F1={met['f1']:.4f} AUC={met['roc_auc']:.4f}")
        else:
            print(f"  {name:30s} test RMSE(log)={met['rmse']:.4f} RMSE(AZN)={met['rmse_AZN']:,.0f}")
    return results


if __name__ == "__main__":
    run()
