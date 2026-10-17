# CHANGES

What our project (`b-murad/PegaSold`, master `285edcb`) was missing compared with the two
reference branches, and exactly what was added to which file. Nothing is committed.

## 1. How the gap was found

Every function in both branches was compared with ours, first by name and then by logic
(Python AST with docstrings removed).

- **`clean-pipeline`** (`kenan-karimli/PegaSold`): **nothing missing.** Our master already contains
  every function it has, and is newer where they differ: our `_best_split` uses the fast
  sorted-prefix search instead of rescanning each threshold, and `validate.py` includes the
  leakage audit.
- **`decision-tree`** (`kenan-karimli/PegaSold`): the missing parts below.

### Not ported, and why
| Branch part | Why not |
|---|---|
| `remove_training_price_outliers` (drops 246 train rows) | not in `clean-pipeline`; kept out as agreed, so the dataset stays 57,003 rows |
| branch versions of `deduplicate`, `split`, `apply_filters`, `save` | a different cleaning design (80 features, no repost removal); ours is the `clean-pipeline` design |
| `src/reporting.py` | writes the branch's own markdown/LaTeX/slides from its results format; our report and slides are written directly |
| branch `src/run_all.py` structure | replaced by `src/experiments.py`, which runs the same experiments on our pipeline |
| `styles.css`, `docs/code_explanations/` | not used by the website pages; documentation of the branch's own code |

## 2. Modified files

### `src/decision_tree.py`
- **Feature importances:** each split node now records `n_samples` and `impurity_decrease`, and `fit` computes `feature_importances_`.
- **Branch `_best_split`:** zero-gain splits are now allowed (as in scikit-learn). Without this our tree could not split XOR-like data.
- **Constant-target stop:** a regression node now stops when all targets are identical, compared directly instead of through a float MSE that need not be exactly 0.

The single-tree results on our data are unchanged.
```diff

@@ -32,4 +32,6 @@ class TreeNode:
         self.left = left
         self.right = right
+        self.impurity_decrease = 0.0
+        self.n_samples = 0
 
 class DecisionTree:
@@ -133,4 +135,13 @@ class DecisionTree:
         # Grow the whole tree recursively, starting at depth 0
         self.root = self._build_tree(X, y, depth=0)
+        importances = np.zeros(self.n_features_in_, dtype=float)
+        def collect_importance(node):
+            if node.feature_index is not None:
+                importances[node.feature_index] += node.n_samples * node.impurity_decrease
+                collect_importance(node.left)
+                collect_importance(node.right)
+        collect_importance(self.root)
+        total_importance = importances.sum()
+        self.feature_importances_ = importances / total_importance if total_importance > 0 else importances
 
         return self
@@ -216,58 +227,52 @@ class DecisionTree:
         right_child = self._build_tree(X_right, y_right, depth + 1)
 
-        return TreeNode( feature_index=feature_index, threshold=threshold, left=left_child, right=right_child )
+        node = TreeNode(feature_index=feature_index, threshold=threshold, left=left_child, right=right_child)
+        node.n_samples = len(y)
+        node.impurity_decrease = float(best_split[2])
+        return node
 
     def _best_split(self, X, y):
-        """
-        Search every feature and threshold and return the best split.
-
-        Each feature is sorted once, and cumulative sums give the impurity of
-        every possible left/right division in one pass, instead of recomputing
-        it for each threshold.
+        """Search features and thresholds; return the best useful split.
 
-        :param X: 2D feature array for the rows at this node.
-        :param y: 1D targets for those rows.
-        :return: (feature_index, threshold, gain), or None if no split helps.
+        For numeric features, candidate thresholds can be placed between
+        neighboring distinct feature values. Evaluate each split with
+        _split_score. Return (feature_index, threshold, gain), or None when no
+        split improves the parent node.
         """
         n_samples, n_features = X.shape
-        # Starting at min_impurity_decrease means only splits that beat it can win
-        best_gain, best_feature, best_threshold = self.min_impurity_decrease, None, None
 
-        # Quantities that do not depend on the feature, computed once per node
+        best_gain, best_feature, best_threshold = -np.inf, None, None
+        n = len(y)
         if self.task == "classification":
-            _, encoded = np.unique(y, return_inverse=True)       # labels -> 0..k-1
-            class_count = encoded.max() + 1
-            onehot = np.eye(class_count, dtype=float)[encoded]   # one row per sample, one column per class
-            total_counts = onehot.sum(axis=0)
-            parent_score = self._impurity(y)
+            _, encoded = np.unique(y, return_inverse=True)
+            k = encoded.max() + 1
+            onehot = np.eye(k, dtype=float)[encoded]
+            total = onehot.sum(axis=0)
         else:
-            # Centre y first: variance from sums of squares loses precision on large values
-            yy = y.astype(float) - np.mean(y)
+            # Center the target before prefix-summing squares. Regression
+            # targets here are log(AZN), typically around 12; computing
+            # E[y^2] - E[y]^2 from those large offsets loses precision for
+            # small child nodes and can change which near-tied split wins.
+            # Variance and every MSE split gain are translation invariant.
+            yy = y.astype(float)
+            yy = yy - yy.mean()
             total_sum, total_sq = yy.sum(), np.dot(yy, yy)
-            parent_score = float(total_sq / n_samples)           # MSE of the parent node
+        parent_score = self._impurity(yy) if self.task == "regression" else self._impurity(y)
 
         for feature_index in range(n_features):
-            # Sort rows by this feature; mergesort is stable, so results are reproducible
             order = np.argsort(X[:, feature_index], kind="mergesort")
             values = X[order, feature_index]
-
-            # A split is only possible between two different neighbouring values
+            # A boundary is possible only between unequal adjacent feature values.
             positions = np.flatnonzero(values[:-1] < values[1:]) + 1
-            # min_samples_leaf: keep only splits where both sides are big enough
-            positions = positions[(positions >= self.min_samples_leaf) & (n_samples - positions >= self.min_samples_leaf)]
+            positions = positions[(positions >= self.min_samples_leaf) &
+                                  (n - positions >= self.min_samples_leaf)]
             if not positions.size:
                 continue
-
-            # Threshold halfway between the two neighbouring values
             thresholds = values[positions - 1] + (values[positions] - values[positions - 1]) / 2
-            nl, nr = positions.astype(float), (n_samples - positions).astype(float)   # left/right sizes per candidate
-
             if self.task == "classification":
-                # Class counts on the left of every candidate; the right side is the remainder
                 left_counts = np.cumsum(onehot[order], axis=0)[positions - 1]
-                right_counts = total_counts - left_counts
-
+                right_counts = total - left_counts
+                nl, nr = positions.astype(float), (n - positions).astype(float)
                 if self.criterion == "gini":
-                    # Gini = 1 - sum(p^2), with p = count / size
                     il = 1.0 - np.sum(left_counts ** 2, axis=1) / nl ** 2
                     ir = 1.0 - np.sum(right_counts ** 2, axis=1) / nr ** 2
@@ -275,31 +280,21 @@ class DecisionTree:
                     def entropy(counts, sizes):
                         probs = counts / sizes[:, None]
-                        # maximum() avoids log2(0); where() makes those terms contribute 0
                         return -np.sum(np.where(probs > 0, probs * np.log2(np.maximum(probs, 1e-300)), 0), axis=1)
-
                     il, ir = entropy(left_counts, nl), entropy(right_counts, nr)
             else:
-                # Variance = E[y^2] - E[y]^2, computed from prefix sums for each candidate
                 sorted_y = yy[order]
                 prefix = np.cumsum(sorted_y)
                 prefix_sq = np.cumsum(sorted_y ** 2)
-                left_sum = prefix[positions - 1]
-                left_sq = prefix_sq[positions - 1]
-                right_sum, right_sq = total_sum - left_sum, total_sq - left_sq
-                # maximum(0, ...) guards against tiny negative values from rounding
-                il = np.maximum(0.0, left_sq / nl - (left_sum / nl) ** 2)
-                ir = np.maximum(0.0, right_sq / nr - (right_sum / nr) ** 2)
-
-            # Gain = parent impurity minus the size-weighted child impurity
-            gains = parent_score - (nl / n_samples * il + nr / n_samples * ir)
-            local = int(np.argmax(gains))   # best candidate for this feature
+                nl, nr = positions.astype(float), (n - positions).astype(float)
+                sl, sql = prefix[positions - 1], prefix_sq[positions - 1]
+                sr, sqr = total_sum - sl, total_sq - sql
+                il = np.maximum(0.0, sql / nl - (sl / nl) ** 2)
+                ir = np.maximum(0.0, sqr / nr - (sr / nr) ** 2)
+            gains = parent_score - (positions / n * il + (n - positions) / n * ir)
+            local = int(np.argmax(gains))
             gain = float(gains[local])
+            if gain > best_gain and gain >= self.min_impurity_decrease:
+                best_gain, best_feature, best_threshold = gain, feature_index, float(thresholds[local])
 
-            # Keep it only if it beats the best split found on earlier features
-            if gain > best_gain:
-                best_gain, best_feature = gain, feature_index
-                best_threshold = float(thresholds[local])
-
-        # Nothing beat min_impurity_decrease (or no valid split existed)
         if best_feature is None:
             return None
@@ -323,6 +318,5 @@ class DecisionTree:
 
         # Regression equivalent of "pure": all targets identical, so MSE is 0
-        # (relies on _impurity, which is implemented by a teammate)
-        return self.task == "regression" and self._impurity(y) == 0
+        return self.task == "regression" and bool(np.all(y == y[0]))
 
     def _leaf_value(self, y):
@@ -352,4 +346,5 @@ class DecisionTree:
         """
         node = TreeNode(value=prediction)
+        node.n_samples = len(y)
 
         if self.task == "classification":
```

### `src/svm.py`
- **Pegasos fixes:** the intercept step is capped at 1, and `w` is projected onto `||w|| <= 1/sqrt(lambda)` after every update, as in the original algorithm.
- **New validation:** `learning_rate * lambda < 1` is now checked for a constant learning rate.
- **New class:** `RandomFourierPegasosSVM`, the nonlinear SVM, is added.

Effect on our data: validation accuracy 0.9029 -> 0.9028, test accuracy unchanged at 0.9084.
```diff

@@ -32,6 +32,12 @@ class PegasosSVM:
         if epochs < 1:
             raise ValueError("epochs must be at least 1")
-        if learning_rate != "pegasos" and (not np.isscalar(learning_rate) or learning_rate <= 0):
-            raise ValueError("learning_rate must be 'pegasos' or a positive number")
+        if isinstance(learning_rate, str):
+            if learning_rate != "pegasos":
+                raise ValueError("learning_rate must be 'pegasos' or a positive number")
+        else:
+            if not np.isscalar(learning_rate) or learning_rate <= 0:
+                raise ValueError("learning_rate must be 'pegasos' or a positive number")
+            if learning_rate * lambda_param >= 1:
+                raise ValueError("learning_rate * lambda_param must be < 1")
         self.lambda_param = float(lambda_param)
         self.epochs = int(epochs)
@@ -74,5 +80,17 @@ class PegasosSVM:
                 if margin < 1.0:
                     self.w += eta * signs[i] * X[i]
-                    self.b += eta * signs[i]
+                    # Bound the unregularized intercept update. The first
+                    # Pegasos steps can be very large for small lambda; without
+                    # this cap the bias can become oversized on centered data,
+                    # destabilizing the objective and decision boundary.
+                    self.b += min(eta, 1.0) * signs[i]
+                # Pegasos includes a projection onto ||w|| <= 1/sqrt(lambda).
+                # Without it, the first 1/(lambda*t) updates can leave the
+                # feasible ball (especially at t=1), making the optimizer
+                # depend heavily on the first few shuffled rows.
+                norm = np.linalg.norm(self.w)
+                radius = 1.0 / np.sqrt(self.lambda_param)
+                if norm > radius:
+                    self.w *= radius / norm
             self.objective_history_.append(self._objective(X, signs))
 
@@ -103,2 +121,54 @@ class PegasosSVM:
         scores = self.decision_function(X)
         return self.classes_[(scores >= 0).astype(int)]
+
+
+class RandomFourierPegasosSVM:
+    """Nonlinear RBF approximation followed by the same NumPy Pegasos solver.
+
+    Random Fourier features approximate the RBF kernel using
+    ``sqrt(2/D) cos(X @ W + phase)``. This meets the nonlinear-model option
+    without calling a library kernel/model implementation.
+    """
+
+    def __init__(self, lambda_param=0.01, epochs=30, n_components=256,
+                 gamma=0.1, random_state=42):
+        if n_components < 1 or gamma <= 0:
+            raise ValueError("n_components and gamma must be positive")
+        self.lambda_param = lambda_param
+        self.epochs = epochs
+        self.n_components = int(n_components)
+        self.gamma = float(gamma)
+        self.random_state = random_state
+
+    def _map(self, X):
+        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.random_weights_ + self.random_phase_)
+
+    def fit(self, X, y):
+        X = np.asarray(X, dtype=float)
+        if X.ndim != 2 or X.shape[0] == 0 or not np.isfinite(X).all():
+            raise ValueError("X must be a non-empty finite 2D array")
+        self.n_features_in_ = X.shape[1]
+        rng = np.random.default_rng(self.random_state)
+        self.random_weights_ = rng.normal(0, np.sqrt(2 * self.gamma),
+                                          size=(self.n_features_in_, self.n_components))
+        self.random_phase_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
+        self.model_ = PegasosSVM(lambda_param=self.lambda_param, epochs=self.epochs,
+                                 random_state=self.random_state).fit(self._map(X), y)
+        self.classes_ = self.model_.classes_
+        self.objective_history_ = self.model_.objective_history_
+        self.margin_width_ = self.model_.margin_width_
+        self.support_vector_mask_ = self.model_.support_vector_mask_
+        self.n_support_vectors_ = int(self.support_vector_mask_.sum())
+        return self
+
+    def decision_function(self, X):
+        if not hasattr(self, "model_"):
+            raise ValueError("fit must be called before prediction")
+        X = np.asarray(X, dtype=float)
+        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
+            raise ValueError("X must be finite, 2D, with the fitted feature count")
+        return self.model_.decision_function(self._map(X))
+
+    def predict(self, X):
+        scores = self.decision_function(X)
+        return self.classes_[(scores >= 0).astype(int)]
```

### `src/run_all.py`
New stage 5 runs `src.experiments`, and `--skip-experiments` turns it off.
```diff

@@ -5,5 +5,5 @@
     python -m src.run_all --skip-prep  # reuse data/processed, re-score only
 
-Pipeline: data_prep -> validate -> explore (EDA) -> evaluate -> summary.
+Pipeline: data_prep -> validate -> explore (EDA) -> evaluate -> experiments -> summary.
 
 Every seed is fixed (RANDOM_SEED = 42 in src/data_prep.py and src/evaluate.py),
@@ -132,4 +132,6 @@ def main() -> int:
     p.add_argument("--no-ensemble", action="store_true",
                    help="skip bagging (saves most of the model runtime)")
+    p.add_argument("--skip-experiments", action="store_true",
+                   help="skip src.experiments (RFF SVM, cross-validation, web report data)")
     a = p.parse_args()
     start = time.time()
@@ -169,5 +171,11 @@ def main() -> int:
     sh(*cmd)
 
-    # 5. one-page summary pointing at the artifacts
+    # 5. extra experiments from the decision-tree branch + web report data
+    if a.skip_experiments or a.fast:
+        print("\n[skip-experiments] src.experiments not run")
+    else:
+        sh(sys.executable, "-m", "src.experiments")
+
+    # 6. one-page summary pointing at the artifacts
     print("\n$ write summary")
     write_summary(a.processed, a.fig_out, time.time() - start)
```

### `explore.py`
Replaced with the branch's `src/explore.py`, copied verbatim. The branch version adds:
- `explore_raw()`
- `latest_listing_per_id()`, so the EDA counts the same one-row-per-advert snapshot that `data_prep` keeps
- normalised advert IDs
- zero-division guards
- no hard-coded counts in the text

The diff is too large to list. Compare with `git diff explore.py`.

## 3. New files

| File | Source |
|---|---|
| `src/metrics.py` | copy of the branch `src/evaluate.py`; its two summary wrappers are renamed `regression_summary` / `classification_summary`, so they cannot be confused with the different `evaluate.py` metrics (NumPy accuracy, confusion matrix, precision/recall/F1, ROC/AUC, MAE/MSE/RMSE/R2/MAPE, `compare_models`, `kfold_indices`, `cross_validate`, `svm_lambda_study`, plots). A new name, because our `evaluate.py` already uses `roc_curve` and `confusion_matrix` from scikit-learn |
| `src/tree_viz.py` | **verbatim** (`export_text`, `to_dot`, `plot_tree`) |
| `src/site_data.py` | **verbatim** (builds `site-data.js` for the report page) |
| `src/experiments.py` | the branch's `run_all` experiments ported to our pipeline. Code below |
| `src/web/web_demo.py` | the branch's server, rewired to our `fit_preprocessor`/`build_features` and `metrics.json`. Code below |
| `src/web/demo.css`, `app.js`, `report-theme.css` | **verbatim** |
| `src/web/index.html`, `demo.js`, `report.html` | branch files with small fixes. Diffs below |
| `src/web/assets/pegasold-logo.png`, `src/web/favicon.svg` | the original Pegasus logo recoloured to the site's palette (black artwork on lime); new minimal favicon |
| `src/web/site-data.js` | generated by `src.experiments` |
| `tests/test_tree_viz.py`, `tests/test_site_data.py` | **verbatim** |
| `tests/test_metrics.py` | the branch test for the NumPy metrics |

### `src/experiments.py`
```python
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
```

### `src/web/web_demo.py`
```python
"""Local web demo for PegaSold: price estimate (Task A) + premium/standard tier (Task B).

Run from the repo root (after `python -m src.run_all`):

    python -m src.web.web_demo            # single trees (fast start)
    python -m src.web.web_demo --bagged   # 30-tree bagging ensemble (slower start, more accurate)

Everything is fitted from the saved training split with the project's own NumPy
models; nothing is mocked. Feature construction reuses src/data_prep.py.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from src.data_prep import build_features, fit_preprocessor
from src.decision_tree import DecisionTree
from src.ensemble import BaggedTrees

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "reports" / "figures" / "metrics.json"
WEB_ROOT = Path(__file__).resolve().parent
MIN_SAMPLES_LEAF = 5   # same as src/evaluate.py
PROJECT_FILES = ("reports/figures/", "reports/summary.md", "report/report.pdf",
                 "presentation/slides.pdf", "CHANGES.md")


class EstimateService:
    """Fit the selected project models from the saved training split and predict."""

    def __init__(self, processed_dir: Path = PROCESSED, bagged: bool = False) -> None:
        self.processed_dir = Path(processed_dir)
        required = ["X_train.npy", "y_train_log.npy", "tier_train.npy", "train_clean.csv"]
        missing = [n for n in required if not (self.processed_dir / n).is_file()]
        if not METRICS.is_file():
            missing.append(str(METRICS))
        if missing:
            raise FileNotFoundError("Prepared project data is missing: " + ", ".join(missing)
                                    + ". Run `python -m src.run_all` first.")
        metrics = json.loads(METRICS.read_text(encoding="utf-8"))

        clean = pd.read_csv(self.processed_dir / "train_clean.csv")
        # Same function and same rows as the original run, so the rules are identical.
        self.rules = fit_preprocessor(clean)

        X = np.load(self.processed_dir / "X_train.npy")
        y_reg = np.load(self.processed_dir / "y_train_log.npy")
        y_cls = np.load(self.processed_dir / "tier_train.npy")
        if not (len(X) == len(y_reg) == len(y_cls) == len(clean)):
            raise ValueError("Training features and targets have inconsistent row counts")

        if bagged:
            kw = dict(n_estimators=metrics["n_estimators"], max_samples=metrics["bag_max_samples"],
                      max_depth=metrics["bag_max_depth"], min_samples_leaf=2, random_state=42)
            self.regressor = BaggedTrees(task="regression", **kw).fit(X, y_reg)
            self.classifier = BaggedTrees(task="classification", **kw).fit(X, y_cls)
            self.model_name = "PegaSold NumPy Bagged Trees"
        else:
            self.regressor = DecisionTree(task="regression", max_depth=metrics["best_depth_regression"],
                                          min_samples_leaf=MIN_SAMPLES_LEAF).fit(X, y_reg)
            self.classifier = DecisionTree(task="classification", max_depth=metrics["best_depth_classification"],
                                           min_samples_leaf=MIN_SAMPLES_LEAF).fit(X, y_cls)
            self.model_name = "PegaSold NumPy Decision Tree"
        self.train_rows = int(len(X))

        clean["lat"] = pd.to_numeric(clean["lat"], errors="coerce")
        clean["lng"] = pd.to_numeric(clean["lng"], errors="coerce")
        coordinates = clean.dropna(subset=["lat", "lng"]).groupby("location")[["lat", "lng"]].median()
        self.coordinates = {str(k): [float(v.lat), float(v.lng)] for k, v in coordinates.iterrows()}

    def metadata(self) -> dict:
        locations = sorted(self.rules["locations"])
        return {
            "categories": self.rules["categories"],
            "locations": locations,
            "threshold_azn": float(self.rules["tier_threshold_AZN"]),
            "training_rows": self.train_rows,
            "location_count": len(locations),
        }

    def predict(self, request: dict) -> dict:
        try:
            category = str(request["category"])
            location = str(request["location"])
            area = float(request["area_m2"])
            rooms_value = float(request["rooms"])
            if not rooms_value.is_integer():
                raise ValueError("Room count must be a whole number.")
            rooms = int(rooms_value)
            floor = request.get("floor")
            n_floors = request.get("n_floors")
            land = float(request.get("land_m2", 0) or 0)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Please check the property details.") from exc
        if category not in self.rules["categories"]:
            raise ValueError("Select a supported residential property type.")
        if not (10 <= area <= 2000):
            raise ValueError("Floor area must be between 10 and 2,000 m².")
        if not (1 <= rooms <= 30):
            raise ValueError("Room count must be between 1 and 30.")
        if not (0 <= land <= 200000):
            raise ValueError("Land area must be between 0 and 200,000 m².")
        # Rare and unseen locations correctly become the preprocessing 'other' bucket.
        location_for_features = location.strip()
        if not location_for_features:
            raise ValueError("Select a location.")
        has_floor = floor not in (None, "") and n_floors not in (None, "")
        try:
            floor_value = float(floor) if has_floor else np.nan
            floors_value = float(n_floors) if has_floor else np.nan
        except (TypeError, ValueError) as exc:
            raise ValueError("Floor values must be whole numbers.") from exc
        if has_floor and (not floor_value.is_integer() or not floors_value.is_integer()):
            raise ValueError("Floor values must be whole numbers.")
        floor = int(floor_value) if has_floor else np.nan
        n_floors = int(floors_value) if has_floor else np.nan
        if has_floor and (floor < 1 or n_floors < floor or n_floors > 100):
            raise ValueError("Floor number must fit within the building's total floors.")

        coords = self.coordinates.get(location)
        approximate_location = coords is None
        if coords is None:
            coords = [self.rules["fill_values"]["lat"], self.rules["fill_values"]["lng"]]
        def flag(name: str) -> int:
            value = request.get(name, False)
            if isinstance(value, bool):
                return int(value)
            if value in (0, 1):
                return int(value)
            if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                return int(value.strip().lower() == "true")
            raise ValueError(f"{name} must be true or false.")

        row = pd.DataFrame([{
            "area_m2": area, "land_m2": land, "rooms": rooms,
            "floor": floor, "n_floors": n_floors,
            "lat": coords[0], "lng": coords[1], "category": category,
            "location": location_for_features,
            "has_repair": flag("has_repair"),
            "has_bill_of_sale": flag("has_bill_of_sale"),
            "has_mortgage": flag("has_mortgage"),
            "is_vip": 0, "is_featured": 0, "sold_by_owner": 0,
        }])
        vector = build_features(row, self.rules).to_numpy(dtype=float)
        log_price = float(self.regressor.predict(vector)[0])
        price = float(np.exp(np.clip(log_price, -50, 50)))
        predicted_class = int(self.classifier.predict(vector)[0])
        probabilities = self.classifier.predict_proba(vector)[0]
        class_index = int(np.flatnonzero(self.classifier.classes_ == predicted_class)[0])
        return {
            "price_azn": round(price),
            "price_per_m2_azn": round(price / area),
            "tier": "Premium" if predicted_class == 1 else "Standard",
            "tier_probability": float(probabilities[class_index]),
            "tier_threshold_azn": float(self.rules["tier_threshold_AZN"]),
            "location_coordinate_approximate": approximate_location,
            "model": self.model_name,
        }


SERVICE = None


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print("[web] " + fmt % args)

    def send_json(self, status: int, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(200, {"ready": SERVICE is not None})
            return
        if path == "/api/metadata":
            self.send_json(200, SERVICE.metadata())
            return
        relative = "index.html" if path in ("", "/") else unquote(path.lstrip("/"))
        # The report page links project artifacts outside src/web; serve only these.
        base = ROOT if relative.startswith(PROJECT_FILES) else WEB_ROOT
        target = (base / relative).resolve()
        inside = base.resolve() in target.parents
        if base is ROOT:   # re-check after resolving, so "reports/figures/../.." cannot escape
            inside = inside and target.relative_to(ROOT.resolve()).as_posix().startswith(PROJECT_FILES)
        if not inside:
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        mime = "text/html; charset=utf-8" if target.suffix == ".html" else {
            ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
            ".png": "image/png", ".svg": "image/svg+xml", ".jpg": "image/jpeg",
            ".pdf": "application/pdf", ".md": "text/markdown; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")   # so a changed favicon/logo shows up
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                raise ValueError("Request size is invalid.")
            payload = json.loads(self.rfile.read(length))
            self.send_json(200, SERVICE.predict(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})


def main() -> None:
    global SERVICE
    parser = argparse.ArgumentParser(description="Run the local PegaSold estimate demo.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bagged", action="store_true", help="serve the bagging ensemble instead of single trees")
    args = parser.parse_args()
    SERVICE = EstimateService(bagged=args.bagged)
    port = args.port
    print(f"PegaSold price demo is ready: http://127.0.0.1:{port}")
    print("Models fitted from the saved training split; Ctrl+C to stop.")
    ThreadingHTTPServer(("127.0.0.1", port), DemoHandler).serve_forever()


if __name__ == "__main__":
    main()
```

### `src/web/index.html` vs branch `index.html`: favicon cache-busting (`?v=3`); the hero training count was hard-coded to the branch's 40,279 and now comes from the server
```diff
@@ -5,7 +5,7 @@
   <meta name="theme-color" content="#f7f7f8">
   <meta name="description" content="Estimate a Baku home's asking price with PegaSold.">
   <title>PegaSold — Home Price Estimate</title>
-  <link rel="icon" href="favicon.svg" type="image/svg+xml">
+  <link rel="icon" href="favicon.svg?v=3" type="image/svg+xml">
   <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
   <link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
   <link rel="stylesheet" href="demo.css"><script src="demo.js" defer></script>
@@ -21,7 +21,7 @@
     <div class="hero-copy"><div class="eyebrow"><span class="live-dot"></span> PEGASOLD · BAKU</div>
       <h1>Estimate your home's<br><span>asking price.</span></h1>
       <p>Choose a few property details to see an estimated price and price tier.</p>
-      <div class="hero-meta"><span><b class="meta-mark">01</b> Bina.az listings</span><span><b class="meta-mark">02</b> 40,279 training listings</span></div>
+      <div class="hero-meta"><span><b class="meta-mark">01</b> Bina.az listings</span><span><b class="meta-mark">02</b> <span id="hero-rows">-</span> training listings</span></div>
     </div>
     <aside class="hero-note"><span class="note-icon">PS</span><span class="eyebrow">PROPERTY SNAPSHOT</span><strong>Three residential types</strong><p>Based on historical Bina.az asking prices in Baku.</p><a href="report.html">View results <span>↗</span></a></aside>
   </section>
```

### `src/web/demo.js` vs branch `demo.js`: fills the hero training count
```diff
@@ -34,6 +34,7 @@
     addOptions(locationSelect, metadata.locations, 'Nəsimi r.');
     addOptions(roomSelect, Array.from({ length: 30 }, (_, i) => String(i + 1)), '3');
     $('#stat-rows').textContent = Number(metadata.training_rows).toLocaleString('en-US');
+    $('#hero-rows').textContent = Number(metadata.training_rows).toLocaleString('en-US');
     $('#stat-threshold').textContent = money(metadata.threshold_azn);
     $('#model-state').textContent = 'Ready';
     $('#footer-status').textContent = 'Estimator ready';
```

### `src/web/report.html` vs branch `report.html`: facts corrected to our pipeline (544 reposts instead of 1,007 ambiguous rows, random split, CV on the full train split), bagging sentence, team roles, links to files that exist, our depth-sweep figure
```diff
@@ -2,7 +2,7 @@
 <html lang="en">
 <head>
   <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
-  <meta name="theme-color" content="#f7f7f8"><link rel="icon" href="favicon.svg" type="image/svg+xml">
+  <meta name="theme-color" content="#f7f7f8"><link rel="icon" href="favicon.svg?v=3" type="image/svg+xml">
   <meta name="description" content="Current PegaSold experiment report: data preparation, metrics, model comparison, validation, diagnostics and limitations.">
   <title>Project Results & Report — PegaSold</title>
   <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
@@ -11,29 +11,29 @@
 <body id="top">
   <header class="topbar"><div class="topbar-inner"><a class="brand" href="index.html"><img src="assets/pegasold-logo.png" alt="PegaSold logo"><span class="brand-name">PegaSold <i>//</i></span><span class="brand-caption">PROJECT RESULTS</span></a><nav aria-label="Main navigation"><a href="index.html">Price estimate</a><a class="nav-active" href="report.html">Report</a></nav><a class="button button-lime top-action" href="presentation/slides.pdf">View slides <span>↗</span></a></div></header>
   <main class="wrap report-main">
-    <section class="report-hero"><div class="eyebrow"><span class="live-dot"></span> BINA.AZ · 2024</div><h1>Project results</h1><p>Model performance, data preparation and evaluation on held-out listings.</p><div class="hero-actions"><a class="button button-lime" href="reports/final_report.md">Full Markdown report <span>↓</span></a><a class="button button-outline" href="presentation/slides.pdf">Presentation slides <span>↗</span></a></div></section>
+    <section class="report-hero"><div class="eyebrow"><span class="live-dot"></span> BINA.AZ · 2024</div><h1>Project results</h1><p>Model performance, data preparation and evaluation on held-out listings.</p><div class="hero-actions"><a class="button button-lime" href="reports/summary.md">Full Markdown report <span>↓</span></a><a class="button button-outline" href="presentation/slides.pdf">Presentation slides <span>↗</span></a></div></section>
 
-    <section class="report-section" id="status"><h2>Executive summary</h2><p>Two tasks use the same cleaned Baku residential listings: regression predicts logarithmic advertised price, while classification labels a listing premium when price is at or above the training-set median. The scratch NumPy tree performs strongly on the binary tier task and closely matches the scikit-learn reference. The AZN regression error remains substantial, particularly for the highest-priced listings.</p><div class="metric-grid"><article class="metric-card metric-featured"><div class="metric-top"><span>TREE · TEST ACCURACY</span><span class="metric-icon">◈</span></div><strong id="metric-accuracy">—</strong><p>price-tier classification</p><div class="metric-divider"></div><div class="metric-detail"><span>F1 <b id="metric-tree-f1">—</b></span><span>AUC <b id="metric-tree-auc">—</b></span></div></article><article class="metric-card"><div class="metric-top"><span>PEGASOS · TEST</span><span class="metric-icon icon-purple">∿</span></div><strong id="metric-svm-acc">—</strong><p>classification accuracy</p><div class="metric-divider"></div><div class="metric-detail"><span>F1 <b id="metric-svm-f1">—</b></span><span>AUC <b id="metric-svm-auc">—</b></span></div></article><article class="metric-card"><div class="metric-top"><span>TREE · REGRESSION</span><span class="metric-icon icon-blue">⌁</span></div><strong id="metric-rmse">—</strong><p>RMSE in ln(AZN)</p><div class="metric-divider"></div><div class="metric-detail"><span>RMSE <b id="metric-rmse-azn">—</b></span><span>MAE <b id="metric-mae-azn">—</b></span></div></article><article class="metric-card metric-yellow"><div class="metric-top"><span>CLEAN DATASET</span><span class="metric-icon">▦</span></div><strong id="metric-clean">—</strong><p>model-ready listings</p><div class="metric-divider"></div><div class="metric-detail"><span>features <b id="metric-features">—</b></span><span>cut <b id="threshold-inline">— AZN</b></span></div></article></div></section>
+    <section class="report-section" id="status"><h2>Executive summary</h2><p>Two tasks use the same cleaned Baku residential listings: regression predicts logarithmic advertised price, while classification labels a listing premium when price is at or above the training-set median. The scratch NumPy tree performs strongly on the binary tier task and closely matches the scikit-learn reference. The AZN regression error remains substantial, particularly for the highest-priced listings. Our 30-tree bagging ensemble is the strongest model on both tasks: test R² 0.886, accuracy 0.934 and ROC-AUC 0.983 (see the run summary).</p><div class="metric-grid"><article class="metric-card metric-featured"><div class="metric-top"><span>TREE · TEST ACCURACY</span><span class="metric-icon">◈</span></div><strong id="metric-accuracy">—</strong><p>price-tier classification</p><div class="metric-divider"></div><div class="metric-detail"><span>F1 <b id="metric-tree-f1">—</b></span><span>AUC <b id="metric-tree-auc">—</b></span></div></article><article class="metric-card"><div class="metric-top"><span>PEGASOS · TEST</span><span class="metric-icon icon-purple">∿</span></div><strong id="metric-svm-acc">—</strong><p>classification accuracy</p><div class="metric-divider"></div><div class="metric-detail"><span>F1 <b id="metric-svm-f1">—</b></span><span>AUC <b id="metric-svm-auc">—</b></span></div></article><article class="metric-card"><div class="metric-top"><span>TREE · REGRESSION</span><span class="metric-icon icon-blue">⌁</span></div><strong id="metric-rmse">—</strong><p>RMSE in ln(AZN)</p><div class="metric-divider"></div><div class="metric-detail"><span>RMSE <b id="metric-rmse-azn">—</b></span><span>MAE <b id="metric-mae-azn">—</b></span></div></article><article class="metric-card metric-yellow"><div class="metric-top"><span>CLEAN DATASET</span><span class="metric-icon">▦</span></div><strong id="metric-clean">—</strong><p>model-ready listings</p><div class="metric-divider"></div><div class="metric-detail"><span>features <b id="metric-features">—</b></span><span>cut <b id="threshold-inline">— AZN</b></span></div></article></div></section>
 
-    <section class="report-section"><h2>1. Dataset preparation and audit</h2><p>The source has repeated scrape rows, mixed Azerbaijani text/numeric columns, absent measurements, price-derived fields and property categories with different feature availability. The pipeline keeps one latest dated record per normalized advert ID, scopes to Baku and three residential types, and applies the declared domain windows. The raw CSV remains untouched.</p><div class="report-table-wrap"><table class="report-table" id="cleaning-table"><tbody><tr><td colspan="3">Loading audit…</td></tr></tbody></table></div><div class="report-callout"><strong>Filtering decisions:</strong> price 10,000–5,000,000 AZN; floor area 10–2,000 m²; unit price 200–20,000 AZN/m²; rooms 1–30 where present. The source EDA informed these fixed limits before splitting. This is a limitation: the cutoffs were not selected from outside knowledge alone.</div><p style="margin-top:12px">A further 1,007 rows participate in ambiguous coordinate/area/room/price candidate matches. They are retained and reported because those fields alone do not prove two adverts represent the same property. Leakage-prone `total_price`, `unit_price`, descriptions, identifiers, raw address and owner/agency identifiers do not enter the model matrix.</p><div class="report-chart-grid"><figure class="report-chart"><img src="reports/figures/data_and_prices.png" alt="Cleaning attrition and training price histogram" loading="lazy"><figcaption>Clean-up attrition and the long-tailed price distribution after preparation.</figcaption></figure><figure class="report-chart"><img src="reports/figures/missingness.png" alt="Feature missingness by column" loading="lazy"><figcaption>Missingness in the model inputs; missing rooms and coordinates are imputed from train medians.</figcaption></figure></div></section>
+    <section class="report-section"><h2>1. Dataset preparation and audit</h2><p>The source has repeated scrape rows, mixed Azerbaijani text/numeric columns, absent measurements, price-derived fields and property categories with different feature availability. The pipeline keeps one latest dated record per normalized advert ID, scopes to Baku and three residential types, and applies the declared domain windows. The raw CSV remains untouched.</p><div class="report-table-wrap"><table class="report-table" id="cleaning-table"><tbody><tr><td colspan="3">Loading audit…</td></tr></tbody></table></div><div class="report-callout"><strong>Filtering decisions:</strong> price 10,000–5,000,000 AZN; floor area 10–2,000 m²; unit price 200–20,000 AZN/m². The source EDA informed these fixed limits before splitting. This is a limitation: the cutoffs were not selected from outside knowledge alone.</div><p style="margin-top:12px">A further 544 reposted twins (the same advert posted again under a new ID) are removed during deduplication. Leakage-prone `total_price`, `unit_price`, descriptions, identifiers, raw address and owner/agency identifiers do not enter the model matrix.</p><div class="report-chart-grid"><figure class="report-chart"><img src="reports/figures/data_and_prices.png" alt="Cleaning attrition and training price histogram" loading="lazy"><figcaption>Clean-up attrition and the long-tailed price distribution after preparation.</figcaption></figure><figure class="report-chart"><img src="reports/figures/missingness.png" alt="Feature missingness by column" loading="lazy"><figcaption>Missingness in the model inputs; missing rooms and coordinates are imputed from train medians.</figcaption></figure></div></section>
 
-    <section class="report-section"><h2>2. Targets, split and leakage control</h2><p><b>Task A:</b> fit the tree to ln(price) to reduce the leverage of the long right tail, then report error in both logarithmic space and AZN. <b>Task B:</b> premium is price ≥ the training partition's median; the current threshold is <b id="threshold-report">—</b> AZN. The split is seed-controlled and allocated across price bins to keep class shares close to balanced.</p><div class="pipeline"><div><b>Train · 70%</b><small id="split-train">— listings; fits preprocessing and models</small></div><div><b>Validation · 15%</b><small id="split-validation">— listings; selects model settings</small></div><div><b>Test · 15%</b><small id="split-test">— listings; final score only</small></div><div><b>Feature matrix</b><small id="split-features">— numerical and encoded columns</small></div></div><div class="report-callout"><strong>Train-only rules:</strong> room/coordinate medians, standardization statistics, frequent-location vocabulary and premium threshold are learned using train data and applied unchanged to validation/test. The raw source distribution was examined when selecting fixed cleaning bounds; this source-informed choice is disclosed.</div></section>
+    <section class="report-section"><h2>2. Targets, split and leakage control</h2><p><b>Task A:</b> fit the tree to ln(price) to reduce the leverage of the long right tail, then report error in both logarithmic space and AZN. <b>Task B:</b> premium is price ≥ the training partition's median; the current threshold is <b id="threshold-report">—</b> AZN. The split is a seeded random permutation (seed 42); premium shares stay close to balanced (0.500 / 0.505 / 0.501).</p><div class="pipeline"><div><b>Train · 70%</b><small id="split-train">— listings; fits preprocessing and models</small></div><div><b>Validation · 15%</b><small id="split-validation">— listings; selects model settings</small></div><div><b>Test · 15%</b><small id="split-test">— listings; final score only</small></div><div><b>Feature matrix</b><small id="split-features">— numerical and encoded columns</small></div></div><div class="report-callout"><strong>Train-only rules:</strong> room/coordinate medians, standardization statistics, frequent-location vocabulary and premium threshold are learned using train data and applied unchanged to validation/test. The raw source distribution was examined when selecting fixed cleaning bounds; this source-informed choice is disclosed.</div></section>
 
     <section class="report-section" id="all-results"><h2>3. Complete held-out results</h2><p>Classification metrics use the positive “premium” class for precision, recall and F1. ROC-AUC uses continuous decision scores. Fit and prediction times are wall-clock results from this run, not portable performance guarantees.</p><div class="result-tabs" role="tablist"><button class="active" data-result-tab="classification" role="tab">Classification</button><button data-result-tab="regression" role="tab">Regression</button></div><div data-result-panel="classification"><div class="report-table-wrap"><table class="report-table" id="classification-table"><tbody><tr><td>Loading test metrics…</td></tr></tbody></table></div></div><div data-result-panel="regression" hidden><div class="report-table-wrap"><table class="report-table" id="regression-table"><tbody><tr><td>Loading regression metrics…</td></tr></tbody></table></div></div><div class="report-callout"><strong>Interpretation:</strong> Scratch and library tree results are close, supporting that the from-scratch split search is behaving plausibly on this data. Linear Pegasos obtains a slightly lower threshold accuracy than the tree but stronger ROC-AUC, indicating better score ranking at alternate thresholds. The RFF approximation and RBF SVC are weaker in this selected configuration.</div><div class="report-chart-grid"><figure class="report-chart"><img src="reports/figures/classification_comparison.png" alt="Classification score comparison" loading="lazy"><figcaption>Test classification scores across scratch models and references.</figcaption></figure><figure class="report-chart"><img src="reports/figures/roc_curves.png" alt="ROC curves for test classifications" loading="lazy"><figcaption>ROC curves over held-out test examples.</figcaption></figure><figure class="report-chart"><img src="reports/figures/regression_error_analysis.png" alt="Regression predictions and residuals" loading="lazy"><figcaption>Log-price prediction and residual diagnostics.</figcaption></figure><figure class="report-chart"><img src="reports/figures/tree_confusion_matrix.png" alt="Decision tree test confusion matrix" loading="lazy"><figcaption>Scratch tree's test confusion matrix (rows: actual tier, columns: predicted tier).</figcaption></figure></div></section>
 
     <section class="report-section" id="interpretation"><h2>4. Regression error analysis and feature signal</h2><p>Currency-scale errors increase sharply in the top price quartile. That pattern explains why the model can have good log-space R² while still missing luxury-property prices by large AZN amounts. Quartile boundaries below come from the training target distribution; the reported errors use test rows.</p><div class="report-table-wrap"><table class="report-table" id="error-table"><tbody><tr><td>Loading error slices…</td></tr></tbody></table></div><h3 style="font:700 15px var(--display);margin:22px 0 6px">Largest decision-tree regression importances</h3><p>Impurity importance indicates which features reduced tree error in this fitted model; correlated variables can share or exchange importance. It is predictive description, not causal impact.</p><div class="feature-bars" id="importance-bars"></div><div class="report-chart-grid"><figure class="report-chart"><img src="reports/figures/feature_importance_regression.png" alt="Decision tree regression feature importance" loading="lazy"><figcaption>Regression tree's leading feature importances.</figcaption></figure><figure class="report-chart"><img src="reports/figures/price_vs_area.png" alt="Log area and log price relationship" loading="lazy"><figcaption>Area and price move together in the cleaned training data; association is not causation.</figcaption></figure></div></section>
 
-    <section class="report-section"><h2>5. Cross-validation and model selection</h2><p>Five seeded folds are evaluated on a 6,000-row sample drawn from train. Settings were selected on validation, then fixed for the cross-validation diagnostic.</p><div class="report-table-wrap"><table class="report-table" id="cv-table"><tbody><tr><td>Loading cross-validation results…</td></tr></tbody></table></div><div class="report-callout"><strong>CV caveat:</strong> the feature imputation/scaling and categorical vocabulary are prepared on the full training split before these folds are formed. Thus the cross-validation spread is useful as a stability diagnostic, but it is not a fully nested pipeline CV estimate. The held-out test result remains separate.</div><div class="report-chart-grid"><figure class="report-chart"><img src="reports/figures/tree_tuning.png" alt="Validation tree hyperparameter sweeps" loading="lazy"><figcaption>Validation-based tree sweeps used to select depth, split criterion and leaf size.</figcaption></figure><figure class="report-chart"><img src="reports/figures/tree_learning_curves.png" alt="Decision tree learning curves" loading="lazy"><figcaption>Training/validation behavior as training sample size grows.</figcaption></figure><figure class="report-chart"><img src="reports/figures/pegasos_objective.png" alt="Pegasos objective by epoch" loading="lazy"><figcaption>Linear Pegasos objective decreases over training epochs.</figcaption></figure><figure class="report-chart"><img src="reports/figures/svm_margin_scores.png" alt="Pegasos signed decision scores" loading="lazy"><figcaption>Signed SVM decision scores around the zero decision boundary.</figcaption></figure></div></section>
+    <section class="report-section"><h2>5. Cross-validation and model selection</h2><p>Five seeded folds are evaluated on the full training split. Settings were selected on validation, then fixed for the cross-validation diagnostic.</p><div class="report-table-wrap"><table class="report-table" id="cv-table"><tbody><tr><td>Loading cross-validation results…</td></tr></tbody></table></div><div class="report-callout"><strong>CV caveat:</strong> the feature imputation/scaling and categorical vocabulary are prepared on the full training split before these folds are formed. Thus the cross-validation spread is useful as a stability diagnostic, but it is not a fully nested pipeline CV estimate. The held-out test result remains separate.</div><div class="report-chart-grid"><figure class="report-chart"><img src="reports/figures/fig1_bias_variance.png" alt="Validation tree hyperparameter sweeps" loading="lazy"><figcaption>Training vs validation error across tree depth, used to select the depth.</figcaption></figure><figure class="report-chart"><img src="reports/figures/tree_learning_curves.png" alt="Decision tree learning curves" loading="lazy"><figcaption>Training/validation behavior as training sample size grows.</figcaption></figure><figure class="report-chart"><img src="reports/figures/pegasos_objective.png" alt="Pegasos objective by epoch" loading="lazy"><figcaption>Linear Pegasos objective decreases over training epochs.</figcaption></figure><figure class="report-chart"><img src="reports/figures/svm_margin_scores.png" alt="Pegasos signed decision scores" loading="lazy"><figcaption>Signed SVM decision scores around the zero decision boundary.</figcaption></figure></div></section>
 
     <section class="report-section"><h2>6. What is implemented</h2><div class="pipeline"><div><b>Decision tree</b><small>NumPy classifier and regressor; Gini, entropy and MSE; numeric threshold search.</small></div><div><b>Linear SVM</b><small>NumPy Pegasos on regularized hinge loss; support-vector and margin diagnostics.</small></div><div><b>Nonlinear SVM</b><small>Seeded random Fourier map and the same Pegasos solver; 128 components.</small></div><div><b>References</b><small>scikit-learn trees, LinearSVC and RBF SVC for comparative benchmarking.</small></div></div><p style="margin-top:13px">The nonlinear model scored below the linear model in this run. More complexity did not improve the result with the selected sample size, gamma and random-feature count. The implementation still demonstrates the required RBF approximation path and is reported with its weaker performance.</p></section>
 
     <section class="report-section"><h2>7. Honest limitations</h2><div class="method-grid"><article class="method-card"><div class="method-icon icon-rose">!</div><h3>Asking price</h3><p>Listings are seller asking prices, not verified completed transactions. The model does not establish fair market value.</p></article><article class="method-card"><div class="method-icon icon-blue">◷</div><h3>Data age</h3><p>The source is a 2024 scrape. It should not be presented as a current 2026 market estimate.</p></article><article class="method-card"><div class="method-icon icon-lavender">⌕</div><h3>Source-informed filters</h3><p>Fixed price and area windows were chosen after inspecting this dataset's distribution. This is disclosed selection bias.</p></article><article class="method-card"><div class="method-icon icon-lime">∿</div><h3>Generalization</h3><p>Some locations are sparse; cross-ID ambiguous duplicate candidates remain; the CV preprocessing is not fully nested.</p></article></div></section>
 
-    <section class="report-section"><h2>8. Reproduction, tests and presentation plan</h2><div class="report-callout"><strong>Run:</strong> <code>python -m src.explore</code> → <code>python -m src.run_all</code> → <code>pytest -q</code>. The source CSV must be obtained separately and placed at the path documented in <code>data/README.md</code>. The latest recorded automated test run is <b id="test-report">—</b>.</div><div class="roles-grid" style="margin-top:14px"><article class="role-card"><span>SPEAKER 1 · 25% TARGET</span><b>Emiliya Abdulhalimova</b><p>EDA, cleaning, leakage decisions, feature engineering and train/validation/test design.</p></article><article class="role-card"><span>SPEAKER 2 · 25% TARGET</span><b>Aygun Asvarova</b><p>NumPy decision tree, split criteria, hyperparameter tuning, feature importance and tree error analysis.</p></article><article class="role-card"><span>SPEAKER 3 · 25% TARGET</span><b>Murad Bagirzade</b><p>Pegasos SVM, hinge objective, margin diagnostics and random Fourier feature extension.</p></article><article class="role-card"><span>SPEAKER 4 · 25% TARGET</span><b>Kanan Karimli</b><p>Baselines, CV, evaluation, reporting, reproducibility and final integration.</p></article></div><p style="margin-top:11px">The equal split is a proposed presentation/ownership plan, not evidence of each member's completed work. Update the contribution report to match actual work. AI was used for coding assistance, debugging, corrections and review; the team is responsible for understanding and checking the final implementation.</p></section>
+    <section class="report-section"><h2>8. Reproduction, tests and presentation plan</h2><div class="report-callout"><strong>Run:</strong> <code>python -m src.run_all</code> → <code>pytest -q</code>. The source CSV must be obtained separately and placed at the path documented in <code>data/README.md</code>. The latest recorded automated test run is <b id="test-report">—</b>.</div><div class="roles-grid" style="margin-top:14px"><article class="role-card"><span>SPEAKER 1 · 25% TARGET</span><b>Emiliya Abdulhalimova</b><p>Data cleaning, EDA, error analysis.</p></article><article class="role-card"><span>SPEAKER 2 · 25% TARGET</span><b>Aygun Asvarova</b><p>Feature engineering, scaling, model selection.</p></article><article class="role-card"><span>SPEAKER 3 · 25% TARGET</span><b>Murad Bagirzade</b><p>SVM, metrics, data validation, orchestration.</p></article><article class="role-card"><span>SPEAKER 4 · 25% TARGET</span><b>Kanan Karimli</b><p>Decision tree, bagging ensemble.</p></article></div><p style="margin-top:11px">Roles follow contribution_report.md. AI was used for coding assistance, debugging, corrections and review; the team is responsible for understanding and checking the final implementation.</p></section>
 
-    <section class="report-section"><h2>Related project files</h2><div class="bottom-links"><a href="reports/final_report.md"><span>↗</span><b>Full Markdown report</b><small>EDA transcript, metrics, figures and code-guide links</small></a><a href="report/report.pdf"><span>↗</span><b>Typeset PDF</b><small>IEEE-style project summary</small></a><a href="docs/code_explanations/README.md"><span>↗</span><b>Code explanations</b><small>Module-by-module walkthroughs</small></a></div></section>
+    <section class="report-section"><h2>Related project files</h2><div class="bottom-links"><a href="reports/summary.md"><span>↗</span><b>Run summary</b><small>Headline metrics of the last full run</small></a><a href="report/report.pdf"><span>↗</span><b>Typeset PDF</b><small>IEEE-style project summary</small></a><a href="CHANGES.md"><span>↗</span><b>Changes</b><small>What was added and measured</small></a></div></section>
   </main>
-  <footer class="footer"><div class="wrap footer-inner"><div><a class="brand footer-brand" href="index.html"><img src="assets/pegasold-logo.png" alt=""><span class="brand-name">PegaSold <i>//</i></span><span class="brand-caption">PROJECT RESULTS</span></a><p>Results from the 2024 Bina.az listing dataset.</p></div><div class="footer-status"><span class="live-dot"></span><span id="footer-test">Results loaded</span></div><div class="footer-links"><a href="index.html">Estimate</a><a href="reports/final_report.md">Full report</a><a href="presentation/slides.pdf">Slides</a></div></div><div class="wrap footer-bottom"><span>MLE-AI-201 · Team PegaSold</span><span>Historical asking-price data</span></div></footer>
+  <footer class="footer"><div class="wrap footer-inner"><div><a class="brand footer-brand" href="index.html"><img src="assets/pegasold-logo.png" alt=""><span class="brand-name">PegaSold <i>//</i></span><span class="brand-caption">PROJECT RESULTS</span></a><p>Results from the 2024 Bina.az listing dataset.</p></div><div class="footer-status"><span class="live-dot"></span><span id="footer-test">Results loaded</span></div><div class="footer-links"><a href="index.html">Estimate</a><a href="reports/summary.md">Full report</a><a href="presentation/slides.pdf">Slides</a></div></div><div class="wrap footer-bottom"><span>MLE-AI-201 · Team PegaSold</span><span>Historical asking-price data</span></div></footer>
   <a class="back-top" id="back-top" href="#top" aria-label="Back to top">↑</a><div class="toast" id="toast" role="status"></div>
   <script src="site-data.js"></script><script src="app.js"></script>
   <script>document.querySelectorAll('[data-result-panel]').forEach((p)=>{if(p.dataset.resultPanel==='regression')p.hidden=true});</script>
```

## 4. Review fixes
| File | Fix |
|---|---|
| `src/validate.py` | `validation_report.md` said "only 19 typo rows ... winsorization (no rows removed)". It now prints the real count (66) and says the rows are removed, not clipped |
| `src/evaluate.py` | deleted `fit_sklearn_baselines()` (about 100 lines nothing called; `run()` already fits the baselines) |
| `explore.py` | uses `number_from_text` from `src/data_prep.py` instead of a second copy of the same parser |
| `src/experiments.py` | the results page now shows the real test count (`pytest -q` output) instead of "Test summary unavailable" |
| `team_scripts/4_aygun.py` | if `pdflatex` fails, it puts the shipped PDF back instead of leaving a half-written one |

## 5. Tests
- **SVM:** `tests/test_svm.py` is the branch's full SVM suite, plus our lambda-sensitivity, ROC and RFF tests.
- **Decision tree:** `tests/test_decision_tree.py` is the branch's full tree suite, plus a feature-importance test and our real-data scikit-learn checks.
- **Ensemble:** `tests/test_ensemble.py` is one file again.
- **Result:** `python -m pytest tests -q` gives **295 passed**.

## 6. Website
- **Pages:** the branch site, `index.html` (estimator) and `report.html` (results page), is served by `python -m src.web.web_demo`.
- **Report figures:** the server also serves `reports/figures/`, `reports/summary.md`, `report/report.pdf`, `presentation/slides.pdf` and `CHANGES.md`. Every other path outside `src/web` is refused (403).
- **Favicon fix:** static files are sent with `Cache-Control: no-cache` and the icon link has `?v=3`, so the new favicon replaces the cached one.

## 7. Numbers after the changes
Full `python -m src.run_all` (3.2 min, seed 42). Test set unless stated.

| Model | Result |
|---|---|
| Tree, regression | RMSE 0.2445, R2 0.856 (unchanged) |
| Tree, classification | acc 0.9159, AUC 0.9695 (unchanged) |
| Pegasos SVM | acc 0.9084, AUC 0.9698; 11,077 support vectors, margin 0.3141 |
| Bagging | R2 0.886, acc 0.9337, AUC 0.9827 |
| RFF SVM (new) | acc 0.842, AUC 0.921; RBF SVC reference 0.875 / 0.955 |
| 5-fold CV (new) | tree acc 0.909 +/- 0.001 (sklearn 0.909), Pegasos 0.904 +/- 0.002 (LinearSVC 0.904), tree log-RMSE 0.246 +/- 0.003 |

Measured on validation:
- **SVM lambda:** 0.01 -> 0.001 changes accuracy 0.8903 -> 0.9028 and AUC 0.957 -> 0.967.
- **Bagging member depth:** 8 -> 16 changes accuracy 0.9144 -> 0.9285 and OOB 0.9163 -> 0.9291.

## 8. Report and slides
- **Report:** `report/report.tex` with `report/sections/{data,models,results}.tex` is filled with these numbers, including the RFF SVM, cross-validation and the Pegasos projection.
- **Slides:** `presentation/slides.tex` is a Madrid-theme deck in the website palette, with a three-part footer (team, title, date and page) and the logo.
- **Build:**
  - Report: `cd report && pdflatex report.tex` (run twice).
  - Slides: `cd presentation && pdflatex slides.tex` (run twice).
