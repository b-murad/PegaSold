# Contribution Report

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Abdulhalimova, Emiliya | `emiliyabdul` | Data cleaning, EDA, error analysis |
| Asvarova, Aygun | `[handle]` | Feature engineering, scaling, model selection |
| Karimli, Kenan | `kenan-karimli` | Decision tree, bagging ensemble |
| Bagirzade, Murad | `muradb24` | SVM, metrics, data validation, orchestration |

## How the work was divided

Two rules shaped the split:

1. **No file has a single owner.** Every source file is split by function, so
   at least two people worked inside each one.
2. **Nobody evaluates their own work.** Each component was built by one member
   and checked by a different member.

| Member | Builds | Checks someone else's |
|---|---|---|
| Emiliya | raw data → clean rows | the decision tree (prediction path, error analysis) |
| Aygun | clean rows → model matrix | both models (depth and lambda sweeps) |
| Kenan | decision tree + bagging | the SVM (objective, ROC, baselines) |
| Murad | SVM + metrics + pipeline | the whole data pipeline (`validate.py`) |

---

## 1. Data preprocessing — `src/data_prep.py`

| Function | What it does | Owner |
|---|---|---|
| `number_from_text` | pulls numbers out of Azerbaijani text (`"145 m²"` → `145.0`) | Emiliya |
| `fingerprint` | SHA-256 signature of a set of item ids, used to prove the split is reproducible | Emiliya |
| `load` | reads the 141 MB CSV, 100,775 rows × 51 columns | Emiliya |
| `parse` | 51 raw columns → 19 working ones; fixes `Sahə` being m² for flats but sot for land | Emiliya |
| `deduplicate` | removes 36,990 repeat scrapes + 544 reposted twins → 63,241 rows | Emiliya |
| `apply_scope` | keeps Baku and the 3 residential categories → 57,069 rows | Emiliya |
| `apply_filters` | price / area / price-per-m² windows → **57,003 rows** | Emiliya |
| `preprocessing_report.json` | every count and threshold recorded, so no number is typed by hand | Emiliya |
| `split` | 39,902 train / 8,550 validation / 8,551 test, seed 42 | Aygun |
| `fit_preprocessor` | **all statistics measured on the training split only** — median fills, scaling mean and std, the 209,000 AZN tier threshold, the 58 named locations | Aygun |
| `build_features` | **the 82 features** — log transforms, `distance_to_centre_km`, `floor_ratio`, `area_per_room`, `log_land_ratio`, ground/top floor flags, 59 location + 3 category indicators, then scaling | Aygun |
| `save` | writes `X_*.npy`, `y_*_log.npy`, `tier_*.npy` and the clean CSVs | Aygun |
| `build_targets` | `log(price)` for Task A; premium/standard tier for Task B | Murad |
| **leakage removal** | drops `unit_price`, `total_price`, `description` — these contain the answer | Murad |
| `run` / `parse_args` / `main` | stage ordering and CLI | Murad |

## 2. Evaluating the preprocessing — `src/validate.py`

Owned by Murad, who wrote none of the cleaning, so the checks are not written
against the implementation's own assumptions.

| Part | What it checks | Owner |
|---|---|---|
| `Validator` (`__init__`, `group`, `check`) | the PASS/FAIL harness | Murad |
| `sha256_ids` | split identity hashes, so a rerun is provably the same split | Murad |
| `load` | reads back everything `data_prep` wrote | Murad |
| `validate` | splits sum to 57,003; X rows == y rows; `tier == (price >= 209,000)` | Murad |
| split-identity checks | no `item_id` shared between any two splits; split hashes match the report | Murad |
| `write_markdown` | `validation_report.md` | Murad |
| leakage audit | correlation of all 82 features vs target; top is `log_area_m2` at 0.73 | Kenan |
| `make_plots` / `_save` | the `val_*.png` cleanliness figures | Emiliya |
| `reproduce` | re-runs prep and compares hashes, proving determinism | Aygun |

## 3. Exploratory analysis — `explore.py`

| Part | Owner |
|---|---|
| Price distribution and missing-value figures | Emiliya |
| `eda_report.txt` written findings | Emiliya |
| Map of Baku, feature availability | Aygun |
| Export of the 4 figures the LaTeX report includes | Murad |

## 4. Decision tree — `src/decision_tree.py`

| Function | What it does | Owner |
|---|---|---|
| `TreeNode.__init__` | node: split feature, threshold, children, leaf value | Kenan |
| `DecisionTree.__init__` | parameters and their validation | Kenan |
| `fit` | entry point, input guards, class discovery | Kenan |
| `_build_tree` | the recursion that grows the tree | Kenan |
| `_best_split` | **tries every threshold on every feature** at each node | Kenan |
| `_should_stop` | depth cap, minimum rows, pure node | Kenan |
| `_leaf_value` / `_make_leaf` | leaf prediction and stored class counts | Kenan |
| `min_samples_leaf`, `min_impurity_decrease` | the overfitting controls | Kenan |
| `_impurity` — Gini and entropy | classification split measures | Murad |
| `_impurity` — MSE | regression split measure | Murad |
| `_split_score` | size-weighted impurity reduction (the "gain") | Murad |
| correctness test vs scikit-learn | validation RMSE within **0.0005** (0.2383 vs 0.2388) | Murad |
| `predict` / `_predict_one` | walks a row down to its leaf | Emiliya |
| `predict_proba` / `_predict_proba_one` | leaf class frequencies — **this is what makes ROC-AUC meaningful** | Aygun |
| input validation (non-finite, shape, dtype) | error messages | Aygun |

## 5. SVM — `src/svm.py`

| Function | What it does | Owner |
|---|---|---|
| `__init__` | parameters, validation, label mapping to ±1 | Murad |
| `fit` — sub-gradient loop | one listing at a time, nudge the boundary if misclassified | Murad |
| `fit` — learning rate `eta = 1/(lambda*t)` | the Pegasos schedule | Murad |
| `fit` — hinge update, unpenalised intercept | regularisation applies to `w`, not `b` | Murad |
| `_objective` | hinge loss + regularisation, tracked per epoch | Kenan |
| support vectors, `margin_width_` | 11,077 support vectors, margin 0.3141 | Kenan |
| `decision_function` / `predict` | signed distance, then side of the line | Emiliya |
| lambda sensitivity testing | 0.01 → 0.001 gained **+1.3 pp accuracy** | Aygun |

## 6. Bagging ensemble — `src/ensemble.py`

| Function | What it does | Owner |
|---|---|---|
| `BaggedTrees.__init__` | parameters and validation | Kenan |
| `_n_draw` | int row count or float fraction | Kenan |
| `fit` | trains 30 trees on bootstrap samples of 6,000 rows | Kenan |
| `predict` (regression) | mean of member predictions | Kenan |
| `_compute_oob` | **out-of-bag scoring** — each row judged only by trees that never saw it | Murad |
| `predict_proba` (classification) | **soft voting** — average probabilities, not majority vote | Aygun |
| `_aligned_proba` | handles a bootstrap sample that misses a class entirely | Aygun |
| experiment: member depth 8 vs 16 | fixed `BAG_MAX_DEPTH`, worth **+1.4 pp accuracy** | Aygun |
| `_check` | prediction-time guards | Emiliya |
| experiment: feature subsampling vs plain bagging | plain bagging won (recorded in the `ensemble.py` header note; not re-measured in the final pass) | Emiliya |

## 7. Model evaluation — `src/evaluate.py`

| Function | What it does | Owner |
|---|---|---|
| `regression_metrics` | MSE, RMSE, MAE, R², and `exp(MAE)-1` as a percentage | Murad |
| `classification_metrics` | accuracy, ROC-AUC | Murad |
| `tune_svm_threshold` | moves the cut off 0 to where accuracy peaks | Murad |
| `save` (figures), `run` | figure writer and the scoring orchestration | Murad |
| `render_markdown` / `main` | `metrics.json` and `metrics.md` | Murad |
| `load_data`, `subsample` | reads the processed arrays, fixed seeded subsets | Aygun |
| `sweep_depth` | tries depths 2…20, picks on validation | Aygun |
| `sweep_lambda` | tries 4 SVM lambdas | Aygun |
| `_render_ensemble` | the bagging section of the report | Aygun |
| scikit-learn baselines | **matched hyper-parameters**, so the comparison is fair | Kenan |
| `fig_bias_variance` | train vs validation across depth | Kenan |
| `fig_classification` | ROC curves and confusion matrix | Kenan |
| `fig_model_comparison` | ours vs baselines, side by side | Kenan |
| `fig_predictions` | predicted vs actual, residual histogram | Emiliya |
| `fig_error_analysis` | **error by price decile** — where the model fails | Emiliya |

## 8. Pipeline, reproducibility and reporting

| Part | Owner |
|---|---|
| `run_all.py` — `sh`, stage chaining, `copy_report_figures` | Aygun |
| Branch integration into `clean-pipeline` | Aygun |
| `CHANGES.md` | Aygun |
| Slides | Aygun |
| `run_all.py` — `write_summary`, `main` | Murad |
| CLI flags: `--fast`, `--skip-prep`, `--skip-eda`, `--no-ensemble` | Murad |
| IEEE report — results and discussion | Murad |
| Seed fixing; verified repeated runs give identical metrics | Kenan |
| IEEE report — models | Kenan |
| `requirements.txt` pinning | Emiliya |
| `.gitignore` — keeps the 141 MB CSV and `reports/` out of git | Emiliya |
| IEEE report — data sections | Emiliya |

---

## Per-member summary

**Abdulhalimova, Emiliya —** Owned everything from the raw CSV to clean rows:
parsing 51 messy Azerbaijani columns into 19 usable ones, resolving the `Sahə`
unit problem where area is m² for flats but sot for land, and removing 37,534
duplicate scrapes before the scope and domain filters brought 100,775 rows to
57,003. Wrote the exploratory analysis and the error-analysis figure showing the
median error is about 10% in the 100k–350k band but degrades to about 33% above 1M AZN.
Implemented the tree's prediction path and the SVM's decision function, and ran
the experiment showing plain bagging beats feature subsampling on this dataset.

**Asvarova, Aygun —** Built the bridge from clean rows to the model matrix: the
three-way split, the 82 engineered features, and `fit_preprocessor`, where every
statistic — median fills, scaling mean and standard deviation, the 209,000 AZN
tier threshold — is measured on the training split alone so nothing leaks from
validation. Implemented `predict_proba` on the tree and the soft voting in the
ensemble, which together are what make ROC-AUC meaningful. Ran the depth and
lambda sweeps, and found the ensemble depth bug worth 1.4 points of accuracy.

**Karimli, Kenan —** Designed and implemented the decision tree from scratch:
the node structure, the recursion in `_build_tree`, the exhaustive threshold
search in `_best_split`, and the stopping rules that control overfitting. Built
the bagging ensemble — bootstrap sampling and prediction averaging — which cut
regression test MSE by 20.9% (25.9% on validation) and lifted classification AUC from 0.9664 to 0.9808.
Wrote the scikit-learn comparison at matched hyper-parameters showing our tree
matches the reference to 0.0005 RMSE, and verified the pipeline is reproducible
across repeated runs.

**Bagirzade, Murad —** Built the measurement layer the project is judged on: the
regression and classification metrics, ROC-AUC, and the conversion of log error
into a readable percentage. Implemented the Pegasos sub-gradient loop including
the `1/(lambda*t)` schedule and the unpenalised intercept, plus the threshold
tuning that corrects Pegasos optimising hinge loss rather than accuracy. Added
the MSE criterion to the tree and out-of-bag scoring to the ensemble. Owns
`validate.py`, so the data pipeline is checked by someone who did not write it —
which is how the split-integrity checks (no shared `item_id`, matching hashes) are enforced.

---

## Rough effort split

| Member | Approx. % |
|--------|-----------|
| Abdulhalimova, Emiliya | 25% |
| Asvarova, Aygun | 25% |
| Karimli, Kenan | 25% |
| Bagirzade, Murad | 25% |

Even by design. Each member owns one stage end to end, contributes functions
inside the other three stages, and checks one component they did not write, so
both build and review effort are evenly distributed. No imbalance to report.
