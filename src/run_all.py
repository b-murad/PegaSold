"""ONE command reproduces every number and figure.

    python -m src.run_all              # full run
    python -m src.run_all --fast       # smoke test (minutes, not final numbers)
    python -m src.run_all --skip-prep  # reuse data/processed, re-score only

Pipeline: data_prep -> validate -> explore (EDA) -> evaluate -> experiments -> summary.

Every seed is fixed (RANDOM_SEED = 42 in src/data_prep.py and src/evaluate.py),
so two runs on the same CSV produce the same numbers.

Runtime note: the models are our own NumPy code, and our DecisionTree is exact
but slow -- it tries every threshold on every feature at every node. A full run
takes roughly 15-25 minutes, almost all of it inside the final tree fits and
the bagging. `--fast` subsamples everything and finishes in about a minute; it
is for checking that the pipeline works, not for numbers you would report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# report/data_report.tex includes these by the short name on the left;
# explore.py writes them under the name on the right. They are generated
# artifacts that .gitignore keeps out of the repo, so a fresh clone has an
# empty report/figures/ until this step runs.
REPORT_FIGURES = {
    "price_distribution.png": "fig03_price_distribution.png",
    "missing_values.png": "fig01_missing_values.png",
    "map_of_baku.png": "fig09_map_of_baku.png",
    "feature_availability.png": "fig12_feature_availability.png",
}


def sh(*cmd: str) -> None:
    """Run one stage as a subprocess and stop the pipeline if it fails."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def copy_report_figures(eda_out: str) -> None:
    """Put the figures the LaTeX report needs where the LaTeX report expects."""
    source_dir = ROOT / eda_out / "figures"
    target_dir = ROOT / "report" / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for target_name, source_name in REPORT_FIGURES.items():
        source = source_dir / source_name
        if not source.exists():
            print(f"  WARNING: {source} missing; report/data_report.tex wants "
                  f"it as {target_name}")
            continue
        shutil.copyfile(source, target_dir / target_name)
        copied += 1
    print(f"  copied {copied}/{len(REPORT_FIGURES)} figures -> {target_dir}")


def write_summary(processed: str, fig_out: str, elapsed: float) -> None:
    """One page pointing at every artifact, built from the saved reports."""
    prep = json.loads((ROOT / processed / "preprocessing_report.json").read_text())
    m = json.loads((ROOT / fig_out / "metrics.json").read_text())

    def block(key, label):
        d = m.get(key)
        if not d:
            return ""
        if "rmse" in d:
            return (f"| {label} | MSE={d['mse']:.5f} RMSE={d['rmse']:.4f} "
                    f"MAE={d['mae']:.4f} R2={d['r2']:.4f} |\n")
        return (f"| {label} | acc={d['acc']:.4f} "
                f"AUC={d.get('roc_auc', float('nan')):.4f} |\n")

    rows = "".join([
        block("our_tree_regression_valid", "DecisionTree A valid"),
        block("our_tree_regression_test", "DecisionTree A test"),
        block("our_tree_classification_valid", "DecisionTree B valid"),
        block("our_tree_classification_test", "DecisionTree B test"),
        block("our_svm_valid", "PegasosSVM B valid"),
        block("our_svm_test", "PegasosSVM B test"),
        block("our_bagged_regression_valid", "BaggedTrees A valid"),
        block("our_bagged_regression_test", "BaggedTrees A test"),
        block("our_bagged_classification_valid", "BaggedTrees B valid"),
        block("our_bagged_classification_test", "BaggedTrees B test"),
    ])

    text = (
        "# Run summary\n\n"
        "Reproduced by `python -m src.run_all`"
        f"{'  **(--fast: numbers are not final)**' if m.get('fast_mode') else ''} "
        f"in {elapsed / 60:.1f} min.\n\n"
        "## Data\n\n"
        f"- Cleaned rows: {prep['final_rows']:,} (raw {prep['raw_rows']:,}, "
        f"dedup -{prep['rows_removed_by_item_deduplication']:,}, "
        f"scope -{prep['scope']['rows_removed_by_scope']:,}, "
        f"domain -{prep['domain_filters']['total_removed_by_domain_filters']:,})\n"
        f"- Split: {prep['split_rows']}\n"
        f"- Features: {prep['n_processed_features']}\n"
        f"- Tier threshold: {prep['tier_threshold_AZN']:,.0f} AZN\n\n"
        "## Our models\n\n"
        f"Selected: tree depth {m.get('best_depth_regression')} (regression), "
        f"{m.get('best_depth_classification')} (classification), "
        f"SVM lambda={m.get('best_lambda_svm')}.\n\n"
        "| model | result |\n|---|---|\n" + rows + "\n"
        "Artifacts: `reports/eda/eda_report.txt`, "
        "`reports/eda/validation_report.md`, `reports/figures/fig1-5_*.png`, "
        "`reports/figures/metrics.json`, `reports/figures/metrics.md`.\n"
    )
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "summary.md").write_text(text, encoding="utf-8")
    print(f"  wrote -> {ROOT / 'reports' / 'summary.md'}")


def main() -> int:
    p = argparse.ArgumentParser(description="Reproduce every number and figure.")
    p.add_argument("--data", default="data/bina_az_sale.csv")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--eda-out", default="reports/eda")
    p.add_argument("--fig-out", default="reports/figures")
    p.add_argument("--fast", action="store_true",
                   help="subsample everything; pipeline check, not final numbers")
    p.add_argument("--skip-prep", action="store_true",
                   help="reuse an existing data/processed instead of rebuilding")
    p.add_argument("--skip-eda", action="store_true",
                   help="skip explore.py (the slowest non-model step)")
    p.add_argument("--no-ensemble", action="store_true",
                   help="skip bagging (saves most of the model runtime)")
    p.add_argument("--skip-experiments", action="store_true",
                   help="skip src.experiments (RFF SVM, cross-validation, web report data)")
    a = p.parse_args()
    start = time.time()

    raw = ROOT / a.data
    if not a.skip_prep and not raw.exists():
        print(f"ERROR: {raw} not found.\n"
              f"Download the dataset into data/ (see data/README.md), or pass "
              f"--skip-prep to reuse an existing {a.processed}/.", file=sys.stderr)
        return 1

    # 1. clean + split + model-ready arrays (seeds fixed in src/data_prep.py)
    if a.skip_prep:
        print(f"\n[skip-prep] reusing {a.processed}/")
    else:
        sh(sys.executable, "-m", "src.data_prep",
           "--input", a.data, "--output-dir", a.processed)

    # 2. integrity checks + cleanliness plots + validation_report.md
    sh(sys.executable, "-m", "src.validate", "--dir", a.processed, "--out", a.eda_out)

    # 3. EDA report + figures (read-only, never modifies data/)
    if a.skip_eda or not (ROOT / "explore.py").exists():
        print("\n[skip-eda] explore.py not run")
    else:
        sh(sys.executable, "explore.py", "--data", a.data, "--out", a.eda_out)
        print("\n$ copy report figures")
        copy_report_figures(a.eda_out)

    # 4. our models: selection, final fits, metrics, figures
    cmd = [sys.executable, "-m", "src.evaluate",
           "--dir", a.processed, "--out", a.fig_out]
    if a.fast:
        cmd.append("--fast")
    if a.no_ensemble:
        cmd.append("--no-ensemble")
    sh(*cmd)

    # 5. extra experiments from the decision-tree branch + web report data
    if a.skip_experiments or a.fast:
        print("\n[skip-experiments] src.experiments not run")
    else:
        sh(sys.executable, "-m", "src.experiments")

    # 6. one-page summary pointing at the artifacts
    print("\n$ write summary")
    write_summary(a.processed, a.fig_out, time.time() - start)

    print(f"\nAll numbers/figures reproduced in {(time.time() - start) / 60:.1f} min. "
          f"See reports/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
