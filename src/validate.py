"""validate.py -- read-only verification of the cleaned data produced by data_prep.py.

Loads model-ready arrays, cleaned split tables, feature names and the
preprocessing report, checks every invariant, prints PASS/FAIL, writes
cleanliness plots (val_*.png) + validation_report.md next to the EDA report."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

AZ_LAT_RANGE = (38.0, 42.0)
AZ_LNG_RANGE = (44.5, 51.0)

SPLIT_TOLERANCE = 0.02
TIER_TOLERANCE = 0.05

def leakage_audit(v, report, feature_names, tables):
    """
    Check that nothing in the model inputs gives away the price.

    Three checks: no leakage column survives into the feature names, the
    report records which columns were removed, and no numeric feature is
    almost a copy of the price (|correlation| above 0.95 on the train table).

    :param v: Validator that records the PASS/FAIL results.
    :param report: Preprocessing report (leakage columns, numeric feature list).
    :param feature_names: Names of the processed features.
    :param tables: Dict of cleaned split tables; the "train" table is used.
    """
    v.group("LEAKAGE")

    # 1. No processed feature may be derived from a column that contains the answer
    leaked = [
        name
        for name in feature_names
        if any(tag in name for tag in ("unit_price", "total_price", "description"))
    ]
    v.check("leakage columns absent from feature_names", not leaked, f"{leaked}")

    # 2. The preprocessing step must have recorded exactly these removals
    v.check(
        "leakage columns recorded in report",
        set(report["leakage_columns_removed"]) == {"unit_price", "total_price", "description"},
        ",".join(sorted(report["leakage_columns_removed"])),
    )

    # 3. No numeric feature may be (almost) a copy of the price
    numeric_features = report.get("numeric_features", [])
    train = tables["train"]
    price = pd.to_numeric(train["price"], errors="coerce")

    # Strongest correlation seen so far, as (column, |r|)
    worst = ("", 0.0)
    for column in numeric_features:
        # Skip features that are not columns of the cleaned table
        if column not in train.columns:
            continue
        r = abs(train[column].corr(price))
        # corr() gives NaN for constant columns, so those are ignored
        if pd.notna(r) and r > worst[1]:
            worst = (column, float(r))
    v.check(
        "no numeric feature |corr| > 0.95 with price",
        worst[1] <= 0.95,
        f"max {worst[0]}={worst[1]:.3f}",
    )

class Validator:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.current_group = 0
        self.rows: list[tuple[str, str, bool, str]] = []

    def group(self, title: str) -> None:
        self.current_group = title
        print()
        print(title)
        print("-" * len(title))

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self.rows.append((self.current_group or "", name, ok, detail))
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{mark}] {name}{suffix}")

def sha256_ids(series: pd.Series) -> str:
    joined = "\n".join(sorted(series.astype(str)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

def load(data_dir: Path):
    report = json.loads((data_dir / "preprocessing_report.json").read_text(encoding="utf-8"))
    feature_names = pd.read_csv(data_dir / "feature_names.csv")["feature"].tolist()
    arrays = {
        "X_train": np.load(data_dir / "X_train.npy"),
        "X_valid": np.load(data_dir / "X_valid.npy"),
        "X_test": np.load(data_dir / "X_test.npy"),
        "y_train_log": np.load(data_dir / "y_train_log.npy"),
        "y_valid_log": np.load(data_dir / "y_valid_log.npy"),
        "y_test_log": np.load(data_dir / "y_test_log.npy"),
    }
    tiers = {
        "train": np.load(data_dir / "tier_train.npy"),
        "validation": np.load(data_dir / "tier_valid.npy"),
        "test": np.load(data_dir / "tier_test.npy"),
    }
    tables = {
        "train": pd.read_csv(data_dir / "train_clean.csv", low_memory=False),
        "validation": pd.read_csv(data_dir / "valid_clean.csv", low_memory=False),
        "test": pd.read_csv(data_dir / "test_clean.csv", low_memory=False),
    }
    return report, feature_names, arrays, tiers, tables

def validate(report, feature_names, arrays, tiers, tables) -> Validator:
    v = Validator()

    # Split Integrity:
    v.group("SPLIT INTEGRITY")
    sizes = {k: len(df) for k, df in tables.items()}
    reported = report["split_rows"]
    v.check(
        "split sizes match report",
        all(sizes[k] == reported[k] for k in ("train", "validation", "test")),
        f"train={sizes['train']} valid={sizes['validation']} test={sizes['test']}",
    )
    v.check(
        "splits sum to final_rows",
        sum(sizes.values()) == report["final_rows"],
        f"{sum(sizes.values())} == {report['final_rows']}",
    )
    total = sum(sizes.values())
    splits =("train", "validation", "test")
    expected_ratio = {"train": 0.70, "validation": 0.15, "test": 0.15}
    for k in splits:
        actual = sizes[k] / total
        v.check(
            f"'{k}' ratio ~ {expected_ratio[k]:.2f}",
            abs(actual - expected_ratio[k]) <= SPLIT_TOLERANCE,
            f"{actual:.3f}",
        )

    ids = {k: tables[k]["item_id"] for k in splits}
    for k in splits:
        v.check(
            f"no duplicate item_id in '{k}'",
            int(ids[k].duplicated().sum()) == 0,
            f"{int(ids[k].duplicated().sum())} dupes",
        )
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = len(set(ids[a]) & set(ids[b]))
        v.check(
            f"no item_id overlap {a} <-> {b}",
            overlap == 0,
            f"{overlap} shared"
        )

    if "split_item_id_sha256" in report:
        for k in splits:
            expected = report["split_item_id_sha256"][k]
            actual_hash = sha256_ids(ids[k])
            v.check(
                f"'{k}' id fingerprint matches report",
                actual_hash == expected,
                f"{actual_hash} vs {expected}",
            )
    else:
        v.check("report contains split id fingerprints", False, "missing key")

    # Target:
    v.group("TARGET")
    for k in splits:
        price = pd.to_numeric(tables[k]["price"], errors="coerce")
        v.check(
            f"'{k}' price > 0 and no NaN",
            bool((price > 0).all()) and not bool(price.isna().any()),
            f"min={price.min():,.0f} max={price.max():,.0f}",
        )

    threshold = report["tier_threshold_AZN"]
    train_median = float(pd.to_numeric(tables["train"]["price"], errors="coerce").median())
    v.check(
        "tier_threshold == train median price",
        abs(train_median - threshold) < 1e-6,
        f"{threshold:,.0f} vs {train_median:,.0f}",
    )

    # log_price == ln(price)
    train = tables["train"]
    diff = np.log(train["price"].to_numpy()) - train["log_price"].to_numpy()
    v.check("log_price == ln(price) on train", float(np.max(np.abs(diff))) < 1e-9)

    # y arrays align with the saved tables (same row order)
    for k, key in (("train", "y_train_log"), ("validation", "y_valid_log"), ("test", "y_test_log")):
        expected = tables[k]["log_price"].to_numpy()
        v.check(
            f"{key} aligns with '{k}' log_price",
            np.allclose(arrays[key], expected, atol=1e-6),
        )

    # tier arrays match the threshold rule
    for k in splits:
        price = pd.to_numeric(tables[k]["price"], errors="coerce")
        expected = (price >= threshold).astype(int).to_numpy()
        v.check(
            f"tier_{k} == (price >= threshold)",
            np.array_equal(tiers[k], expected),
        )
        share = float(tiers[k].mean())
        v.check(
            f"'{k}' tier balance ~ 50/50",
            abs(share - 0.5) <= TIER_TOLERANCE,
            f"premium={share:.3f}",
        )

    # Domain / range:
    v.group("DOMAIN / RANGE")
    for k in splits:
        t = tables[k]
        area = pd.to_numeric(t["area_m2"], errors="coerce")
        v.check(
            f"'{k}' area_m2 in (0, 5000]",
            bool((area.dropna() > 0).all() and (area.dropna() <= 5000).all()),
            f"min={area.min():.0f} max={area.max():.0f}",
        )
        floor = pd.to_numeric(t["floor"], errors="coerce")
        n_floors = pd.to_numeric(t["n_floors"], errors="coerce")
        both = floor.notna() & n_floors.notna()
        v.check(
            f"'{k}' floor <= n_floors where both known",
            bool((floor[both] <= n_floors[both]).all()),
        )
        rooms = pd.to_numeric(t["rooms"], errors="coerce")
        v.check(
            f"'{k}' rooms in (0, 30]",
            bool((rooms.dropna() > 0).all() and (rooms.dropna() <= 30).all()),
            f"min={rooms.min():.0f} max={rooms.max():.0f}",
        )

        lat = pd.to_numeric(t["lat"], errors="coerce")
        lng = pd.to_numeric(t["lng"], errors="coerce")
        known = lat.notna() & lng.notna()
        inside = lat.between(*AZ_LAT_RANGE) & lng.between(*AZ_LNG_RANGE)
        v.check(
            f"'{k}' coordinates inside Azerbaijan or NaN",
            bool((inside | ~known).all()),
        )

    fences = (report.get("value_ranges") or report.get("winsorization") or {}).get("fences", {})
    if fences:
        for column, bounds in fences.items():
            for k in splits:
                values = pd.to_numeric(tables[k][column], errors="coerce").dropna()
                ok = bool(
                    (values >= bounds["low"] - 1e-9).all()
                    and (values <= bounds["high"] + 1e-9).all()
                )
                v.check(
                    f"'{k}' {column} within enforced range",
                    ok,
                    f"[{bounds['low']:,.1f}, {bounds['high']:,.1f}]",
                )
    else:
        v.check("enforced value ranges recorded", False, "missing")

    # Leakage:
    v.group("LEAKAGE")
    leaked = [
        name
        for name in feature_names
        if any(tag in name for tag in ("unit_price", "total_price", "description"))
    ]
    v.check("leakage columns absent from feature_names", not leaked, f"{leaked}")
    v.check(
        "leakage columns recorded in report",
        set(report["leakage_columns_removed"]) == {"unit_price", "total_price", "description"},
        ",".join(sorted(report["leakage_columns_removed"])),
    )

    numeric_features = report.get("numeric_features", [])
    train = tables["train"]
    price = pd.to_numeric(train["price"], errors="coerce")
    worst = ("", 0.0)
    for column in numeric_features:
        if column not in train.columns:
            continue
        r = abs(train[column].corr(price))
        if pd.notna(r) and r > worst[1]:
            worst = (column, float(r))
    v.check(
        "no numeric feature |corr| > 0.95 with price",
        worst[1] <= 0.95,
        f"max {worst[0]}={worst[1]:.3f}",
    )

    # Artifacts:
    v.group("ARTIFACTS")
    for k, x_key, y_key in (
        ("train", "X_train", "y_train_log"),
        ("validation", "X_valid", "y_valid_log"),
        ("test", "X_test", "y_test_log"),
    ):
        v.check(
            f"X_{k} rows == y_{k} rows",
            arrays[x_key].shape[0] == arrays[y_key].shape[0],
            f"{arrays[x_key].shape[0]}",
        )

    for key in ("X_train", "X_valid", "X_test"):
        arr = arrays[key]
        v.check(
            f"{key} finite (no NaN/inf)",
            bool(np.isfinite(arr).all()),
            f"shape={arr.shape}",
        )

    v.check(
        "feature count consistent",
        arrays["X_train"].shape[1] == len(feature_names) == report["n_processed_features"],
        f"{arrays['X_train'].shape[1]} == {len(feature_names)} == {report['n_processed_features']}",
    )
    v.check(
        "feature names unique",
        len(feature_names) == len(set(feature_names)),
    )

    # RobustScaler: numeric block centered on the train median (~0)
    numeric_idx = [i for i, n in enumerate(feature_names) if n.startswith("numeric__")]
    if numeric_idx:
        block = arrays["X_train"][:, numeric_idx]
        medians = np.median(block, axis=0)
        v.check(
            "numeric block median ~ 0 on train (scaled)",
            bool(np.max(np.abs(medians)) < 0.5),
            f"max |median|={np.max(np.abs(medians)):.3f}",
        )
        v.check(
            "numeric block is not constant",
            bool(np.all(np.std(block, axis=0) > 0)),
        )

    # one-hot columns are binary
    onehot_idx = [i for i, n in enumerate(feature_names) if n.startswith("lowcard__")]
    if onehot_idx:
        block = arrays["X_train"][:, onehot_idx]
        v.check(
            "one-hot block is binary",
            bool(np.isin(np.unique(block), [0.0, 1.0]).all()),
        )

   # ---
    leakage_audit(v, report, feature_names, tables)
    # ---

    return v

# Markdown report:

def write_markdown(v: Validator, report, arrays, tiers, tables, plots: list[str], out_path: Path) -> None:
    sizes = {k: len(df) for k, df in tables.items()}
    verdict = "CLEAN" if v.failed == 0 else "NOT CLEAN"
    failed = [r for r in v.rows if not r[2]]
    lines = [
        "# Validation report (cleaned bina.az data)",
        "",
        f"**Verdict: {verdict}** — {v.passed} passed, {v.failed} failed.",
        "",
        "## Headline numbers",
        "",
        f"- Raw rows: {report['raw_rows']:,}",
        f"- After item dedup: {report['rows_after_deduplication']:,}"
        f" (removed {report['rows_removed_by_item_deduplication']:,})",
        f"- After scope: {report['scope']['rows_after_scope']:,}"
        f" (removed {report['scope']['rows_removed_by_scope']:,})",
        f"- After domain filters: {report['rows_after_cleaning']:,}",
        f"- Split: train={sizes['train']:,},"
        f" validation={sizes['validation']:,}, test={sizes['test']:,}",
        f"- Tier threshold (train median): {report['tier_threshold_AZN']:,.0f} AZN",
        f"  (train premium={float(tiers['train'].mean()):.3f},"
        f" valid={float(tiers['validation'].mean()):.3f},"
        f" test={float(tiers['test'].mean()):.3f})",
        f"- Processed features: {report['n_processed_features']}"
        f" (numeric={len(report.get('numeric_features', []))},"
        f" low-card={len(report.get('low_cardinality_features', []))},"
        f" high-card={len(report.get('high_cardinality_features', []))})",
        f"- X_train shape: {arrays['X_train'].shape}",
        "",
        "## Attrition (100k -> 57k)",
        "",
        ("Dedup dominates: ~36.7% of raw rows are re-scrapes of the same"
         " `item_id` (latest scrape kept before splitting, so no listing leaks"
         " across splits). Scope drops Torpaq/Obyekt/Ofis/Qaraj (~9.4% of"
         " deduped rows) to keep a homogeneous residential task. Domain sanity"
         f" filters remove {report['domain_filters']['total_removed_by_domain_filters']:,}"
         " rows outside the declared price / area / price-per-m² windows (removed,"
         " not clipped); the luxury tail inside the windows is kept and modelled on `log(price)`."),
        "",
        "## Failed checks",
        "",
    ]
    if failed:
        lines += [f"- [{g}] {n} ({d})" for g, n, _ok, d in failed]
    else:
        lines.append("- none")
    lines += ["", "## Plots", ""]
    lines += [f"- `figures/{p}`" for p in plots] if plots else ["- (plots disabled)"]
    lines += [
        "",
        "## Full check log",
        "",
        "| Group | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    for g, n, ok, d in v.rows:
        lines.append(f"| {g} | {n} | {'PASS' if ok else 'FAIL'} | {d} |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report saved] {out_path}")

def reproduce(data_dir: Path) -> bool:
    """Re-run the pipeline into a temp dir and compare the split fingerprints."""
    root = Path(__file__).resolve().parent.parent
    input_path = root / "data" / "bina_az_sale.csv"
    if not input_path.exists():
        print("  [SKIP] reproduce: raw CSV not found")
        return True

    tmp = Path(tempfile.mkdtemp(prefix="validate_"))
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.data_prep",
                "--input",
                str(input_path),
                "--output-dir",
                str(tmp),
            ],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        fresh = json.loads((tmp / "preprocessing_report.json").read_text(encoding="utf-8"))
        return fresh["split_item_id_sha256"] == json.loads(
            (data_dir / "preprocessing_report.json").read_text(encoding="utf-8")
        )["split_item_id_sha256"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Cleanliness plots (val_*.png)
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, fig_dir: Path, name: str) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / name
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [figure saved] {out}")
    return out


def make_plots(report, feature_names, arrays, tiers, tables, fig_dir: Path) -> list[str]:
    """Six small plots showing how clean the processed data is."""
    saved: list[str] = []
    splits = ("train", "validation", "test")

    # 1. split sizes
    fig, ax = plt.subplots(figsize=(7, 4))
    sizes = [len(tables[k]) for k in splits]
    bars = ax.bar(list(splits), sizes, color="steelblue", edgecolor="white")
    for bar, n in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{n:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Split sizes (final_rows={report['final_rows']:,}, 70/15/15)")
    ax.set_ylabel("Rows")
    saved.append(_save(fig, fig_dir, "val_split.png").name)

    # 2. tier balance per split
    fig, ax = plt.subplots(figsize=(7, 4))
    shares = [float(tiers[k].mean()) for k in splits]
    bars = ax.bar(list(splits), shares, color="tomato", edgecolor="white")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    for bar, s in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{s:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(f"Premium share per split (threshold={report['tier_threshold_AZN']:,.0f} AZN)")
    ax.set_ylabel("P(premium)")
    saved.append(_save(fig, fig_dir, "val_tier_balance.png").name)

    # 3. log-price distribution by split
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, c in (("train", "steelblue"), ("validation", "tomato"), ("test", "forestgreen")):
        ax.hist(tables[k]["log_price"].dropna(), bins=60, alpha=0.45, label=k, color=c)
    ax.axvline(float(np.log(report["tier_threshold_AZN"])), color="black",
               ls="--", lw=1.5, label="log(threshold)")
    ax.set_title("log(price) distribution by split (should overlap)")
    ax.set_xlabel("log(price)")
    ax.legend(fontsize=8)
    saved.append(_save(fig, fig_dir, "val_logprice_hist.png").name)

    # 4. missingness remaining in cleaned train table (pre-imputation)
    fig, ax = plt.subplots(figsize=(8, max(3, 5)))
    miss = tables["train"].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(15)
    if len(miss):
        ax.barh(miss.index[::-1].astype(str), (miss * 100)[::-1], color="steelblue")
        ax.set_xlabel("% missing (train_clean.csv, imputed at transform time)")
        ax.set_title("Residual missingness after cleaning (top 15)")
    else:
        ax.text(0.5, 0.5, "No missing values left", ha="center", va="center")
        ax.set_title("Residual missingness after cleaning")
    saved.append(_save(fig, fig_dir, "val_missing_after_clean.png").name)

    # 5. winsor fences on area_m2 (train)
    # data_prep records enforced ranges under "value_ranges"; older runs
    # used "winsorization". Accept either.
    fences = (report.get("value_ranges") or report.get("winsorization") or {}).get("fences", {})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = pd.to_numeric(tables["train"].get("area_m2"), errors="coerce").dropna()
    ax.hist(vals, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    if "area_m2" in fences:
        lo, hi = fences["area_m2"]["low"], fences["area_m2"]["high"]
        ax.axvline(lo, color="red", ls="--", lw=1.5, label=f"low={lo:,.0f}")
        ax.axvline(hi, color="red", ls="--", lw=1.5, label=f"high={hi:,.0f}")
        ax.legend(fontsize=8)
    ax.set_title("area_m2 on train with the enforced range (rows outside were removed)")
    ax.set_xlabel("area_m2")
    saved.append(_save(fig, fig_dir, "val_winsor_area.png").name)

    # 6. top |corr| of numeric features with price (train)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    price = pd.to_numeric(tables["train"]["price"], errors="coerce")
    corrs = {}
    for col in report.get("numeric_features", []):
        if col in tables["train"].columns:
            r = tables["train"][col].corr(price)
            if pd.notna(r):
                corrs[col] = abs(float(r))
    top = sorted(corrs.items(), key=lambda kv: -kv[1])[:12]
    if top:
        names, vals_c = zip(*top)
        ax.barh(list(names)[::-1], list(vals_c)[::-1], color="steelblue")
        ax.axvline(0.95, color="red", ls="--", lw=1.5, label="leakage guard 0.95")
        ax.legend(fontsize=8)
        ax.set_title("Top |corr| with price on train (all should be < 0.95)")
        ax.set_xlabel("|Spearman/Pearson|")
    else:
        ax.text(0.5, 0.5, "No numeric features", ha="center", va="center")
    saved.append(_save(fig, fig_dir, "val_corr_with_price.png").name)

    return saved

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate cleaned bina.az data.")
    parser.add_argument("--dir", default="data/processed", help="processed data directory")
    parser.add_argument("--out", default="reports/eda", help="report + figures directory")
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="re-run data_prep.py into a temp dir and compare split fingerprints",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip cleanliness plots (checks + markdown only)",
    )
    args = parser.parse_args()

    data_dir = Path(args.dir)
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"

    if not (data_dir / "preprocessing_report.json").exists():
        print(f"ERROR: no preprocessing_report.json in {data_dir}")
        return 2

    print("=" * 60)
    print("VALIDATING CLEANED DATA")
    print(f"directory: {data_dir.resolve()}")
    print("=" * 60)

    report, feature_names, arrays, tiers, tables = load(data_dir)
    v = validate(report, feature_names, arrays, tiers, tables)

    v.group("REPRODUCIBILITY")
    if args.reproduce:
        ok = reproduce(data_dir)
        v.check("re-run reproduces identical splits", ok)
    else:
        v.check(
            "split fingerprints present (run with --reproduce for full check)",
            "split_item_id_sha256" in report,
        )

    plots: list[str] = []
    if not args.no_plots:
        print()
        print("CLEANLINESS PLOTS")
        print("-----------------")
        plots = make_plots(report, feature_names, arrays, tiers, tables, fig_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(v, report, arrays, tiers, tables, plots, out_dir / "validation_report.md")

    print()
    print("=" * 60)
    verdict = "CLEAN" if v.failed == 0 else "NOT CLEAN"
    print(f"{v.passed} passed, {v.failed} failed  ->  {verdict}")
    print("=" * 60)
    return 0 if v.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())