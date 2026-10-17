"""
Exploratory Data Analysis for the bina.az sale dataset.

WHAT THIS FILE IS FOR
---------------------
This script only *looks* at the raw CSV. It never changes it and never writes
anything into data/. Its job is to answer, with numbers and pictures, the
questions we must settle before writing src/data_prep.py:

  1. How big is the data and what is actually in each column?
  2. Which values are missing, and does "missing" really mean "missing"?
  3. Are there duplicate rows, and why?
  4. Which columns leak the answer (are the target in disguise)?
  5. What does the target `price` look like? Does it need a log transform?
  6. Which columns look useful as features, and how do they relate to price?
  7. What is obviously wrong (impossible prices, impossible coordinates, ...)?
  8. For Task B, how balanced are the two price tiers?

Sections 11-13 then settle the three questions that cannot be answered by
describing the data alone, and that src/data_prep.py has to act on:

  11. SCOPE          which rows belong to the task at all?
  12. DEDUPLICATION  one row per property, but which copy?
  13. PRICE WINDOW   where do we cut impossible prices?

Each of those ends with a RECOMMENDATION line stating what the evidence
supports, so the report can quote it.

Everything here is deliberately simple: counts, percentages, medians,
quantiles, histograms, bar charts, boxplots, scatter plots and Pearson
correlation. No modelling, no feature selection, no dimensionality reduction.

HOW TO RUN
----------
    python explore.py --data data/bina_az_sale.csv --out reports/eda

Outputs:
    <out>/eda_report.txt     every number printed below, saved as text
    <out>/figures/fig*.png   the figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_prep import number_from_text

# --------------------------------------------------------------------------
# Colours. One fixed set, used the same way in every figure, so that a colour
# always means the same thing. Taken from a colour-blind-safe palette.
# --------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT = "#0b0b0b", "#52514e"
GRID = "#e3e1dc"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]  # negative -> neutral -> positive

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "font.size": 9,
    "figure.dpi": 130,
})

# --------------------------------------------------------------------------
# How we group the 51 raw columns. This grouping is a *hypothesis* that the
# report below tests; data_prep.py will act on whatever we confirm here.
# --------------------------------------------------------------------------
TARGET = "price"

# The advert's page address, e.g. "/items/4521724". Used in sections 11-13.
ADVERT_ID = "estate_rel_url_x"
APARTMENTS = ["Yeni tikili", "Köhnə tikili"]
RESIDENTIAL = APARTMENTS + ["Həyət evi/Bağ evi"]

# Columns that identify a row or record when it was scraped. They carry no
# information about how much a flat is worth.
ID_AND_BOOKKEEPING = [
    "id_x", "id_y", "estate_id", "estate_details_id_x", "estate_details_id_y",
    "rel_url", "estate_rel_url_x", "estate_rel_url_y", "estate_rel_url", "img_url",
    "datetime_scrape_x", "datetime_scrape_y", "day_x", "hour_x", "day_y", "hour_y",
    "city_when", "updated", "currency_x", "currency_y",
]

# Columns we suspect are the target in disguise. Checked in section 4.
SUSPECTED_LEAKAGE = ["total_price", "unit_price"]

# Free text. Usable only with NLP, which is out of scope for this project.
FREE_TEXT = ["description", "address", "attributes", "extra_info", "owner_name", "shop_name"]

# Columns that look like real, usable features.
NUMERIC_CANDIDATES = ["Sahə", "Otaq sayı", "Mərtəbə", "Torpaq sahəsi", "lat", "lng", "views"]
CATEGORICAL_CANDIDATES = [
    "Kateqoriya", "location", "city", "Təmir", "Çıxarış", "İpoteka",
    "owner_title", "products_label", "shop_title", "Binanın növü",
]
# Columns where the value is only ever present or absent ("flag" columns).
PRESENCE_FLAGS = ["vip", "featured", "repair", "bill_of_sale", "mortgage"]

# Azerbaijan's real bounding box, used to spot impossible coordinates.
AZ_LAT, AZ_LNG = (38.3, 41.95), (44.7, 50.6)


# ==========================================================================
# A tiny helper so that everything we print also lands in the report file.
# ==========================================================================
class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def header(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def table(self, frame: pd.DataFrame | pd.Series | str) -> None:
        self(frame if isinstance(frame, str) else frame.to_string())

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ==========================================================================
# Light parsers. The raw file stores numbers inside text, e.g. "145 m²" or
# "7 / 9". We unpack them here ONLY so the plots can be drawn. The real,
# tested versions belong in src/data_prep.py.
# ==========================================================================
parse_number_from_text = number_from_text   # one parser, shared with the pipeline


def parse_floor(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'7 / 9' -> (floor 7, building height 9)."""
    parts = series.astype("string").str.extract(r"(\d+)\s*/\s*(\d+)")
    return (
        pd.to_numeric(parts[0], errors="coerce"),
        pd.to_numeric(parts[1], errors="coerce"),
    )


def build_exploration_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add the parsed numeric columns used by the figures."""
    out = df.copy()
    out["area_m2"] = parse_number_from_text(out["Sahə"])
    out["land_sot"] = parse_number_from_text(out["Torpaq sahəsi"])
    out["unit_price_azn_m2"] = parse_number_from_text(out["unit_price"])
    out["floor"], out["building_floors"] = parse_floor(out["Mərtəbə"])
    out["rooms"] = pd.to_numeric(out["Otaq sayı"], errors="coerce")
    # Scrape date, stored in the raw file as dd.mm.yyyy text. Sections 11-13
    # need it to tell the copies of one advert apart.
    out["scraped"] = pd.to_datetime(out["day_x"], format="%d.%m.%Y", errors="coerce")
    out["item_id"] = out[ADVERT_ID].astype("string").str.extract(r"(\d+)", expand=False)
    return out


def latest_listing_per_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return the same one-row-per-ad snapshot used by data_prep.

    Scope and price-window EDA must describe the same observation that the
    modelling pipeline will keep. Selecting the first raw row here caused the
    EDA row counts to drift when a listing changed category/city over time.
    Rows without an extractable advert ID are retained for transparent audit.
    """
    identified = df[df["item_id"].notna()].sort_values(
        ["item_id", "scraped"], kind="mergesort"
    ).drop_duplicates(subset=["item_id"], keep="last")
    unidentified = df[df["item_id"].isna()]
    return pd.concat([identified, unidentified]).sort_index()


def save(fig: plt.Figure, out_dir: Path, name: str, rep: Report) -> None:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    rep(f"    [figure] {path}")


def style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """One place for the look of every axis: recessive grid, no box."""
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=INK_SOFT)
    ax.set_ylabel(ylabel, color=INK_SOFT)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def money(value: float) -> str:
    return f"{value:,.0f}"


def compact(value: float) -> str:
    """1234567 -> '1.2M'. Keeps crowded axes readable."""
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= size:
            text = f"{value / size:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{value:,.0f}"


# ==========================================================================
# 1. What is in the file?
# ==========================================================================
def section_overview(df: pd.DataFrame, rep: Report) -> None:
    rep.header("1. OVERVIEW — how big is the data and what are the columns?")
    rep(f"Rows:    {len(df):,}")
    rep(f"Columns: {df.shape[1]}")
    rep(f"Memory:  {df.memory_usage(deep=True).sum() / 1e6:,.0f} MB in RAM")
    rep("")
    rep("Every column, its type, how much is missing, and how many distinct")
    rep("values it holds. A column with 1 distinct value carries no information;")
    rep("a column with ~1 distinct value per row is an identifier.")
    rep("")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_%": (df.isna().mean() * 100).round(1),
        "n_unique": df.nunique(),
        "example": [df[c].dropna().astype(str).head(1).tolist()[:1] for c in df.columns],
    })
    summary["example"] = summary["example"].apply(
        lambda v: (v[0][:45] + "…") if v and len(v[0]) > 45 else (v[0] if v else "")
    )
    rep.table(summary)

    constant = summary.index[summary["n_unique"] <= 1].tolist()
    identifier = summary.index[summary["n_unique"] >= 0.98 * len(df)].tolist()
    rep("")
    rep(f"Constant columns (one value only, useless as features): {constant}")
    rep(f"Identifier-like columns (nearly unique per row):        {identifier}")


# ==========================================================================
# 2. Missing values — and the trap hiding in them
# ==========================================================================
def section_missing(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("2. MISSING VALUES — and whether 'missing' really means 'unknown'")
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    rep("Percentage missing per column (highest first, only columns > 0):")
    rep.table(missing[missing > 0].round(1))

    rep("")
    rep("THE TRAP. Some columns are never written as 'no'. The scraper only")
    rep("records them when the answer is 'yes', so NaN means 'no', not 'unknown'.")
    rep("Check: every one of these columns has exactly ONE distinct value.")
    rep("")
    for col in PRESENCE_FLAGS + ["İpoteka"]:
        values = df[col].dropna().unique().tolist()
        rep(f"  {col:<14} distinct non-missing values = {values}"
            f"   present in {df[col].notna().mean() * 100:5.1f}% of rows")
    rep("")
    rep("So these must be filled with 0/1, NOT dropped and NOT imputed with a")
    rep("mean. Dropping rows where 'vip' is missing would delete 91% of the data")
    rep("for no reason.")

    top = missing[missing > 0].head(18).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    flaglike = set(PRESENCE_FLAGS) | {"İpoteka"}
    colors = [ORANGE if c in flaglike else BLUE for c in top.index]
    ax.barh(top.index, top.to_numpy(), color=colors, height=0.72)
    for name, value in top.items():
        ax.text(value + 0.8, name, f"{value:.0f}%", va="center", color=INK_SOFT, fontsize=8)
    ax.set_xlim(0, 108)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style(ax, "Missing values per column", "% of rows missing")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="flag column — missing means 'no'"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="genuinely unknown"),
    ], frameon=False, loc="lower right", fontsize=8)
    save(fig, out_dir, "fig01_missing_values.png", rep)


# ==========================================================================
# 3. Duplicates
# ==========================================================================
def section_duplicates(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("3. DUPLICATES — is every row a different flat?")
    rep(f"Identical rows (all 51 columns equal): {df.duplicated().sum():,}")
    rep("")
    rep("But each advert has a stable page address in `estate_rel_url_x`")
    rep("(e.g. /items/4521724). If the same address appears twice, the same")
    rep("flat was scraped twice.")
    listings = df["item_id"].dropna().value_counts()
    rep(f"  rows in file:            {len(df):,}")
    rep(f"  distinct adverts:        {listings.size:,}")
    rep(f"  extra (repeat) rows:     {len(df) - listings.size:,}"
        f"  = {(len(df) - listings.size) / len(df) * 100:.1f}% of the file")
    rep("")
    rep("How many times each advert appears:")
    rep.table(listings.value_counts().sort_index().rename("adverts").to_frame())
    rep("")
    rep("Does the price change between the copies of one advert?")
    spread = df.dropna(subset=["item_id"]).groupby("item_id")[TARGET].nunique()
    rep(f"  adverts with a single price across their copies: {(spread == 1).sum():,}")
    rep(f"  adverts whose price changed:                     {(spread > 1).sum():,}")
    rep("")
    rep("WHY THIS MATTERS. If we split the data randomly without removing these,")
    rep("the same flat can land in both train and test. The model then recalls a")
    rep("price it has already seen and the test score is a lie.")

    counts = listings.value_counts().sort_index()
    counts = counts[counts.index <= 6]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=BLUE, width=0.6)
    for x, y in zip(counts.index.astype(str), counts.to_numpy()):
        ax.text(x, y, f"{y:,}", ha="center", va="bottom", color=INK_SOFT, fontsize=8)
    ax.xaxis.grid(False)
    style(ax, "Most adverts were scraped once, many twice",
          "times the same advert appears in the file", "number of adverts")
    save(fig, out_dir, "fig02_duplicate_adverts.png", rep)


# ==========================================================================
# 4. Leakage
# ==========================================================================
def section_leakage(df: pd.DataFrame, rep: Report) -> None:
    rep.header("4. LEAKAGE — which columns already contain the answer?")
    rep("A feature leaks if it could only be known once the price is known.")
    rep("Training on one makes the score perfect and the model worthless.")
    rep("")

    same = (df["total_price"] == df[TARGET]).mean() * 100
    rep(f"`total_price` equals `price` in {same:.2f}% of rows.")
    rep("  -> it is the target under a second name. DROP.")
    rep("")

    both = df[["unit_price_azn_m2", "area_m2"]].notna().all(axis=1)
    reconstructed = df.loc[both, "unit_price_azn_m2"] * df.loc[both, "area_m2"]
    error = (reconstructed - df.loc[both, TARGET]).abs() / df.loc[both, TARGET]
    rep(f"`unit_price` is price per m². On the {both.sum():,} rows where both it")
    rep("and the area are present, unit_price x area reconstructs the price:")
    rep(f"  within  1% of the true price: {(error < 0.01).mean() * 100:.1f}% of rows")
    rep(f"  within  5% of the true price: {(error < 0.05).mean() * 100:.1f}% of rows")
    rep("  -> it is the target divided by a feature we keep. DROP.")
    rep("")
    rep("`description` is free text written by the seller and very often states")
    rep("the price in words. Treat as leakage unless the price is stripped out:")
    sample = df["description"].dropna().head(4000).astype(str)
    mentions = sample.str.contains(r"\b(?:qiymət|azn|man\.?)\b", case=False, regex=True).mean()
    rep(f"  {mentions * 100:.0f}% of a 4,000-row sample mention a price word.")
    rep("")
    rep("VERDICT — drop before training: total_price, unit_price, description.")


# ==========================================================================
# 5. The target
# ==========================================================================
def section_target(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("5. THE TARGET `price` — what are we predicting?")
    price = df[TARGET]
    rep("Summary in AZN:")
    quantiles = price.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    rep.table(quantiles.apply(money))
    rep("")
    rep(f"The mean ({money(price.mean())}) is far above the median "
        f"({money(price.median())}).")
    rep("That gap is the signature of a long right tail: a few enormous values")
    rep("drag the average up. Look at both ends:")
    rep("")
    rep(f"  cheapest 10 prices: {sorted(price.nsmallest(10).tolist())}")
    rep(f"  priciest 5 prices:  {[money(v) for v in price.nlargest(5)]}")
    rep("")
    rep(f"  rows priced under 1,000 AZN:      {(price < 1_000).sum():,}")
    rep(f"  rows priced over 10,000,000 AZN:  {(price > 10_000_000).sum():,}")
    rep("  These are data-entry errors or whole buildings, not flats. A flat")
    rep("  does not cost 11 AZN. data_prep should cut an explicit price window")
    rep("  and the report must name the cut.")
    rep("")
    rep("WHY log(price). Prices span six orders of magnitude. In AZN, one")
    rep("mistake on a 10,000,000 flat outweighs thousands of ordinary flats, so")
    rep("squared error chases the outliers. log(price) turns 'twice as")
    rep("expensive' into a constant step, which is how people compare houses,")
    rep("and makes the distribution close to symmetric — see the figure.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    window = price[(price > 0) & (price < 1_000_000)]
    axes[0].hist(window, bins=80, color=BLUE)
    axes[0].axvline(price.median(), color=ORANGE, lw=2)
    axes[0].text(price.median() * 1.08, axes[0].get_ylim()[1] * 0.85,
                 f"median\n{money(price.median())} AZN", color=ORANGE, fontsize=8)
    axes[0].xaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f}k")
    style(axes[0], "Raw price — squeezed against the left wall",
          "price, AZN (shown up to 1,000,000)", "number of adverts")

    logged = np.log10(price[price > 0])
    axes[1].hist(logged, bins=80, color=AQUA)
    axes[1].axvline(np.log10(price.median()), color=ORANGE, lw=2)
    # One tick per power of ten, labelled compactly, so nothing overlaps.
    decades = range(int(np.floor(logged.min())), int(np.ceil(logged.max())) + 1)
    axes[1].set_xticks(list(decades), [compact(10 ** d) for d in decades])
    style(axes[1], "log10(price) — roughly symmetric, a usable target",
          "price, AZN on a log scale", "number of adverts")
    save(fig, out_dir, "fig03_price_distribution.png", rep)


# ==========================================================================
# 6. Numeric features
# ==========================================================================
def section_numeric(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("6. NUMERIC FEATURES — ranges, nonsense values, and the link to price")
    rep("Price and floor-area comparisons use residential listings only. Bare-land")
    rep("`Sahə` is measured in sot, so it is not comparable to residential m².")
    df = df[df["Kateqoriya"].isin(RESIDENTIAL)].copy()
    cols = ["area_m2", "rooms", "floor", "building_floors", "land_sot", "views", "lat", "lng"]
    rep("The raw file hides numbers inside text ('145 m²', '7 / 9'). After")
    rep("unpacking them:")
    rep("")
    rep.table(df[cols].describe(percentiles=[0.01, 0.5, 0.99]).round(2))
    rep("")
    rep("Impossible or implausible values that data_prep must handle:")
    rep(f"  area <= 5 m²:                   {(df['area_m2'] <= 5).sum():,}")
    rep(f"  area > 1,000 m²:                {(df['area_m2'] > 1_000).sum():,}")
    rep(f"  floor above building height:    {(df['floor'] > df['building_floors']).sum():,}")
    rep(f"  rooms > 10:                     {(df['rooms'] > 10).sum():,}")
    outside = (
        ~df["lat"].between(*AZ_LAT) | ~df["lng"].between(*AZ_LNG)
    ) & df["lat"].notna()
    rep(f"  coordinates outside Azerbaijan: {outside.sum():,}")
    rep("")
    rep("`views` is how many people opened the advert. It is recorded AFTER the")
    rep("advert goes live, so it is not something you know about a flat you are")
    rep("about to price. Treat it as unusable for an honest model.")

    # 6a. area vs price
    both = df[(df["area_m2"].between(10, 1000)) & (df[TARGET].between(5_000, 5_000_000))]
    sample = both.sample(min(12_000, len(both)), random_state=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(sample["area_m2"], sample[TARGET], s=5, alpha=0.18,
               color=BLUE, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    r = np.corrcoef(np.log(sample["area_m2"]), np.log(sample[TARGET]))[0, 1]
    style(ax, f"Bigger flats cost more (correlation of the logs r = {r:.2f})",
          "area, m² (log scale)", "price, AZN (log scale)")
    save(fig, out_dir, "fig04_area_vs_price.png", rep)
    rep("")
    rep(f"Correlation between log(area) and log(price) = {r:.2f}. Strong and")
    rep("close to a straight line on log axes, which is the clearest single")
    rep("argument for modelling log(price) rather than price.")

    # 6b. rooms
    subset = df[(df["rooms"].between(1, 7)) & (df[TARGET].between(5_000, 3_000_000))]
    groups = [subset.loc[subset["rooms"] == k, TARGET].to_numpy() for k in range(1, 8)]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bp = ax.boxplot(groups, tick_labels=[str(k) for k in range(1, 8)],
                    patch_artist=True, showfliers=False, widths=0.6)
    for patch in bp["boxes"]:
        patch.set(facecolor=SEQ_BLUE[2], edgecolor=SEQ_BLUE[5], linewidth=1.2)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set_color(SEQ_BLUE[5])
    for line in bp["medians"]:
        line.set(color=ORANGE, linewidth=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.grid(False)
    style(ax, "Price rises steadily with room count (orange line = median)",
          "number of rooms", "price, AZN (log scale)")
    save(fig, out_dir, "fig05_rooms_vs_price.png", rep)

    # 6c. floor
    sub = df[(df["floor"].between(1, 20)) & (df[TARGET].between(5_000, 3_000_000))]
    by_floor = sub.groupby("floor")[TARGET].median()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(by_floor.index, by_floor.to_numpy(), color=BLUE, lw=2, marker="o", ms=5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f}k")
    style(ax, "Which floor a flat is on barely moves the median price",
          "floor the flat is on", "median price, AZN")
    save(fig, out_dir, "fig06_floor_vs_price.png", rep)
    rep("")
    rep("Median price by floor is nearly flat — floor is a weak feature on its")
    rep("own. Ground floor and top floor are the only ones that stand out.")


# ==========================================================================
# 7. Categorical features
# ==========================================================================
def section_categorical(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("7. CATEGORICAL FEATURES — how many groups, and do they separate price?")
    rep("For a feature to be useful, its groups must have different prices.")
    rep("We judge that with the median price per group (the median ignores the")
    rep("few huge outliers that would wreck a mean).")
    for col in ["Kateqoriya", "Təmir", "Çıxarış", "İpoteka", "owner_title", "products_label", "city"]:
        rep("")
        rep(f"--- {col} ---")
        table = pd.DataFrame({
            "rows": df[col].value_counts(dropna=False),
            "median_price": df.groupby(col, dropna=False)[TARGET].median().round(0),
        })
        rep.table(table)

    rep("")
    rep(f"`location` holds {df['location'].nunique()} different district/metro")
    rep("names. One-hot encoding all of them would add that many columns, most")
    rep("of them almost always zero. data_prep should keep the frequent ones and")
    rep("fold the rest into a single 'other' bucket.")
    rep("")
    rep("REDUNDANCY CHECK — some pairs of columns say the same thing twice:")
    for flag, cat, yes in [("repair", "Təmir", "var"),
                           ("bill_of_sale", "Çıxarış", "var"),
                           ("mortgage", "İpoteka", "var")]:
        agree = (df[flag].notna() == (df[cat] == yes)).mean() * 100
        rep(f"  {flag:<13} notna  vs  {cat} == '{yes}'   agree on {agree:.1f}% of rows")
    rep("  -> keep one of each pair, drop the other.")

    # 7a. category counts and prices, side by side
    cats = df["Kateqoriya"].value_counts()
    med = df.groupby("Kateqoriya")[TARGET].median().reindex(cats.index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].barh(cats.index[::-1], cats.to_numpy()[::-1], color=BLUE, height=0.7)
    for name, value in zip(cats.index[::-1], cats.to_numpy()[::-1]):
        axes[0].text(value * 1.02, name, f"{value:,}", va="center", color=INK_SOFT, fontsize=8)
    axes[0].set_xlim(0, cats.max() * 1.18)
    axes[0].yaxis.grid(False)
    style(axes[0], "How many adverts per property type", "number of adverts")

    axes[1].barh(med.index[::-1], med.to_numpy()[::-1], color=ORANGE, height=0.7)
    for name, value in zip(med.index[::-1], med.to_numpy()[::-1]):
        axes[1].text(value * 1.02, name, money(value), va="center", color=INK_SOFT, fontsize=8)
    axes[1].set_xlim(0, med.max() * 1.25)
    axes[1].yaxis.grid(False)
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f}k")
    style(axes[1], "Median price per property type", "median price, AZN")
    save(fig, out_dir, "fig07_property_type.png", rep)

    # 7b. top locations
    frequent = df["location"].value_counts().head(20).index
    loc = (df[df["location"].isin(frequent)]
           .groupby("location")[TARGET].median().sort_values())
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.barh(loc.index, loc.to_numpy(), color=BLUE, height=0.72)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f}k")
    ax.yaxis.grid(False)
    style(ax, "Median price across the 20 most common locations",
          "median price, AZN")
    save(fig, out_dir, "fig08_location_vs_price.png", rep)
    rep("")
    rep(f"Across those 20 locations the median price runs from "
        f"{money(loc.min())} to {money(loc.max())} AZN — a "
        f"{loc.max() / loc.min():.1f}x spread. Location is a strong feature.")


# ==========================================================================
# 8. Geography
# ==========================================================================
def section_geography(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("8. GEOGRAPHY — do the coordinates make sense, and do they matter?")
    inside = df["lat"].between(*AZ_LAT) & df["lng"].between(*AZ_LNG)
    rep(f"Coordinates inside Azerbaijan: {inside.sum():,} ({inside.mean() * 100:.2f}%)")
    rep(f"Coordinates outside:           {(~inside).sum():,}")
    rep("  A listing in Baku cannot sit at longitude 12°. Those rows have a")
    rep("  broken coordinate; the rest of the row may still be fine, so")
    rep("  data_prep should blank the coordinate rather than delete the row.")

    sub = df[inside & df[TARGET].between(20_000, 2_000_000)]
    sub = sub[sub["lat"].between(40.3, 40.5) & sub["lng"].between(49.7, 50.0)]  # Baku
    # Plotting 20,000 overlapping dots hides the pattern: later dots cover
    # earlier ones. Instead we chop the city into small hexagons and colour
    # each one by the MEDIAN price of the adverts inside it. Hexagons holding
    # fewer than 10 adverts are left blank, since their median is unreliable.
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    hexes = ax.hexbin(
        sub["lng"], sub["lat"], C=sub[TARGET],
        reduce_C_function=np.median, mincnt=10, gridsize=55,
        cmap=ramp, norm=matplotlib.colors.LogNorm(vmin=60_000, vmax=900_000),
        linewidths=0.2, edgecolors="white",
    )
    bar = fig.colorbar(hexes, ax=ax, shrink=0.85)
    bar.set_label("median price in the hexagon, AZN", color=INK_SOFT)
    bar.set_ticks([60_000, 100_000, 200_000, 400_000, 900_000])
    bar.ax.yaxis.set_major_formatter(lambda v, _: compact(v))
    bar.ax.minorticks_off()
    bar.outline.set_visible(False)
    ax.grid(False)
    style(ax, "Median advert price across Baku", "longitude", "latitude")
    save(fig, out_dir, "fig09_map_of_baku.png", rep)
    rep("")
    rep("The map shows a clear, smooth price gradient: dark hexagons (expensive)")
    rep("sit in the city centre by the bay and fade outwards in every direction.")
    rep("Latitude and longitude therefore carry real signal — but")
    rep("as a pair, not individually, which a decision tree handles well")
    rep("(it can cut on both) and a linear SVM handles poorly.")


# ==========================================================================
# 9. Correlation
# ==========================================================================
def section_correlation(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("9. CORRELATION — which numeric columns move together?")
    rep("Pearson correlation measures straight-line agreement between two")
    rep("columns: +1 move together, 0 unrelated, -1 move oppositely. We use")
    rep("log(price) and log(area) because both are heavily skewed.")
    work = df[df["Kateqoriya"].isin(RESIDENTIAL) & (df[TARGET] > 0)].copy()
    work["log_price"] = np.log(work[TARGET])
    work["log_area"] = np.log(work["area_m2"].where(work["area_m2"] > 0))
    cols = ["log_price", "log_area", "rooms", "floor", "building_floors", "lat", "lng", "views"]
    corr = work[cols].corr(numeric_only=True)
    rep("")
    rep.table(corr.round(2))
    rep("")
    rep("Reading it: area and rooms both track price strongly, and they track")
    rep("each other strongly too — they partly repeat the same information.")
    rep("`views` is near zero, another reason to leave it out.")

    fig, ax = plt.subplots(figsize=(6.2, 5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", DIVERGING[::-1])
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.to_numpy()[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else INK)
    bar = fig.colorbar(im, ax=ax, shrink=0.8)
    bar.outline.set_visible(False)
    ax.grid(False)
    style(ax, "Pearson correlation between numeric columns")
    save(fig, out_dir, "fig10_correlation.png", rep)


# ==========================================================================
# 10. Task B preview
# ==========================================================================
def section_tiers(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("10. TASK B PREVIEW — the premium / standard split")
    threshold = df[TARGET].median()
    premium = df[TARGET] > threshold
    rep("This is a whole-source descriptive illustration only. The model uses")
    rep("a separately computed training-set median and >= for the premium label.")
    rep(f"For this chart only, the whole-file median is {money(threshold)} AZN:")
    rep(f"  standard (<= median): {(~premium).sum():,}  ({(~premium).mean() * 100:.1f}%)")
    rep(f"  premium  (>  median): {premium.sum():,}  ({premium.mean() * 100:.1f}%)")
    rep("")
    rep("These descriptive labels do not enter the model or held-out evaluation.")
    rep("")
    rep("Share of premium adverts inside each property type:")
    share = (df.assign(premium=premium)
             .groupby("Kateqoriya")["premium"].agg(["mean", "size"])
             .sort_values("mean", ascending=False))
    share.columns = ["premium_share", "rows"]
    rep.table(share.assign(premium_share=lambda t: (t["premium_share"] * 100).round(1)))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    order = share.index
    vals = share["premium_share"].to_numpy() * 100
    ax.barh(order[::-1], vals[::-1], color=BLUE, height=0.7)
    ax.axvline(50, color=ORANGE, lw=1.5, ls="--")
    ax.text(51, -0.45, "50% = balanced", color=ORANGE, fontsize=8)
    for name, value in zip(order[::-1], vals[::-1]):
        ax.text(value + 1.5, name, f"{value:.0f}%", va="center", color=INK_SOFT, fontsize=8)
    ax.set_xlim(0, 108)
    ax.yaxis.grid(False)
    style(ax, "Share of 'premium' adverts within each property type",
          "% of the type priced above the overall median")
    save(fig, out_dir, "fig11_price_tiers.png", rep)


# ==========================================================================
# 11. SCOPE DECISION
# ==========================================================================
def section_scope(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("11. SCOPE DECISION — which rows belong in the task at all?")
    rep("`Kateqoriya` mixes seven kinds of property. The question is whether")
    rep("they are one prediction problem or several. Three measurements.")

    # ---- 1a. unit inconsistency -------------------------------------------
    rep("")
    rep("1a. THE `Sahə` COLUMN DOES NOT USE ONE UNIT.")
    rep("Reading the text, not just the number:")
    rep("")
    units = (df.assign(unit=df["Sahə"].astype("string")
                       .str.extract(r"([^0-9.,\s]+)", expand=False))
             .groupby("Kateqoriya")["unit"].agg(lambda s: s.value_counts().index[0]))
    examples = df.groupby("Kateqoriya")["Sahə"].first()
    rep.table(pd.DataFrame({"unit": units, "example": examples}))
    rep("")
    rep("`Torpaq` (bare land) measures area in SOT, everybody else in m².")
    rep("1 sot = 100 m². A naive numeric parse puts '12 sot' (1,200 m²) and")
    rep("'65 m²' in the same column as 12 and 65, so the plot of land looks")
    rep("five times SMALLER than the flat. This is silent: nothing crashes,")
    rep("the model simply learns nonsense.")

    # ---- 1b. structurally missing features ---------------------------------
    rep("")
    rep("1b. THE CATEGORIES DO NOT SHARE A FEATURE SET.")
    rep("Percentage of rows where each feature is present:")
    rep("")
    presence = pd.DataFrame({
        "rows": df.groupby("Kateqoriya").size(),
        "area": df.groupby("Kateqoriya")["Sahə"].apply(lambda s: s.notna().mean() * 100),
        "rooms": df.groupby("Kateqoriya")["Otaq sayı"].apply(lambda s: s.notna().mean() * 100),
        "floor": df.groupby("Kateqoriya")["Mərtəbə"].apply(lambda s: s.notna().mean() * 100),
        "land size": df.groupby("Kateqoriya")["Torpaq sahəsi"].apply(lambda s: s.notna().mean() * 100),
    }).sort_values("rows", ascending=False)
    rep.table(presence.round(1))
    rep("")
    rep("These are not gaps to impute. A plot of land HAS no floor and HAS no")
    rep("rooms; the question does not apply. Training one model across all")
    rep("seven means most of the feature matrix is structurally empty.")

    order = presence.index
    grid = presence[["area", "rooms", "floor", "land size"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    im = ax.imshow(grid.to_numpy(), cmap=ramp, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(len(order)), [f"{c}  (n={presence.loc[c, 'rows']:,})" for c in order])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.to_numpy()[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 55 else INK)
    ax.grid(False)
    style(ax, "Which features exist at all, by property type (% of rows present)")
    save(fig, out_dir, "fig12_feature_availability.png", rep)

    # ---- 1c. is price per m2 even comparable? ------------------------------
    rep("")
    rep("1c. PRICE PER m² IS CONSISTENT WITHIN FLATS, WILD ACROSS THE REST.")
    rep("If a category is one coherent market, its price per m² should sit in")
    rep("a narrow band. We measure the band as the 90th percentile divided by")
    rep("the 10th: '1.9' means the dear end is 1.9x the cheap end.")
    rep("")
    flats_only = df[df["Kateqoriya"] != "Torpaq"]  # excluded: different unit
    ppm2 = flats_only[TARGET] / flats_only["area_m2"]
    band = (ppm2.groupby(flats_only["Kateqoriya"])
            .quantile([0.1, 0.5, 0.9]).unstack())
    band["p90 / p10"] = (band[0.9] / band[0.1]).round(1)
    band.columns = ["p10", "median", "p90", "p90 / p10"]
    rep.table(band.round(0).sort_values("p90 / p10"))
    rep("")
    rep("Flats are the tightest (1.9-2.0x). Houses are looser at 3.8x, because")
    rep("a house's worth is partly its plot, and plot size varies hugely — but")
    rep("houses DO record their land size, so that spread is explainable with")
    rep("the columns we have. `Obyekt` at 8.7x is not: what a commercial unit")
    rep("earns is nowhere in this file.")
    rep("")
    rep("`Torpaq` is excluded from the table because its area is in sot, so a")
    rep("price per m² would be a hundredfold wrong. Measured in its own unit")
    land = df[df["Kateqoriya"] == "Torpaq"]
    per_sot = (land[TARGET] / land["area_m2"]).quantile([0.1, 0.9])
    rep(f"(AZN per sot) land still spreads {per_sot.iloc[1] / per_sot.iloc[0]:.0f}x "
        "from its 10th to its 90th percentile — because what a plot is worth")
    rep("depends on permission to build on it, which is not in this dataset.")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    srt = band.sort_values("p90 / p10")
    colors = [AQUA if name in RESIDENTIAL else ORANGE for name in srt.index]
    ax.barh(srt.index, srt["p90 / p10"], color=colors, height=0.7)
    for name, value in srt["p90 / p10"].items():
        ax.text(value + 0.1, name, f"{value:.1f}x", va="center", color=INK_SOFT, fontsize=8)
    ax.set_xlim(0, srt["p90 / p10"].max() * 1.2)
    ax.yaxis.grid(False)
    style(ax, "How tightly priced each property type is",
          "90th percentile price per m²  ÷  10th percentile")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=AQUA, label="kept — residential"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="dropped — see 1b"),
    ], frameon=False, loc="lower right", fontsize=8)
    save(fig, out_dir, "fig13_price_consistency.png", rep)

    # ---- 1d. what each scope costs us -------------------------------------
    rep("")
    rep("1d. WHAT EACH SCOPE COSTS IN ROWS (counted after deduplication).")
    one = latest_listing_per_id(df)
    rep("")
    rep.table(one["city"].value_counts().rename("adverts").to_frame())
    rep("")
    options = {
        "everything": pd.Series(True, index=one.index),
        "Baku only": one["city"].eq("bakı"),
        "Baku + residential (flats & houses)":
            one["city"].eq("bakı") & one["Kateqoriya"].isin(RESIDENTIAL),
        "Baku + apartments only": one["city"].eq("bakı") & one["Kateqoriya"].isin(APARTMENTS),
    }
    table = pd.DataFrame({
        "adverts kept": {k: int(v.sum()) for k, v in options.items()},
        "% of adverts": {k: round(v.mean() * 100, 1) for k, v in options.items()},
    })
    rep.table(table)
    rep("")
    rep("RECOMMENDATION. Keep Baku + the three residential types")
    kept = int(options['Baku + residential (flats & houses)'].sum())
    rep(f"({kept:,} of {len(one):,} normalized unique advert IDs, {kept / max(len(one), 1) * 100:.1f}%).")
    rep(f"  - {int((~one['city'].eq('bakı')).sum()):,} unique IDs are outside Baku; this project")
    rep("    makes a deliberately Baku-specific prediction task.")
    rep(f"  - {int((~one['Kateqoriya'].isin(RESIDENTIAL)).sum()):,} unique IDs are outside the three residential categories;")
    rep("    land uses sot while residential floor area uses m², and other categories")
    rep("    often lack rooms/floor. This is a scope choice, not a claim that those")
    rep("    listings are invalid.")
    rep("  - the three residential types share one feature set and price per m²")
    rep("    bands of 1.9-3.8x, so they are one problem.")
    rep("  - `Kateqoriya` stays as a feature, so the model can still tell a")
    rep("    house from a new-build flat.")


# ==========================================================================
# 12. DEDUPLICATION DECISION
# ==========================================================================
def section_dedup(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("12. DEDUPLICATION DECISION — one row per advert, but which copy?")
    span = df["scraped"].agg(["min", "max"])
    rep(f"The file was scraped repeatedly between {span['min']:%d %b %Y} and "
        f"{span['max']:%d %b %Y}")
    rep(f"({(span['max'] - span['min']).days} days). A normalized item ID may appear")
    rep("on multiple scrape days. The repeated-row share is")
    rep(f"{(len(df) - df['item_id'].nunique()) / max(len(df), 1) * 100:.1f}%; these are")
    rep("repeat observations of an advert, not necessarily distinct properties.")

    copies = df["item_id"].value_counts()
    multi = df[df["item_id"].isin(copies[copies > 1].index)]
    grouped = multi.groupby("item_id")

    # ---- 2a. what actually differs between the copies ---------------------
    rep("")
    rep("2a. WHAT CHANGES BETWEEN THE COPIES OF ONE ADVERT?")
    rep("For every column, the share of repeated adverts where that column")
    rep("takes more than one value:")
    rep("")
    watch = ["price", "views", "lat", "lng", "description", "Sahə", "Otaq sayı",
             "Mərtəbə", "Kateqoriya", "location", "Təmir", "vip"]
    varies = pd.Series(
        {c: (grouped[c].nunique(dropna=False) > 1).mean() * 100 for c in watch}
    ).sort_values(ascending=False)
    rep.table(varies.round(1).rename("% of repeated adverts that change"))
    rep("")
    rep("The table above shows which values change and how often. Repeated IDs")
    rep("are the same advert observed at different moments; retaining the last")
    rep("dated row defines the final observed asking price, not a completed sale.")

    # ---- 2b. how prices move ----------------------------------------------
    rep("")
    rep("2b. WHEN THE PRICE MOVES, WHICH WAY DOES IT GO?")
    ordered = df.sort_values("scraped", kind="mergesort")
    first = ordered.groupby("item_id")[TARGET].first()
    last = ordered.groupby("item_id")[TARGET].last()
    moved = first != last
    changed_n = int(moved.sum())
    down, up = (last[moved] < first[moved]).sum(), (last[moved] > first[moved]).sum()
    rep(f"  adverts observed more than once: {len(copies[copies > 1]):,}")
    rep(f"  of those, the price changed:     {moved.sum():,}")
    rep(f"     price went DOWN: {down:,} ({down / changed_n * 100:.0f}%)" if changed_n else "     price went DOWN: 0 (no price changes)")
    rep(f"     price went UP:   {up:,} ({up / changed_n * 100:.0f}%)" if changed_n else "     price went UP: 0 (no price changes)")
    pct = ((last[moved] - first[moved]) / first[moved] * 100) if changed_n else pd.Series(dtype=float)
    rep("")
    rep("Size of the change, in percent of the first price:")
    rep.table(pct.describe(percentiles=[0.1, 0.5, 0.9]).round(1))
    rep("")
    rep("Sellers cut their asking price far more often than they raise it —")
    rep("the normal behaviour of a listing that is not selling. The LAST")
    rep("observation is therefore the price the market actually saw last.")

    clipped = pct.clip(-40, 40)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(clipped[clipped < 0], bins=40, color=ORANGE, label=f"cut the price ({down:,})")
    ax.hist(clipped[clipped > 0], bins=40, color=BLUE, label=f"raised the price ({up:,})")
    ax.axvline(0, color=INK_SOFT, lw=1)
    ax.legend(frameon=False, fontsize=8)
    style(ax, "Price changes between the first and last sighting of an advert",
          "change, % of the first price (clipped at ±40%)", "number of adverts")
    save(fig, out_dir, "fig14_price_changes.png", rep)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    counts = copies.value_counts().sort_index()
    counts = counts[counts.index <= 8]
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=BLUE, width=0.6)
    for x, y in zip(counts.index.astype(str), counts.to_numpy()):
        ax.text(x, y, f"{y:,}", ha="center", va="bottom", color=INK_SOFT, fontsize=8)
    ax.xaxis.grid(False)
    style(ax, "How often the same advert was scraped",
          "number of sightings", "number of adverts")
    save(fig, out_dir, "fig15_sightings.png", rep)

    # ---- 2c. the second, hidden kind of duplicate --------------------------
    rep("")
    rep("2c. A SECOND KIND OF DUPLICATE, EASY TO MISS.")
    rep("Deduplicating on the advert address still leaves the same flat in the")
    rep("file twice if an agency posted it twice under two addresses. We look")
    rep("for adverts that agree on coordinates, area, rooms AND price:")
    one = latest_listing_per_id(df)
    key = ["lat", "lng", "area_m2", "rooms", TARGET]
    complete = one[key].notna().all(axis=1)
    twins = one[complete].duplicated(subset=key, keep=False)
    rep("")
    rep(f"  adverts after address-deduplication: {len(one):,}")
    rep(f"  of those, sharing all five values with another advert: {twins.sum():,}"
        f" ({twins.mean() * 100:.1f}%)")
    rep("")
    rep("These are candidate matches, not verified duplicates. They may include")
    rep("distinct units with coincidentally equal measurements, so the pipeline")
    rep("does not remove them based on these five fields alone.")

    rep("")
    rep("RECOMMENDATION. Before splitting, normalize the advert URL to its numeric")
    rep("ID, sort by scrape date, and retain its last sighting. Matching on")
    rep("coordinates, area, rooms and price only identifies candidates: separate")
    rep("homes in one building can share these values. Report candidate matches;")
    rep("do not delete them without confirming address/property identity.")


# ==========================================================================
# 13. PRICE WINDOW DECISION
# ==========================================================================
def section_window(df: pd.DataFrame, out_dir: Path, rep: Report) -> None:
    rep.header("13. PRICE WINDOW DECISION — where do we cut impossible prices?")
    one = latest_listing_per_id(df)
    scoped = one[one["city"].eq("bakı") & one["Kateqoriya"].isin(RESIDENTIAL)].copy()
    scoped["ppm2"] = scoped[TARGET] / scoped["area_m2"]
    rep(f"Measured on the recommended scope: {len(scoped):,} Baku residential adverts.")

    # ---- 3a. the tails -----------------------------------------------------
    rep("")
    rep("3a. HOW FAR OUT THE TAILS GO.")
    tails = scoped[TARGET].quantile([0.0001, 0.001, 0.005, 0.01, 0.5, 0.99, 0.995, 0.999, 0.9999])
    rep.table(tails.apply(money).rename("price, AZN"))

    rep("")
    rep("3b. ARE THE EXTREMES REAL LISTINGS OR TYPING MISTAKES?")
    cols = [TARGET, "area_m2", "rooms", "Kateqoriya", "location"]
    cheap = scoped.nsmallest(8, TARGET)[cols].copy()
    cheap["AZN per m²"] = (cheap[TARGET] / cheap["area_m2"]).round(0)
    rep("")
    rep("The eight cheapest:")
    rep.table(cheap.to_string(index=False))
    dear = scoped.nlargest(6, TARGET)[cols].copy()
    dear["AZN per m²"] = (dear[TARGET] / dear["area_m2"]).round(0)
    rep("")
    rep("The six dearest:")
    rep.table(dear.to_string(index=False))
    rep("")
    rep("A 3-room flat does not sell for 73 AZN, and 600,000,000 AZN for 200 m²")
    rep("is 3,000,000 AZN per m² — about a thousand times the going rate, almost")
    rep("certainly a seller who typed too many zeros. These are not rare")
    rep("expensive flats; they are broken records.")

    # ---- 3c. price per m2 is the sharper detector --------------------------
    rep("")
    rep("3c. PRICE PER m² SPOTS THE ERRORS BETTER THAN PRICE ALONE.")
    rep("A cheap price is not automatically wrong — a 30 m² studio really is")
    rep("cheap. What is wrong is a price that cannot be squared with the size.")
    rep("")
    rep.table(scoped["ppm2"].quantile([0.001, 0.01, 0.5, 0.99, 0.999])
              .round(0).rename("AZN per m²"))
    rep("")
    rep(f"  adverts under    200 AZN/m²: {(scoped['ppm2'] < 200).sum():,}")
    rep(f"  adverts over  20,000 AZN/m²: {(scoped['ppm2'] > 20_000).sum():,}")
    rep("The ordinary market sits between roughly 1,000 and 5,000 AZN/m². The")
    rep("handful outside 200-20,000 are the broken ones.")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    valid = scoped["ppm2"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(np.log10(valid[valid > 0]), bins=90, color=BLUE)
    for edge, label in ((200, "200"), (20_000, "20,000")):
        ax.axvline(np.log10(edge), color=ORANGE, lw=2, ls="--")
        ax.text(np.log10(edge), ax.get_ylim()[1] * 0.92, f" {label}",
                color=ORANGE, fontsize=8)
    decades = range(1, 7)
    ax.set_xticks(list(decades), [compact(10 ** d) for d in decades])
    style(ax, "Price per m² — the bulk is tight, the errors sit far outside it",
          "AZN per m² (log scale); dashed lines mark the proposed cut",
          "number of adverts")
    save(fig, out_dir, "fig16_price_per_m2.png", rep)

    # ---- 3d. what each candidate window costs ------------------------------
    rep("")
    rep("3d. WHAT EACH CANDIDATE WINDOW COSTS.")
    rep("")
    candidates = [(5_000, 10_000_000), (10_000, 5_000_000),
                  (20_000, 3_000_000), (30_000, 2_000_000)]
    rows = {}
    for lo, hi in candidates:
        keep = scoped[TARGET].between(lo, hi)
        rows[f"{compact(lo)} - {compact(hi)} AZN"] = {
            "adverts kept": int(keep.sum()),
            "adverts lost": int((~keep).sum()),
            "% lost": round((~keep).mean() * 100, 2),
        }
    rep.table(pd.DataFrame(rows).T)

    lo, hi = 10_000, 5_000_000
    price_ok = scoped[TARGET].between(lo, hi)
    ppm2_ok = scoped["ppm2"].between(200, 20_000)
    rep("")
    rep(f"Combining the price window {compact(lo)}-{compact(hi)} AZN with the")
    rep("price-per-m² window 200-20,000 AZN/m²:")
    rep(f"  removed by the price window only:      {(~price_ok).sum():,}")
    rep(f"  removed by the per-m² window only:     {(~ppm2_ok).sum():,}")
    rep(f"  removed by either:                     {(~(price_ok & ppm2_ok)).sum():,}"
        f" ({(~(price_ok & ppm2_ok)).mean() * 100:.2f}% of the scope)")
    rep("")
    rep("RECOMMENDATION. Keep 10,000 <= price <= 5,000,000 AZN and")
    rep("200 <= price/m² <= 20,000, plus the 10-2,000 m² area range.")
    rep("These rules remove rows outside the declared bounds; they do not verify")
    rep("that every retained record is accurate.")
    rep("")
    rep("HONESTY NOTE. These are fixed domain bounds, not quantiles recomputed")
    rep("inside each split, and they are applied before the split. However, the")
    rep("bounds were selected after inspecting this dataset's full price/area")
    rep("distribution. That is a source-selection decision and should be disclosed;")
    rep("it is not evidence that the held-out test distribution was unknown.")
    rep("Report the exact bounds and count removed, and do not tune them against")
    rep("model test scores.")


# ==========================================================================
# 14. The shortlist that data_prep will act on
# ==========================================================================
def section_conclusions(df: pd.DataFrame, rep: Report) -> None:
    rep.header("14. CONCLUSIONS — what data_prep.py has to do")
    rep("DROP — no information:")
    rep(f"  identifiers and scrape bookkeeping ({len(ID_AND_BOOKKEEPING)} columns): "
        f"{', '.join(ID_AND_BOOKKEEPING)}")
    rep("  constant columns: currency_x, currency_y, shop_title, (Binanın növü: 99.8% empty)")
    rep("")
    rep("DROP — leakage:")
    rep("  total_price (identical to price), unit_price (price per m²), description")
    rep("")
    rep("DROP — not knowable before the sale:")
    rep("  views")
    rep("")
    rep("DEDUPLICATE:")
    rep("  last sighting per numeric ID extracted from estate_rel_url_x, before splitting")
    rep("")
    rep("DECIDE A SCOPE:")
    rep("  use the documented project scope: Baku + the three residential types;")
    rep("  other categories remain in the untouched raw source.")
    rep("")
    rep("PARSE OUT OF TEXT:")
    rep("  Sahə -> area_m2,  Mərtəbə -> floor + building_floors,")
    rep("  Torpaq sahəsi -> land_sot")
    rep("")
    rep("FILL FLAGS WITH 0/1, NOT MEANS:")
    rep(f"  {', '.join(PRESENCE_FLAGS)}")
    rep("")
    rep("CUT IMPOSSIBLE VALUES (and count every row removed):")
    rep("  price 10,000–5,000,000 AZN; area 10–2,000 m²; price/m² 200–20,000;")
    rep("  non-missing rooms 1–30 and positive/plausible floor values")
    rep("")
    rep("ENCODE:")
    rep("  Kateqoriya and the frequent `location` values as one-hot, with a")
    rep("  shared 'other' bucket; yes/no columns as 0/1")
    rep("")
    rep("TARGET:")
    rep("  Task A on log(price);  Task B on a median split of the TRAINING set")
    rep("")
    rep("SCALING:")
    rep("  continuous features are standardized using training-only means/stds;")
    rep("  binary indicators remain 0/1. Tree split ordering is scale-invariant;")
    rep("  Pegasos is additionally scaled within training folds for its objective.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only exploration of the bina.az dataset.")
    p.add_argument("--data", default="data/bina_az_sale.csv", help="path to the raw CSV")
    p.add_argument("--out", default="reports/eda", help="where the report and figures go")
    return p.parse_args()


def explore_raw(data_path: str | Path = "data/bina_az_sale.csv",
                output_dir: str | Path = "reports/eda") -> dict:
    """Run the read-only full EDA and return its basic source summary."""
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()
    rep("bina.az sale — exploratory data analysis")
    rep(f"source file: {data_path}")
    rep("read-only analysis; source CSV is not modified")
    raw = pd.read_csv(data_path)
    df = build_exploration_frame(raw)
    section_overview(raw, rep)
    section_missing(raw, fig_dir, rep)
    section_duplicates(df, fig_dir, rep)
    section_leakage(df, rep)
    section_target(df, fig_dir, rep)
    section_numeric(df, fig_dir, rep)
    section_categorical(df, fig_dir, rep)
    section_geography(df, fig_dir, rep)
    section_correlation(df, fig_dir, rep)
    section_tiers(df, fig_dir, rep)
    section_scope(df, fig_dir, rep)
    section_dedup(df, fig_dir, rep)
    section_window(df, fig_dir, rep)
    section_conclusions(df, rep)
    rep.save(out_dir / "eda_report.txt")
    return {"rows": len(raw), "columns": raw.shape[1], "figures": len(list(fig_dir.glob("*.png")))}


def main() -> int:
    args = parse_args()
    summary = explore_raw(args.data, args.out)
    print(f"\nReport written to {Path(args.out) / 'eda_report.txt'}")
    print(f"Explored {summary['rows']:,} rows and wrote {summary['figures']} figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
