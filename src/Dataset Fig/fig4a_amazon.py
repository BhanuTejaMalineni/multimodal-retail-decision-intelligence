"""
Figure 4a — Amazon Reviews 2023 Dataset Characteristics
========================================================

Generates a publication-quality figure with 3 subplots showing:
  (a) Histogram of review-length distribution
  (b) Top 10 product categories by review count
  (c) Rating distribution pie chart

Usage:
    cd /Users/bhanutejamalineni/Thesis
    python figures/fig4a_amazon.py

Output:
    figures/dataset_amazon.pdf  (publication-ready PDF)
    Console: summary statistics for verification against Table 2

Prerequisites:
    pip install pandas matplotlib seaborn pyarrow
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path("/Users/bhanutejamalineni/Thesis/outputs/prepared")
OUTPUT_DIR = Path("/Users/bhanutejamalineni/Thesis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Try the most common filename patterns - script auto-detects which exists
AMAZON_CANDIDATES = [
    "amazon_merged.parquet",        # actual file name on your machine
    "amazon_filtered.parquet",
    "amazon_prepared.parquet",
    "amazon.parquet",
    "amazon_reviews.parquet",
    "amazon_clean.parquet",
]

# Color palette - matches your other figures (blue family for Amazon)
COLOR_PRIMARY = "#1976D2"
COLOR_SECONDARY = "#42A5F5"
COLOR_ACCENT = "#0D47A1"
PIE_COLORS = ["#D32F2F", "#F57C00", "#FBC02D", "#388E3C", "#1976D2"]

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ============================================================
# DATA LOADING
# ============================================================
def find_amazon_file():
    """Find the Amazon parquet file from common naming patterns."""
    for candidate in AMAZON_CANDIDATES:
        path = DATA_DIR / candidate
        if path.exists():
            print(f"[OK] Found Amazon data: {path}")
            return path

    # Fallback: list everything in the directory
    print(f"[ERROR] No Amazon parquet found in {DATA_DIR}")
    print("Files available:")
    for f in sorted(DATA_DIR.glob("*.parquet")):
        print(f"  - {f.name}")
    raise FileNotFoundError(
        "Could not find Amazon parquet. Update AMAZON_CANDIDATES with the right name."
    )


def load_amazon(path):
    """Load Amazon dataset and detect column names."""
    df = pd.read_parquet(path)
    print(f"[OK] Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"[OK] Columns: {list(df.columns)}")
    return df


def detect_columns(df):
    """Auto-detect the review-text, category, and rating columns."""
    cols = {col.lower(): col for col in df.columns}

    # Review text column
    text_col = None
    for key in ["review_text", "text", "review", "content", "body", "title"]:
        if key in cols:
            text_col = cols[key]
            break

    # Category column
    cat_col = None
    for key in ["category", "main_category", "categories", "product_category"]:
        if key in cols:
            cat_col = cols[key]
            break

    # Rating column
    rating_col = None
    for key in ["rating", "overall", "stars", "score", "review_rating"]:
        if key in cols:
            rating_col = cols[key]
            break

    print(f"[OK] Text column: {text_col}")
    print(f"[OK] Category column: {cat_col}")
    print(f"[OK] Rating column: {rating_col}")

    return text_col, cat_col, rating_col


# ============================================================
# PLOTTING
# ============================================================
def plot_review_length(ax, df, text_col):
    """Subplot (a): histogram of review-text length distribution."""
    if text_col is None:
        ax.text(0.5, 0.5, "Text column not found",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(a) Review-text length distribution")
        return

    # Compute character lengths
    lengths = df[text_col].astype(str).str.len()
    # Trim outliers for cleaner visualization (keep 99th percentile)
    upper = lengths.quantile(0.99)
    lengths_plot = lengths[lengths <= upper]

    ax.hist(lengths_plot, bins=50, color=COLOR_PRIMARY, edgecolor=COLOR_ACCENT, alpha=0.85)
    ax.axvline(lengths.median(), color=COLOR_ACCENT, linestyle="--", linewidth=1.5,
               label=f"Median = {int(lengths.median())} chars")
    ax.axvline(lengths.mean(), color="#D32F2F", linestyle=":", linewidth=1.5,
               label=f"Mean = {int(lengths.mean())} chars")
    ax.set_xlabel("Review length (characters)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("(a) Review-text length distribution")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    print(f"\n  Review length stats:")
    print(f"    Mean:   {lengths.mean():.1f} characters")
    print(f"    Median: {lengths.median():.1f} characters")
    print(f"    Min/Max: {lengths.min()} / {lengths.max()}")


def plot_top_categories(ax, df, cat_col):
    """Subplot (b): top 10 product categories by review count."""
    if cat_col is None:
        ax.text(0.5, 0.5, "Category column not found",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(b) Top 10 product categories")
        return

    top_cats = df[cat_col].value_counts().head(10)

    # Truncate long category names for display
    display_names = [
        (n[:25] + "...") if len(str(n)) > 25 else str(n)
        for n in top_cats.index
    ]

    bars = ax.barh(range(len(top_cats)), top_cats.values,
                   color=COLOR_PRIMARY, edgecolor=COLOR_ACCENT, alpha=0.85)
    ax.set_yticks(range(len(top_cats)))
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()
    ax.set_xlabel("Number of reviews")
    ax.set_title("(b) Top 10 product categories")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels on each bar
    for bar, val in zip(bars, top_cats.values):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f" {val:,}", va="center", fontsize=8, color=COLOR_ACCENT)

    print(f"\n  Top 5 categories:")
    for name, count in top_cats.head().items():
        print(f"    {name}: {count:,}")


def plot_rating_distribution(ax, df, rating_col):
    """Subplot (c): rating distribution pie chart."""
    if rating_col is None:
        ax.text(0.5, 0.5, "Rating column not found",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(c) Rating distribution")
        return

    rating_counts = df[rating_col].value_counts().sort_index()
    labels = [f"{int(r)} star{'s' if r != 1 else ''}" for r in rating_counts.index]
    sizes = rating_counts.values
    pct = sizes / sizes.sum() * 100

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=PIE_COLORS[: len(sizes)],
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)

    ax.set_title("(c) Rating distribution")

    print(f"\n  Rating distribution:")
    for label, count, p in zip(labels, sizes, pct):
        print(f"    {label}: {count:,} ({p:.1f}%)")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Figure 4a — Amazon Reviews 2023 Dataset Characteristics")
    print("=" * 60)

    # Load data
    amazon_path = find_amazon_file()
    df = load_amazon(amazon_path)
    text_col, cat_col, rating_col = detect_columns(df)

    # Summary stats
    print(f"\n[SUMMARY] Total rows in dataset: {len(df):,}")
    if "product_id" in df.columns or "asin" in df.columns:
        pid = "product_id" if "product_id" in df.columns else "asin"
        print(f"[SUMMARY] Unique products: {df[pid].nunique():,}")
    if "user_id" in df.columns or "reviewer_id" in df.columns:
        uid = "user_id" if "user_id" in df.columns else "reviewer_id"
        print(f"[SUMMARY] Unique users: {df[uid].nunique():,}")

    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Amazon Reviews 2023: Dataset Characteristics",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plot_review_length(axes[0], df, text_col)
    plot_top_categories(axes[1], df, cat_col)
    plot_rating_distribution(axes[2], df, rating_col)

    plt.tight_layout()

    # Save
    output_pdf = OUTPUT_DIR / "dataset_amazon.pdf"
    output_png = OUTPUT_DIR / "dataset_amazon.png"
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    print(f"\n[OK] Saved: {output_pdf}")
    print(f"[OK] Saved: {output_png}")
    plt.close()


if __name__ == "__main__":
    main()
