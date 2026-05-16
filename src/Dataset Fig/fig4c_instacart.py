"""
Figure 4c — Instacart Dataset Characteristics
==============================================

Generates a publication-quality figure with 3 subplots showing:
  (a) Basket-size distribution (items per order)
  (b) Top 10 departments by purchase count
  (c) Days-since-prior-order distribution (reorder behavior)

Usage:
    cd /Users/bhanutejamalineni/Thesis
    python figures/fig4c_instacart.py

Output:
    figures/dataset_instacart.pdf
    Console: summary statistics for verification against Table 4

Prerequisites:
    pip install pandas matplotlib seaborn pyarrow
"""

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

# Instacart often has multiple files (orders + order_products + products)
ORDERS_CANDIDATES = [
    "ic_orders.parquet",            # actual file name on your machine
    "instacart_orders.parquet",
    "instacart_orders_clean.parquet",
    "orders.parquet",
    "instacart.parquet",
]
PRODUCTS_CANDIDATES = [
    "ic_basket.parquet",            # actual file name on your machine
    "instacart_order_products.parquet",
    "order_products.parquet",
    "instacart_basket.parquet",
    "instacart_items.parquet",
]
DEPT_CANDIDATES = [
    "ic_products.parquet",          # actual file name on your machine
    "ic_product_features.parquet",  # fallback
    "instacart_departments.parquet",
    "departments.parquet",
    "instacart_products.parquet",
    "products.parquet",
]

# Color palette - green family for Instacart
COLOR_PRIMARY = "#388E3C"
COLOR_SECONDARY = "#66BB6A"
COLOR_ACCENT = "#1B5E20"

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
def find_file(candidates, label):
    for candidate in candidates:
        path = DATA_DIR / candidate
        if path.exists():
            print(f"[OK] Found {label}: {path}")
            return path
    print(f"[WARN] No {label} parquet found.")
    return None


def find_basket_size_data():
    """Find a parquet that has basket-size info or can compute it."""
    products_path = find_file(PRODUCTS_CANDIDATES, "order_products")
    if products_path:
        return products_path, "order_products"

    orders_path = find_file(ORDERS_CANDIDATES, "orders")
    if orders_path:
        return orders_path, "orders"

    return None, None


# ============================================================
# PLOTTING
# ============================================================
def plot_basket_size(ax, df):
    """Subplot (a): basket-size distribution."""
    # If we have order_products data, count items per order
    if "order_id" in df.columns and "product_id" in df.columns:
        basket = df.groupby("order_id").size()
        print(f"\n  Basket size computed from order_id × product_id")
    elif "basket_size" in df.columns:
        basket = df["basket_size"]
        print(f"\n  Basket size from precomputed column")
    elif "n_items" in df.columns:
        basket = df["n_items"]
    elif "order_id" in df.columns:
        basket = df.groupby("order_id").size()
    else:
        ax.text(0.5, 0.5, "Basket-size data not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(a) Basket-size distribution")
        return

    # Trim outliers
    upper = basket.quantile(0.99)
    basket_plot = basket[basket <= upper]

    ax.hist(basket_plot, bins=40, color=COLOR_PRIMARY,
            edgecolor=COLOR_ACCENT, alpha=0.85)
    ax.axvline(basket.median(), color=COLOR_ACCENT, linestyle="--", linewidth=1.5,
               label=f"Median = {basket.median():.0f} items")
    ax.axvline(basket.mean(), color="#D32F2F", linestyle=":", linewidth=1.5,
               label=f"Mean = {basket.mean():.1f} items")
    ax.set_xlabel("Items per order")
    ax.set_ylabel("Number of orders")
    ax.set_title("(a) Basket-size distribution")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    print(f"  Basket size stats:")
    print(f"    Mean:   {basket.mean():.1f}")
    print(f"    Median: {basket.median():.1f}")
    print(f"    Total orders: {len(basket):,}")


def plot_top_departments(ax, df):
    """Subplot (b): top 10 departments by purchase count."""
    dept_col = None
    for key in ["department", "dept", "category", "main_category"]:
        if key in df.columns:
            dept_col = key
            break

    if dept_col is None:
        # Try loading from separate departments file
        dept_path = find_file(DEPT_CANDIDATES, "departments")
        if dept_path:
            dept_df = pd.read_parquet(dept_path)
            print(f"  Loaded departments: {len(dept_df)} rows")
            if "department" in dept_df.columns and "product_id" in df.columns:
                # Merge
                df = df.merge(dept_df, on="product_id", how="left")
                dept_col = "department"

    if dept_col is None:
        ax.text(0.5, 0.5, "Department data not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(b) Top 10 departments")
        return

    top_depts = df[dept_col].value_counts().head(10)

    bars = ax.barh(range(len(top_depts)), top_depts.values,
                   color=COLOR_PRIMARY, edgecolor=COLOR_ACCENT, alpha=0.85)
    ax.set_yticks(range(len(top_depts)))
    ax.set_yticklabels([str(d)[:25] for d in top_depts.index])
    ax.invert_yaxis()
    ax.set_xlabel("Number of purchases")
    ax.set_title("(b) Top 10 departments")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, top_depts.values):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f" {val:,}", va="center", fontsize=8, color=COLOR_ACCENT)

    print(f"\n  Top 5 departments:")
    for name, count in top_depts.head().items():
        print(f"    {name}: {count:,}")


def plot_reorder_timing(ax, df_orders):
    """Subplot (c): days-since-prior-order distribution."""
    col = None
    for key in ["days_since_prior_order", "days_since_prior", "days_since_last_order"]:
        if key in df_orders.columns:
            col = key
            break

    if col is None:
        ax.text(0.5, 0.5, "Reorder-timing data not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(c) Days since prior order")
        return

    days = df_orders[col].dropna()

    ax.hist(days, bins=30, color=COLOR_PRIMARY,
            edgecolor=COLOR_ACCENT, alpha=0.85)
    ax.axvline(days.median(), color=COLOR_ACCENT, linestyle="--", linewidth=1.5,
               label=f"Median = {days.median():.0f} days")
    ax.axvline(days.mean(), color="#D32F2F", linestyle=":", linewidth=1.5,
               label=f"Mean = {days.mean():.1f} days")
    ax.set_xlabel("Days since prior order")
    ax.set_ylabel("Number of orders")
    ax.set_title("(c) Reorder-timing distribution")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    print(f"\n  Reorder-timing stats:")
    print(f"    Median: {days.median():.1f} days")
    print(f"    Mean:   {days.mean():.1f} days")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Figure 4c — Instacart Dataset Characteristics")
    print("=" * 60)

    # Try to load both orders and order_products files
    orders_path = find_file(ORDERS_CANDIDATES, "orders")
    products_path = find_file(PRODUCTS_CANDIDATES, "order_products")

    if orders_path is None and products_path is None:
        print(f"[ERROR] No Instacart parquet found in {DATA_DIR}")
        print("Files available:")
        for f in sorted(DATA_DIR.glob("*.parquet")):
            print(f"  - {f.name}")
        raise FileNotFoundError("Could not find Instacart parquet files.")

    df_orders = pd.read_parquet(orders_path) if orders_path else None
    df_products = pd.read_parquet(products_path) if products_path else None

    if df_orders is not None:
        print(f"[OK] Orders: {len(df_orders):,} rows, columns: {list(df_orders.columns)[:8]}")
    if df_products is not None:
        print(f"[OK] Order_products: {len(df_products):,} rows, columns: {list(df_products.columns)[:8]}")

    # Pick the right dataframe for each plot
    basket_df = df_products if df_products is not None else df_orders
    dept_df = df_products if df_products is not None else df_orders
    timing_df = df_orders if df_orders is not None else df_products

    print(f"\n[SUMMARY]")
    if df_orders is not None:
        print(f"  Total orders: {len(df_orders):,}")
        if "user_id" in df_orders.columns:
            print(f"  Unique users: {df_orders['user_id'].nunique():,}")
    if df_products is not None:
        if "product_id" in df_products.columns:
            print(f"  Unique products: {df_products['product_id'].nunique():,}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Instacart: Dataset Characteristics",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plot_basket_size(axes[0], basket_df)
    plot_top_departments(axes[1], dept_df)
    plot_reorder_timing(axes[2], timing_df)

    plt.tight_layout()

    output_pdf = OUTPUT_DIR / "dataset_instacart.pdf"
    output_png = OUTPUT_DIR / "dataset_instacart.png"
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    print(f"\n[OK] Saved: {output_pdf}")
    print(f"[OK] Saved: {output_png}")
    plt.close()


if __name__ == "__main__":
    main()
