"""
Figure 4b — RetailRocket Dataset Characteristics
=================================================

Generates a publication-quality figure with 3 subplots showing:
  (a) Event-type distribution (view/addtocart/transaction) as pie chart
  (b) Daily event count time-series over the data window
  (c) Session-length distribution (events per session)

Usage:
    cd /Users/bhanutejamalineni/Thesis
    python figures/fig4b_retailrocket.py

Output:
    figures/dataset_retailrocket.pdf
    Console: summary statistics for verification against Table 3

Prerequisites:
    pip install pandas matplotlib seaborn pyarrow
"""

import warnings
from pathlib import Path

import matplotlib.dates as mdates
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

RR_CANDIDATES = [
    "rr_events.parquet",
    "retailrocket_events.parquet",
    "retailrocket.parquet",
    "events.parquet",
    "rr_filtered.parquet",
    "rr_prepared.parquet",
]

# Color palette - orange family for RetailRocket
COLOR_PRIMARY = "#F57C00"
COLOR_SECONDARY = "#FFB74D"
COLOR_ACCENT = "#E65100"
EVENT_COLORS = {
    "view": "#1976D2",
    "addtocart": "#F57C00",
    "transaction": "#388E3C",
}

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
def find_rr_file():
    for candidate in RR_CANDIDATES:
        path = DATA_DIR / candidate
        if path.exists():
            print(f"[OK] Found RetailRocket data: {path}")
            return path
    print(f"[ERROR] No RetailRocket parquet found in {DATA_DIR}")
    print("Files available:")
    for f in sorted(DATA_DIR.glob("*.parquet")):
        print(f"  - {f.name}")
    raise FileNotFoundError("Could not find RetailRocket parquet.")


def detect_columns(df):
    cols = {col.lower(): col for col in df.columns}

    # Event-type column
    event_col = None
    for key in ["event", "event_type", "action", "interaction_type"]:
        if key in cols:
            event_col = cols[key]
            break

    # Timestamp column
    ts_col = None
    for key in ["timestamp", "ts", "time", "datetime", "event_time"]:
        if key in cols:
            ts_col = cols[key]
            break

    # Visitor column
    visitor_col = None
    for key in ["visitorid", "visitor_id", "user_id", "userid", "session_id"]:
        if key in cols:
            visitor_col = cols[key]
            break

    print(f"[OK] Event column: {event_col}")
    print(f"[OK] Timestamp column: {ts_col}")
    print(f"[OK] Visitor/session column: {visitor_col}")

    return event_col, ts_col, visitor_col


# ============================================================
# PLOTTING
# ============================================================
def plot_event_distribution(ax, df, event_col):
    """Subplot (a): event-type distribution pie chart."""
    if event_col is None:
        ax.text(0.5, 0.5, "Event column not found",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(a) Event-type distribution")
        return

    event_counts = df[event_col].value_counts()
    labels = list(event_counts.index)
    sizes = event_counts.values
    colors = [EVENT_COLORS.get(str(lbl).lower(), COLOR_PRIMARY) for lbl in labels]

    # Compute percentages
    pct = sizes / sizes.sum() * 100

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[f"{l}\n({c:,})" for l, c in zip(labels, sizes)],
        autopct="%1.2f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9},
        pctdistance=0.75,
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(9)

    ax.set_title("(a) Event-type distribution")

    print(f"\n  Event distribution:")
    for label, count, p in zip(labels, sizes, pct):
        print(f"    {label}: {count:,} ({p:.2f}%)")


def plot_daily_activity(ax, df, ts_col):
    """Subplot (b): daily event count over time."""
    if ts_col is None:
        ax.text(0.5, 0.5, "Timestamp column not found",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(b) Daily activity")
        return

    # Convert timestamp - try every plausible unit, pick the one giving
    # dates in a sane retail range (2000-2030). Handles ns/us/ms/s transparently.
    ts = pd.to_numeric(df[ts_col], errors="coerce")
    ts_sample = ts.dropna().iloc[0]
    print(f"  [DEBUG] raw timestamp sample: {ts_sample:.6e}")

    def try_unit(values, divisor, unit):
        try:
            result = pd.to_datetime(values // divisor, unit=unit, errors="coerce")
            valid = result.dropna()
            if len(valid) == 0:
                return None
            yr_min, yr_max = valid.dt.year.min(), valid.dt.year.max()
            if 2000 <= yr_min and yr_max <= 2030:
                return result
        except Exception:
            pass
        return None

    df["_dt"] = None
    for divisor, unit, label in [
        (1_000_000, "ms", "nanoseconds"),
        (1_000,     "ms", "microseconds"),
        (1,         "ms", "milliseconds"),
        (1,         "s",  "seconds"),
    ]:
        converted = try_unit(ts, divisor, unit)
        if converted is not None:
            df["_dt"] = converted
            print(f"  [DEBUG] timestamp detected as {label}")
            break
    else:
        df["_dt"] = pd.to_datetime(df[ts_col], errors="coerce")
        print("  [DEBUG] timestamp: pandas auto-infer fallback")

    daily = df.groupby(df["_dt"].dt.date).size()
    daily.index = pd.to_datetime(daily.index)

    ax.plot(daily.index, daily.values, color=COLOR_PRIMARY,
            linewidth=1.2, alpha=0.9)
    ax.fill_between(daily.index, daily.values, color=COLOR_SECONDARY, alpha=0.3)

    # Rolling 7-day average
    smooth = daily.rolling(window=7, center=True).mean()
    ax.plot(smooth.index, smooth.values, color=COLOR_ACCENT,
            linewidth=2, label="7-day moving average")

    ax.set_xlabel("Date")
    ax.set_ylabel("Events per day")
    ax.set_title("(b) Daily activity")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    print(f"\n  Time range: {daily.index.min().date()} to {daily.index.max().date()}")
    print(f"  Total days: {len(daily)}")
    print(f"  Mean events/day: {daily.mean():.0f}")
    print(f"  Peak day: {daily.idxmax().date()} with {daily.max():,} events")

    # Cleanup
    df.drop(columns=["_dt"], inplace=True)


def plot_session_length(ax, df, visitor_col, ts_col):
    """Subplot (c): session-length distribution (events per session)."""
    if visitor_col is None:
        ax.text(0.5, 0.5, "Visitor column not found",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("(c) Session-length distribution")
        return

    # Events per visitor (proxy for session length)
    session_lengths = df.groupby(visitor_col).size()

    # Trim outliers for cleaner visualization
    upper = session_lengths.quantile(0.99)
    sl_plot = session_lengths[session_lengths <= upper]

    ax.hist(sl_plot, bins=50, color=COLOR_PRIMARY,
            edgecolor=COLOR_ACCENT, alpha=0.85)
    ax.axvline(session_lengths.median(), color=COLOR_ACCENT,
               linestyle="--", linewidth=1.5,
               label=f"Median = {session_lengths.median():.0f} events")
    ax.axvline(session_lengths.mean(), color="#D32F2F",
               linestyle=":", linewidth=1.5,
               label=f"Mean = {session_lengths.mean():.1f} events")

    ax.set_xlabel("Events per visitor")
    ax.set_ylabel("Number of visitors")
    ax.set_title("(c) Session-length distribution")
    ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    print(f"\n  Session length stats:")
    print(f"    Total visitors: {len(session_lengths):,}")
    print(f"    Median: {session_lengths.median():.1f}")
    print(f"    Mean:   {session_lengths.mean():.1f}")
    print(f"    Max:    {session_lengths.max()}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Figure 4b — RetailRocket Dataset Characteristics")
    print("=" * 60)

    rr_path = find_rr_file()
    df = pd.read_parquet(rr_path)
    print(f"[OK] Loaded {len(df):,} rows, columns: {list(df.columns)}")

    event_col, ts_col, visitor_col = detect_columns(df)

    print(f"\n[SUMMARY] Total events: {len(df):,}")
    if visitor_col:
        print(f"[SUMMARY] Unique visitors: {df[visitor_col].nunique():,}")
    if "itemid" in df.columns:
        print(f"[SUMMARY] Unique items: {df['itemid'].nunique():,}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "RetailRocket: Dataset Characteristics",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plot_event_distribution(axes[0], df, event_col)
    plot_daily_activity(axes[1], df, ts_col)
    plot_session_length(axes[2], df, visitor_col, ts_col)

    plt.tight_layout()

    output_pdf = OUTPUT_DIR / "dataset_retailrocket.pdf"
    output_png = OUTPUT_DIR / "dataset_retailrocket.png"
    plt.savefig(output_pdf)
    plt.savefig(output_png)
    print(f"\n[OK] Saved: {output_pdf}")
    print(f"[OK] Saved: {output_png}")
    plt.close()


if __name__ == "__main__":
    main()
