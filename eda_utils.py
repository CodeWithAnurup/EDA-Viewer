"""
eda_utils.py
Core EDA logic — dataset summaries + plotting functions.
Kept independent of Streamlit so it can be tested in a notebook first.

Supports: CSV, TSV, Excel (.xls/.xlsx), JSON, Parquet
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------------------------------------------------
# 0. FILE LOADING — any common tabular format
# ---------------------------------------------------------

def load_file(uploaded_file) -> pd.DataFrame:
    """
    Read a file-like object into a DataFrame.
    Detects format from the file name extension.
    Raises ValueError for unsupported formats.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep="\t")
    elif name.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    elif name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(
            f"Unsupported file type: {name.rsplit('.', 1)[-1]}. "
            "Please upload CSV, TSV/TXT, Excel, JSON, or Parquet."
        )


# ---------------------------------------------------------
# 1. DATASET OVERVIEW FUNCTIONS
# ---------------------------------------------------------

def get_dataset_details(df: pd.DataFrame) -> dict:
    """Shape, size, memory usage, duplicate count, missing count."""
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Total Cells": df.size,
        "Memory Usage (KB)": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Total Missing Values": int(df.isnull().sum().sum()),
    }


def get_column_list(df: pd.DataFrame) -> pd.DataFrame:
    """Columns as-is, in original order, with dtype and missing count."""
    info = []
    for i, col in enumerate(df.columns):
        info.append({
            "Position": i,
            "Column Name": col,
            "Dtype": str(df[col].dtype),
            "Non-Null Count": int(df[col].notna().sum()),
            "Missing": int(df[col].isna().sum()),
        })
    return pd.DataFrame(info)


def get_data_description(df: pd.DataFrame) -> pd.DataFrame:
    """Wraps df.describe(include='all') with transpose for readability."""
    return df.describe(include="all").transpose()


def get_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Dtype + inferred category (Discrete/Continuous/Categorical/Datetime)."""
    rows = []
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()

        if pd.api.types.is_numeric_dtype(dtype):
            category = "Discrete" if nunique <= 20 else "Continuous"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            category = "Datetime"
        else:
            category = "Categorical/Text"

        rows.append({
            "Column": col,
            "Dtype": str(dtype),
            "Unique Values": nunique,
            "Inferred Category": category,
        })
    return pd.DataFrame(rows)


def split_columns_by_type(df: pd.DataFrame):
    """Returns (discrete_cols, continuous_cols) based on the ≤20-unique heuristic."""
    discrete, continuous = [], []
    for col in df.select_dtypes(include="number").columns:
        if df[col].nunique() <= 20:
            discrete.append(col)
        else:
            continuous.append(col)
    return discrete, continuous


# ---------------------------------------------------------
# Helper — close figure after returning (caller displays it)
# ---------------------------------------------------------

def _finalise(fig):
    """Tight layout + return. Caller must plt.close() after rendering."""
    fig.tight_layout()
    return fig


# ---------------------------------------------------------
# 2. DISCRETE DISTRIBUTION PLOTS
# ---------------------------------------------------------

def plot_discrete_bar(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df[col].value_counts().sort_index()
    ax.bar([str(v) for v in counts.index], counts.values, color="#4C72B0")
    ax.set_title(f"Discrete Bar Chart — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    return _finalise(fig)


def plot_frequency_polygon(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df[col].value_counts().sort_index()
    ax.plot(counts.index.astype(float), counts.values, marker="o", color="#DD8452")
    ax.set_title(f"Frequency Polygon — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    return _finalise(fig)


def plot_stem_leaf_approx(df: pd.DataFrame, col: str, max_points: int = 500):
    """Approximate stem plot; capped at `max_points` to keep rendering fast."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = df[col].dropna().values
    if len(data) > max_points:
        data = np.random.default_rng(42).choice(data, max_points, replace=False)
    data = np.sort(data)
    ax.stem(range(len(data)), data)
    ax.set_title(f"Stem Plot (approx) — {col}")
    ax.set_ylabel(col)
    ax.set_xlabel("Index (sorted)")
    return _finalise(fig)


def plot_ecdf(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = np.sort(df[col].dropna().values)
    y = np.arange(1, len(data) + 1) / len(data)
    ax.step(data, y, where="post", color="#55A868")
    ax.set_title(f"ECDF — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Cumulative Probability")
    return _finalise(fig)


def plot_dot_plot(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df[col].value_counts().sort_index()
    # limit to top 50 values for readability
    if len(counts) > 50:
        counts = counts.nlargest(50).sort_index()
    for val, cnt in counts.items():
        ax.plot([val] * cnt, range(1, cnt + 1), "o", color="#C44E52", markersize=4)
    ax.set_title(f"Dot Plot — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Stacked Count")
    return _finalise(fig)


# ---------------------------------------------------------
# 3. CONTINUOUS DISTRIBUTION PLOTS
# ---------------------------------------------------------

def plot_histogram(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[col].dropna(), kde=False, ax=ax, color="#4C72B0")
    ax.set_title(f"Histogram — {col}")
    return _finalise(fig)


def plot_kde(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = df[col].dropna()
    if data.nunique() < 2:
        ax.text(0.5, 0.5, "Not enough unique values for KDE",
                ha="center", va="center", transform=ax.transAxes)
    else:
        sns.kdeplot(data, ax=ax, fill=True, color="#DD8452")
    ax.set_title(f"KDE Plot — {col}")
    return _finalise(fig)


def plot_box(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x=df[col].dropna(), ax=ax, color="#55A868")
    ax.set_title(f"Box Plot — {col}")
    return _finalise(fig)


def plot_violin(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = df[col].dropna()
    if data.nunique() < 2:
        ax.text(0.5, 0.5, "Not enough unique values for violin",
                ha="center", va="center", transform=ax.transAxes)
    else:
        sns.violinplot(x=data, ax=ax, color="#C44E52")
    ax.set_title(f"Violin Plot — {col}")
    return _finalise(fig)


def plot_ridge_approx(df: pd.DataFrame, cols: list):
    """Simplified ridge plot: overlapping KDEs for multiple continuous columns."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in cols:
        data = df[c].dropna()
        if data.nunique() >= 2:
            sns.kdeplot(data, ax=ax, fill=True, alpha=0.35, label=c)
    ax.set_title("Ridge-style Overlay (multi-column KDE)")
    ax.legend(fontsize="small")
    return _finalise(fig)


def plot_cdf(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = np.sort(df[col].dropna().values)
    y = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, y, color="#8172B2")
    ax.set_title(f"CDF — {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Cumulative Probability")
    return _finalise(fig)


# ---------------------------------------------------------
# 4. OUTLIER ANALYSIS
# ---------------------------------------------------------

def plot_outlier_box(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x=df[col].dropna(), ax=ax, color="#F5B041")
    ax.set_title(f"Outlier Check (Box) — {col}")
    return _finalise(fig)


def get_outlier_summary(df: pd.DataFrame, col: str) -> dict:
    """IQR-based outlier count."""
    data = df[col].dropna()
    if len(data) == 0:
        return {"Error": "Column has no non-null values"}
    q1, q3 = data.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = data[(data < lower) | (data > upper)]
    return {
        "Q1": round(float(q1), 3),
        "Q3": round(float(q3), 3),
        "IQR": round(float(iqr), 3),
        "Lower Bound": round(float(lower), 3),
        "Upper Bound": round(float(upper), 3),
        "Outlier Count": len(outliers),
        "Outlier %": round(100 * len(outliers) / len(data), 2),
    }


# ---------------------------------------------------------
# 5. CORRELATION HEATMAP
# ---------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, method: str = "pearson"):
    """
    Correlation heatmap for all numeric columns.
    method: 'pearson', 'spearman', or 'kendall'
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Need at least 2 numeric columns",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title("Correlation Heatmap")
        return _finalise(fig)

    corr = numeric_df.corr(method=method)
    size = max(8, len(corr.columns) * 0.6)
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"Correlation Heatmap ({method.title()})", fontsize=14)
    return _finalise(fig)


# ---------------------------------------------------------
# 6. REPORT GENERATION
# ---------------------------------------------------------

def generate_report(df: pd.DataFrame, filename: str = "dataset") -> str:
    """
    Generate a plain-text EDA summary report suitable for download.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"  EDA REPORT — {filename}")
    lines.append("=" * 60)
    lines.append("")

    # Dataset details
    details = get_dataset_details(df)
    lines.append("DATASET OVERVIEW")
    lines.append("-" * 40)
    for k, v in details.items():
        lines.append(f"  {k:<25} {v}")
    lines.append("")

    # Column types
    lines.append("COLUMN TYPES")
    lines.append("-" * 40)
    ct = get_column_types(df)
    lines.append(ct.to_string(index=False))
    lines.append("")

    # Statistical description
    lines.append("STATISTICAL DESCRIPTION")
    lines.append("-" * 40)
    desc = get_data_description(df)
    lines.append(desc.to_string())
    lines.append("")

    # Correlation matrix
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        lines.append("CORRELATION MATRIX (Pearson)")
        lines.append("-" * 40)
        corr = numeric_df.corr(method="pearson")
        lines.append(corr.round(3).to_string())
        lines.append("")

    # Outlier summary for each numeric column
    lines.append("OUTLIER SUMMARY (IQR method)")
    lines.append("-" * 40)
    for col in numeric_df.columns:
        summary = get_outlier_summary(df, col)
        lines.append(f"  Column: {col}")
        for k, v in summary.items():
            lines.append(f"    {k:<20} {v}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("  End of Report")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------
# 7. SAMPLE DATASET
# ---------------------------------------------------------

def load_sample_dataset() -> pd.DataFrame:
    """
    Return a built-in sample dataset (subset of Iris-style data)
    so the app can be demoed without uploading a file.
    """
    np.random.seed(42)
    n = 150
    species = np.repeat(["Setosa", "Versicolor", "Virginica"], 50)

    data = pd.DataFrame({
        "sepal_length": np.concatenate([
            np.random.normal(5.0, 0.35, 50),
            np.random.normal(5.9, 0.52, 50),
            np.random.normal(6.6, 0.64, 50),
        ]),
        "sepal_width": np.concatenate([
            np.random.normal(3.4, 0.38, 50),
            np.random.normal(2.8, 0.31, 50),
            np.random.normal(3.0, 0.32, 50),
        ]),
        "petal_length": np.concatenate([
            np.random.normal(1.5, 0.17, 50),
            np.random.normal(4.3, 0.47, 50),
            np.random.normal(5.6, 0.55, 50),
        ]),
        "petal_width": np.concatenate([
            np.random.normal(0.2, 0.10, 50),
            np.random.normal(1.3, 0.20, 50),
            np.random.normal(2.0, 0.27, 50),
        ]),
        "species": species,
        "rating": np.random.choice(range(1, 6), n),  # discrete column (5 unique)
    })

    # Round numeric columns for realism
    for c in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        data[c] = data[c].round(1)

    # Sprinkle a few NaN values to make it realistic
    for c in ["sepal_width", "petal_width"]:
        idx = np.random.choice(n, 5, replace=False)
        data.loc[idx, c] = np.nan

    return data
