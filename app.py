"""
app.py
Streamlit Dataset Viewer — implements the flowchart:
Input File -> [Details / Columns / Description / Dtypes] -> Before-cleaning EDA
    -> Discrete Distribution  (Bar, Frequency Polygon, Stem plot, ECDF, Dot plot)
    -> Continuous Distribution (Histogram, KDE, Box, Violin, Ridge, CDF)
    -> Correlation Heatmap
    -> Outlier Plots
    -> Download EDA Report

Supports: CSV, TSV/TXT, Excel (.xls/.xlsx), JSON, Parquet

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import eda_utils as eu

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="EDA Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 EDA Dataset Viewer")

# ── Sidebar — file upload + sample dataset ───────────────
SUPPORTED = ["csv", "tsv", "txt", "xls", "xlsx", "xlsm", "xlsb", "json", "parquet", "pq"]

st.sidebar.header("📂 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=SUPPORTED,
    help="Supported formats: CSV, TSV/TXT, Excel, JSON, Parquet",
)

use_sample = st.sidebar.button("🧪 Use Sample Dataset (Iris)", use_container_width=True)

# Manage state for sample dataset
if use_sample:
    st.session_state["use_sample"] = True
    st.session_state.pop("uploaded_file_name", None)

if uploaded_file is not None:
    st.session_state["use_sample"] = False

# ── Load data ────────────────────────────────────────────
df = None
file_label = ""

if uploaded_file is not None:
    try:
        df = eu.load_file(uploaded_file)
        file_label = uploaded_file.name
    except Exception as exc:
        st.error(f"❌ Could not load file: {exc}")
        st.stop()

elif st.session_state.get("use_sample", False):
    df = eu.load_sample_dataset()
    file_label = "Sample Dataset (Iris)"

else:
    st.info("👈 Upload a dataset file or click **Use Sample Dataset** to begin.")
    st.markdown(
        """
        **Supported file formats:**
        | Format | Extensions |
        |--------|-----------|
        | CSV | `.csv` |
        | TSV / Text | `.tsv`, `.txt` |
        | Excel | `.xls`, `.xlsx`, `.xlsm`, `.xlsb` |
        | JSON | `.json` |
        | Parquet | `.parquet`, `.pq` |
        """
    )
    st.stop()

if df is None or df.empty:
    st.warning("The uploaded file resulted in an empty DataFrame.")
    st.stop()

st.sidebar.success(f"✅ **{file_label}**\n\n{df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Sidebar — Download Report ────────────────────────────
st.sidebar.divider()
st.sidebar.header("📥 Export")

report_text = eu.generate_report(df, filename=file_label)

st.sidebar.download_button(
    label="⬇️ Download EDA Report (.txt)",
    data=report_text,
    file_name=f"eda_report_{file_label.replace(' ', '_').split('.')[0]}.txt",
    mime="text/plain",
    use_container_width=True,
)

# Also offer CSV export of the raw data
csv_data = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="⬇️ Download Data as CSV",
    data=csv_data,
    file_name=f"{file_label.split('.')[0]}_export.csv",
    mime="text/csv",
    use_container_width=True,
)

# ── Data preview ─────────────────────────────────────────
with st.expander("🔎 Data Preview (first 100 rows)", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

# ── Overview tabs ────────────────────────────────────────
st.header("📋 Dataset Overview")

tab_details, tab_cols, tab_desc, tab_types = st.tabs(
    ["Dataset Details", "Columns", "Description", "Column Types"]
)

with tab_details:
    details = eu.get_dataset_details(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{details['Rows']:,}")
    col2.metric("Columns", details["Columns"])
    col3.metric("Memory", f"{details['Memory Usage (KB)']:,.1f} KB")

    col4, col5, col6 = st.columns(3)
    col4.metric("Total Cells", f"{details['Total Cells']:,}")
    col5.metric("Duplicate Rows", f"{details['Duplicate Rows']:,}")
    col6.metric("Missing Values", f"{details['Total Missing Values']:,}")

with tab_cols:
    st.dataframe(eu.get_column_list(df), use_container_width=True, hide_index=True)

with tab_desc:
    st.dataframe(eu.get_data_description(df), use_container_width=True)

with tab_types:
    ct = eu.get_column_types(df)
    st.dataframe(ct, use_container_width=True, hide_index=True)
    counts = ct["Inferred Category"].value_counts()
    st.caption("  |  ".join(f"**{cat}**: {n}" for cat, n in counts.items()))

st.divider()

# ── Before-cleaning EDA ─────────────────────────────────
st.header("🔍 Before-Cleaning EDA")

discrete_cols, continuous_cols = eu.split_columns_by_type(df)

eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(
    ["Discrete Distribution", "Continuous Distribution", "📊 Correlation Heatmap", "Outlier Plots"]
)


def _show_fig(fig):
    """Render a matplotlib figure and immediately close it to free memory."""
    st.pyplot(fig)
    plt.close(fig)


# ── Discrete ─────────────────────────────────────────────
with eda_tab1:
    if not discrete_cols:
        st.warning("No discrete numeric columns detected (≤ 20 unique values).")
    else:
        d_col = st.selectbox("Select a discrete column", discrete_cols, key="disc_col")

        plot_type = st.radio(
            "Plot type",
            ["Bar Chart", "Frequency Polygon", "Stem Plot (approx)", "ECDF", "Dot Plot"],
            horizontal=True,
            key="disc_plot",
        )

        dispatch_discrete = {
            "Bar Chart":          eu.plot_discrete_bar,
            "Frequency Polygon":  eu.plot_frequency_polygon,
            "Stem Plot (approx)": eu.plot_stem_leaf_approx,
            "ECDF":               eu.plot_ecdf,
            "Dot Plot":           eu.plot_dot_plot,
        }

        _show_fig(dispatch_discrete[plot_type](df, d_col))

# ── Continuous ───────────────────────────────────────────
with eda_tab2:
    if not continuous_cols:
        st.warning("No continuous numeric columns detected (> 20 unique values).")
    else:
        c_col = st.selectbox("Select a continuous column", continuous_cols, key="cont_col")

        plot_type = st.radio(
            "Plot type",
            ["Histogram", "KDE", "Box Plot", "Violin Plot", "Ridge (multi-col)", "CDF"],
            horizontal=True,
            key="cont_plot",
        )

        if plot_type == "Ridge (multi-col)":
            multi = st.multiselect(
                "Columns to overlay",
                continuous_cols,
                default=continuous_cols[:3],
                key="ridge_cols",
            )
            if multi:
                _show_fig(eu.plot_ridge_approx(df, multi))
            else:
                st.info("Select at least one column for the ridge plot.")
        else:
            dispatch_continuous = {
                "Histogram":    eu.plot_histogram,
                "KDE":          eu.plot_kde,
                "Box Plot":     eu.plot_box,
                "Violin Plot":  eu.plot_violin,
                "CDF":          eu.plot_cdf,
            }
            _show_fig(dispatch_continuous[plot_type](df, c_col))

# ── Correlation Heatmap ──────────────────────────────────
with eda_tab3:
    st.subheader("Correlation Heatmap")

    corr_method = st.radio(
        "Correlation method",
        ["pearson", "spearman", "kendall"],
        horizontal=True,
        key="corr_method",
    )

    _show_fig(eu.plot_correlation_heatmap(df, method=corr_method))

    # Show the raw correlation table below
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        with st.expander("📋 View Correlation Table"):
            corr_table = numeric_df.corr(method=corr_method).round(3)
            st.dataframe(corr_table, use_container_width=True)

# ── Outliers ─────────────────────────────────────────────
with eda_tab4:
    numeric_cols = discrete_cols + continuous_cols
    if not numeric_cols:
        st.warning("No numeric columns available for outlier analysis.")
    else:
        o_col = st.selectbox("Select a column", numeric_cols, key="outlier_col")

        left, right = st.columns([1.2, 1])

        with left:
            _show_fig(eu.plot_outlier_box(df, o_col))

        with right:
            st.subheader("IQR Outlier Summary")
            summary = eu.get_outlier_summary(df, o_col)
            st.table(
                pd.DataFrame(summary.items(), columns=["Metric", "Value"])
            )
