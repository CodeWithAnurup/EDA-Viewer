# EDA Viewer

A Streamlit-based Exploratory Data Analysis tool that provides instant statistical summaries, distribution plots, correlation analysis, and outlier detection for any tabular dataset.

Built as a reusable, pre-cleaning EDA stage — upload a file and get a complete visual and statistical overview in seconds.

---

## Problem Statement

Before any data cleaning or modeling, analysts need to understand what they are working with: the shape of the data, distribution characteristics, correlations between variables, and the presence of outliers. This step is often done manually with repetitive boilerplate code in notebooks.

**EDA Viewer** automates this initial exploration stage. It accepts multiple file formats, classifies columns automatically, and generates the right visualizations based on data type — eliminating the setup overhead so analysts can focus on interpretation.

---

## Features

- **Multi-format support** — CSV, TSV, Excel (.xls/.xlsx), JSON, Parquet
- **Dataset overview** — shape, memory usage, duplicates, missing values, column types
- **Automatic column classification** — Discrete (≤20 unique values), Continuous, Categorical, Datetime
- **Distribution plots** — Bar, Frequency Polygon, Stem Plot, ECDF, Dot Plot (discrete); Histogram, KDE, Box, Violin, Ridge, CDF (continuous)
- **Correlation heatmap** — Pearson, Spearman, Kendall with annotated triangular heatmap
- **Outlier detection** — IQR-based box plot with statistical summary (Q1, Q3, bounds, count, percentage)
- **Report export** — Download a full-text EDA summary report or export data as CSV
- **Sample dataset** — Built-in Iris-style dataset for instant demo without file upload

---

## Project Structure

```
EDA-Viewer/
├── app.py              # Streamlit UI — wires widgets to eda_utils functions
├── eda_utils.py        # Pure logic — all EDA computations and plot functions
├── requirements.txt    # Python dependencies
└── README.md
```

### Architecture

The project follows a clean separation of concerns:

- **`eda_utils.py`** contains all data processing and plotting logic with zero Streamlit dependency. Every function is independently testable in a notebook or script.
- **`app.py`** is the Streamlit interface layer that imports `eda_utils` and maps its functions to UI widgets (tabs, selectors, radio buttons).

---

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        FILE UPLOAD                              │
│   Upload CSV / TSV / Excel / JSON / Parquet                     │
│   OR click "Use Sample Dataset"                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATASET OVERVIEW                            │
│   Tab 1: Details (rows, columns, memory, duplicates, missing)   │
│   Tab 2: Column listing with dtype and null counts              │
│   Tab 3: Statistical description (describe)                     │
│   Tab 4: Column type classification                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┬───────────────┐
              ▼            ▼            ▼               ▼
     ┌──────────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────┐
     │   Discrete   │ │Continuous│ │ Correlation  │ │ Outlier │
     │ Distribution │ │Distribution│ │  Heatmap   │ │  Plots  │
     │              │ │          │ │              │ │         │
     │ Bar Chart    │ │ Histogram│ │ Pearson      │ │ Box Plot│
     │ Freq Polygon │ │ KDE      │ │ Spearman     │ │ IQR     │
     │ Stem Plot    │ │ Box Plot │ │ Kendall      │ │ Summary │
     │ ECDF         │ │ Violin   │ │              │ │         │
     │ Dot Plot     │ │ Ridge    │ │              │ │         │
     │              │ │ CDF      │ │              │ │         │
     └──────────────┘ └──────────┘ └──────────────┘ └─────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │     EXPORT OPTIONS     │
              │  Download EDA Report   │
              │  Download Data as CSV  │
              └────────────────────────┘
```

---

## Setup and Usage

### Prerequisites

- Python 3.9 or higher

### Installation

```bash
git clone https://github.com/CodeWithAnurup/EDA-Viewer.git
cd EDA-Viewer
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload any supported file or click **Use Sample Dataset** to start exploring.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| pandas | Data manipulation |
| numpy | Numerical operations |
| matplotlib | Plotting backend |
| seaborn | Statistical visualizations |
| openpyxl | Excel file support |
| pyarrow | Parquet file support |

---

## Scope

This tool covers the **pre-cleaning EDA stage** — understanding raw data before any transformations. It does not include data cleaning, feature engineering, or model building. These stages can be added as future extensions.

---

## License

MIT
