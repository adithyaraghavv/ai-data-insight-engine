import os
import io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.data_quality import (
    ai_agent_data_quality,
    fuzzy_match_columns,
)

load_dotenv()


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=api_key)


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_null_comparison(df1, df2):
    """Grouped bar chart of null counts per common column."""
    cols = list(set(df1.columns) & set(df2.columns))
    nulls1 = [int(df1[c].isnull().sum()) for c in cols]
    nulls2 = [int(df2[c].isnull().sum()) for c in cols]

    if all(v == 0 for v in nulls1 + nulls2):
        return None

    x = range(len(cols))
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.8), 4))
    w = 0.35
    ax.bar([i - w / 2 for i in x], nulls1, w, label="Dataset 1", color="#2196F3")
    ax.bar([i + w / 2 for i in x], nulls2, w, label="Dataset 2", color="#4CAF50")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_ylabel("Null Count")
    ax.set_title("Missing Values per Column")
    ax.legend()
    plt.tight_layout()
    return fig


def chart_boxplot(df1, df2, col1, col2):
    """Side-by-side box plot for a matched numeric column pair."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df1[col1].dropna().tolist(), df2[col2].dropna().tolist()]
    bp = ax.boxplot(data, patch_artist=True, labels=["Dataset 1", "Dataset 2"])
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][1].set_facecolor("#4CAF50")
    ax.set_title(f"Distribution: {col1}")
    ax.set_ylabel("Value")
    plt.tight_layout()
    return fig


def chart_single_nulls(df):
    """Horizontal bar chart of null counts for a single dataset."""
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values()
    if null_counts.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, max(3, len(null_counts) * 0.5)))
    ax.barh(null_counts.index.tolist(), null_counts.values, color="#EF5350")
    ax.set_xlabel("Null Count")
    ax.set_title("Missing Values per Column")
    plt.tight_layout()
    return fig


def chart_dtypes_pie(df):
    """Pie chart of column data type breakdown."""
    type_counts = df.dtypes.apply(lambda x: x.kind).value_counts()
    label_map = {"f": "Float", "i": "Integer", "O": "Object/String",
                 "b": "Boolean", "M": "Datetime", "U": "Unicode"}
    labels = [label_map.get(k, k) for k in type_counts.index]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(type_counts.values, labels=labels, autopct="%1.0f%%",
           colors=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"])
    ax.set_title("Column Data Types")
    plt.tight_layout()
    return fig


def chart_histogram(df, col):
    """Histogram for a numeric column."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col].dropna(), bins=20, color="#7E57C2", edgecolor="white")
    ax.set_title(f"Distribution: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def generate_charts_two(df1, df2, matched_columns):
    charts = []

    null_chart = chart_null_comparison(df1, df2)
    if null_chart:
        charts.append(null_chart)

    numeric_matches = [
        (c1, c2) for c1, c2 in matched_columns.items()
        if pd.api.types.is_numeric_dtype(df1[c1]) and pd.api.types.is_numeric_dtype(df2[c2])
    ]
    for col1, col2 in numeric_matches[:3]:
        charts.append(chart_boxplot(df1, df2, col1, col2))

    return charts


def generate_charts_single(df):
    charts = []

    null_chart = chart_single_nulls(df)
    if null_chart:
        charts.append(null_chart)

    charts.append(chart_dtypes_pie(df))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols[:2]:
        charts.append(chart_histogram(df, col))

    return charts


# ── AI Summary ────────────────────────────────────────────────────────────────

def generate_ai_summary(dq_report, matched_columns, stats):
    llm = get_llm()
    r1 = dq_report["report_df1"]
    r2 = dq_report.get("report_df2")
    comp = dq_report.get("comparison", {})

    if r2:
        context = (
            f"Dataset 1: {r1['row_count']} rows, {r1['column_count']} columns\n"
            f"Dataset 2: {r2['row_count']} rows, {r2['column_count']} columns\n"
            f"Duplicates — DS1: {r1['duplicate_rows']}, DS2: {r2['duplicate_rows']}\n"
            f"Missing values — DS1: {sum(r1['missing_values'].values())}, DS2: {sum(r2['missing_values'].values())}\n"
            f"Common columns: {comp.get('common_columns', [])}\n"
            f"Columns only in DS1: {comp.get('columns_in_df1_not_in_df2', [])}\n"
            f"Columns only in DS2: {comp.get('columns_in_df2_not_in_df1', [])}\n"
            f"Schema mismatches: {comp.get('dtype_mismatches', {})}\n"
            f"Fuzzy-matched columns: {matched_columns}\n"
            f"Numeric statistics: {stats}\n"
            f"Anomalies (negative values): {comp.get('anomalies', {})}\n"
        )
        prompt = (
            "You are a senior data analyst. Given this comparison report between two datasets, "
            "produce a clear, structured analysis.\n\n"
            f"{context}\n"
            "Cover:\n"
            "1. Dataset overview & size differences\n"
            "2. Data quality issues (duplicates, nulls, anomalies)\n"
            "3. Schema differences and what they mean\n"
            "4. Key statistical differences in matched columns\n"
            "5. Concrete recommendations for data remediation\n\n"
            "Be specific, not generic. Use numbers from the report."
        )
    else:
        null_cols = {k: v for k, v in r1["missing_values"].items() if v > 0}
        context = (
            f"Dataset: {r1['row_count']} rows, {r1['column_count']} columns\n"
            f"Duplicate rows: {r1['duplicate_rows']}\n"
            f"Missing values per column: {null_cols}\n"
            f"Data types: {r1['data_types']}\n"
            f"Anomalies (negative values): {r1.get('anomalies_negative_values', {})}\n"
            f"Numeric statistics: {stats}\n"
        )
        prompt = (
            "You are a senior data analyst. Given this data quality report, "
            "produce a clear, structured analysis.\n\n"
            f"{context}\n"
            "Cover:\n"
            "1. Dataset overview\n"
            "2. Data quality issues (duplicates, nulls, anomalies)\n"
            "3. Notable patterns in the numeric columns\n"
            "4. Concrete recommendations to improve data quality\n\n"
            "Be specific, not generic. Use numbers from the report."
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ── Overview Markdown ─────────────────────────────────────────────────────────

def build_overview_md(dq_report, matched_columns):
    r1 = dq_report["report_df1"]
    r2 = dq_report.get("report_df2")
    comp = dq_report.get("comparison", {})

    if r2:
        rows = [
            "## Dataset Overview",
            "",
            "| Metric | Dataset 1 | Dataset 2 |",
            "|--------|-----------|-----------|",
            f"| Rows | {r1['row_count']} | {r2['row_count']} |",
            f"| Columns | {r1['column_count']} | {r2['column_count']} |",
            f"| Duplicate Rows | {r1['duplicate_rows']} | {r2['duplicate_rows']} |",
            f"| Missing Values | {sum(r1['missing_values'].values())} | {sum(r2['missing_values'].values())} |",
            f"| Fuzzy-Matched Columns | {len(matched_columns)} | — |",
            "",
        ]
        if comp.get("columns_in_df1_not_in_df2"):
            rows += [f"**Columns only in Dataset 1:** `{'`, `'.join(comp['columns_in_df1_not_in_df2'])}`", ""]
        if comp.get("columns_in_df2_not_in_df1"):
            rows += [f"**Columns only in Dataset 2:** `{'`, `'.join(comp['columns_in_df2_not_in_df1'])}`", ""]
        if comp.get("dtype_mismatches"):
            rows += ["### Schema Mismatches"]
            rows += [f"- **{col}**: `{t[0]}` vs `{t[1]}`" for col, t in comp["dtype_mismatches"].items()]
            rows += [""]
        if matched_columns:
            rows += ["### Fuzzy-Matched Columns"]
            rows += [f"- **{k}** → **{v}**" for k, v in matched_columns.items()]
            rows += [""]
    else:
        null_cols = {k: v for k, v in r1["missing_values"].items() if v > 0}
        rows = [
            "## Dataset Overview",
            "",
            f"| Rows | {r1['row_count']} |",
            "|------|------|",
            f"| Columns | {r1['column_count']} |",
            f"| Duplicate Rows | {r1['duplicate_rows']} |",
            f"| Total Missing Values | {sum(r1['missing_values'].values())} |",
            "",
        ]
        if null_cols:
            rows += ["### Missing Values per Column"]
            rows += [f"- `{col}`: {cnt}" for col, cnt in null_cols.items()]
            rows += [""]
        if r1.get("anomalies_negative_values"):
            rows += ["### Anomalies — Negative Values"]
            rows += [f"- `{col}`: {cnt} negative values" for col, cnt in r1["anomalies_negative_values"].items()]
            rows += [""]

    return "\n".join(rows)


def compute_stats(df1, df2, matched_columns):
    stats = {}
    if df2 is not None:
        for col1, col2 in matched_columns.items():
            if pd.api.types.is_numeric_dtype(df1[col1]) and pd.api.types.is_numeric_dtype(df2[col2]):
                stats[col1] = {
                    "df1_mean": round(float(df1[col1].mean()), 2),
                    "df2_mean": round(float(df2[col2].mean()), 2),
                    "df1_std": round(float(df1[col1].std()), 2),
                    "df2_std": round(float(df2[col2].std()), 2),
                    "df1_min": round(float(df1[col1].min()), 2),
                    "df1_max": round(float(df1[col1].max()), 2),
                    "df2_min": round(float(df2[col2].min()), 2),
                    "df2_max": round(float(df2[col2].max()), 2),
                }
    else:
        for col in df1.select_dtypes(include="number").columns:
            stats[col] = {
                "mean": round(float(df1[col].mean()), 2),
                "std": round(float(df1[col].std()), 2),
                "min": round(float(df1[col].min()), 2),
                "max": round(float(df1[col].max()), 2),
            }
    return stats


# ── Main Analysis ─────────────────────────────────────────────────────────────

def analyze_datasets(file1, file2):
    if file1 is None:
        return "Please upload at least one CSV file.", "", None, None, None, None

    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2) if file2 is not None else None

        dq_report = ai_agent_data_quality(df1, df2)
        matched_columns = fuzzy_match_columns(df1.columns, df2.columns) if df2 is not None else {}
        stats = compute_stats(df1, df2, matched_columns)

        overview = build_overview_md(dq_report, matched_columns)
        summary = generate_ai_summary(dq_report, matched_columns, stats)

        charts = (
            generate_charts_two(df1, df2, matched_columns)
            if df2 is not None
            else generate_charts_single(df1)
        )

        padded = charts + [None] * (4 - len(charts))
        return overview, summary, padded[0], padded[1], padded[2], padded[3]

    except ValueError as e:
        return f"Configuration error: {e}", "", None, None, None, None
    except Exception as e:
        return f"Error: {e}", "", None, None, None, None


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="AI Data Insight Engine") as app:
    gr.Markdown("""
# AI-Powered Data Insight & Summary Engine
Upload one or two CSV datasets to get AI-generated quality reports, comparisons, and visual charts.
""")

    with gr.Row():
        file1 = gr.File(label="Upload Dataset 1 (CSV)", file_types=[".csv"])
        file2 = gr.File(label="Upload Dataset 2 (CSV) — optional", file_types=[".csv"])

    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

    with gr.Tabs():
        with gr.Tab("Overview"):
            overview_output = gr.Markdown()
        with gr.Tab("AI Insights"):
            summary_output = gr.Textbox(label="AI-Generated Insights", lines=18, show_copy_button=True)
        with gr.Tab("Charts"):
            with gr.Row():
                chart1_output = gr.Plot()
                chart2_output = gr.Plot()
            with gr.Row():
                chart3_output = gr.Plot()
                chart4_output = gr.Plot()

    analyze_btn.click(
        fn=analyze_datasets,
        inputs=[file1, file2],
        outputs=[
            overview_output, summary_output,
            chart1_output, chart2_output,
            chart3_output, chart4_output,
        ],
    )

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    app.launch(share=share)
