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
    summarize_nulls,
    find_duplicate_rows,
)

load_dotenv()


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=api_key)


def generate_charts(df1, df2, matched_columns):
    charts = []
    numeric_matches = {
        k: v
        for k, v in matched_columns.items()
        if pd.api.types.is_numeric_dtype(df1[k]) and pd.api.types.is_numeric_dtype(df2[v])
    }
    for col1, col2 in list(numeric_matches.items())[:3]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            ["Dataset 1", "Dataset 2"],
            [df1[col1].mean(), df2[col2].mean()],
            color=["#2196F3", "#4CAF50"],
        )
        ax.set_title(f"Average {col1} Comparison")
        ax.set_ylabel("Mean Value")
        plt.tight_layout()
        charts.append(fig)
        plt.close(fig)
    return charts


def generate_ai_summary(dq_report, matched_columns, stats):
    llm = get_llm()
    r1 = dq_report["report_df1"]
    r2 = dq_report.get("report_df2")
    comp = dq_report.get("comparison", {})

    two_dataset = r2 is not None

    if two_dataset:
        context = f"""
Dataset 1: {r1['row_count']} rows, {r1['column_count']} columns
Dataset 2: {r2['row_count']} rows, {r2['column_count']} columns
Duplicate rows — Dataset 1: {r1['duplicate_rows']}, Dataset 2: {r2['duplicate_rows']}
Missing values — Dataset 1: {sum(r1['missing_values'].values())}, Dataset 2: {sum(r2['missing_values'].values())}
Common columns: {comp.get('common_columns', [])}
Columns only in Dataset 1: {comp.get('columns_in_df1_not_in_df2', [])}
Columns only in Dataset 2: {comp.get('columns_in_df2_not_in_df1', [])}
Schema mismatches: {comp.get('dtype_mismatches', {})}
Fuzzy-matched columns: {matched_columns}
Key numeric statistics: {stats}
Anomalies (negative values): {comp.get('anomalies', {})}
"""
        prompt = (
            "You are a data analyst AI. Analyze these two datasets and provide clear insights.\n\n"
            + context
            + "\nProvide:\n"
            "1. A brief overview of both datasets\n"
            "2. Data quality issues found (duplicates, nulls, anomalies)\n"
            "3. Key differences between datasets\n"
            "4. Notable trends or anomalies\n"
            "5. Actionable recommendations\n\n"
            "Be concise and clear."
        )
    else:
        context = f"""
Dataset: {r1['row_count']} rows, {r1['column_count']} columns
Duplicate rows: {r1['duplicate_rows']}
Missing values per column: {r1['missing_values']}
Data types: {r1['data_types']}
Anomalies (negative values): {r1.get('anomalies_negative_values', {})}
Key numeric statistics: {stats}
"""
        prompt = (
            "You are a data analyst AI. Analyze this dataset and provide clear insights.\n\n"
            + context
            + "\nProvide:\n"
            "1. A brief overview of the dataset\n"
            "2. Data quality issues (duplicates, nulls, anomalies)\n"
            "3. Notable patterns or concerns\n"
            "4. Actionable recommendations\n\n"
            "Be concise and clear."
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def build_overview_md(df1, df2, dq_report, matched_columns):
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
            f"| Matched Columns | {len(matched_columns)} | {len(matched_columns)} |",
            "",
        ]
        if comp.get("dtype_mismatches"):
            rows += [
                "### Schema Mismatches",
                *[f"- **{col}**: `{t[0]}` vs `{t[1]}`" for col, t in comp["dtype_mismatches"].items()],
                "",
            ]
        if matched_columns:
            rows += [
                "### Fuzzy-Matched Columns",
                *[f"- **{k}** → **{v}**" for k, v in matched_columns.items()],
                "",
            ]
    else:
        null_cols = {k: v for k, v in r1["missing_values"].items() if v > 0}
        rows = [
            "## Dataset Overview",
            "",
            f"- **Rows**: {r1['row_count']}",
            f"- **Columns**: {r1['column_count']}",
            f"- **Duplicate Rows**: {r1['duplicate_rows']}",
            f"- **Total Missing Values**: {sum(r1['missing_values'].values())}",
            "",
        ]
        if null_cols:
            rows += [
                "### Missing Values per Column",
                *[f"- `{col}`: {cnt}" for col, cnt in null_cols.items()],
                "",
            ]
        if r1.get("anomalies_negative_values"):
            rows += [
                "### Anomalies (Negative Values)",
                *[f"- `{col}`: {cnt}" for col, cnt in r1["anomalies_negative_values"].items()],
                "",
            ]

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


def analyze_datasets(file1, file2):
    if file1 is None:
        return "Please upload at least one CSV file.", None, None, None

    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2) if file2 is not None else None

        dq_report = ai_agent_data_quality(df1, df2)

        matched_columns = fuzzy_match_columns(df1.columns, df2.columns) if df2 is not None else {}
        stats = compute_stats(df1, df2, matched_columns)

        overview = build_overview_md(df1, df2, dq_report, matched_columns)
        summary = generate_ai_summary(dq_report, matched_columns, stats)

        charts = generate_charts(df1, df2, matched_columns) if df2 is not None else []

        chart1 = charts[0] if len(charts) > 0 else None
        chart2 = charts[1] if len(charts) > 1 else None

        return overview, summary, chart1, chart2

    except ValueError as e:
        return f"Configuration error: {e}", None, None, None
    except Exception as e:
        return f"Error: {e}", None, None, None


with gr.Blocks(theme=gr.themes.Soft(), title="AI Data Insight Engine") as app:
    gr.Markdown("""
# AI-Powered Data Insight & Summary Engine
Upload one or two CSV datasets to get AI-generated quality reports, comparisons, and visual charts.
""")

    with gr.Row():
        file1 = gr.File(label="Upload Dataset 1 (CSV)", file_types=[".csv"])
        file2 = gr.File(label="Upload Dataset 2 (CSV) — optional", file_types=[".csv"])

    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

    overview_output = gr.Markdown(label="Dataset Overview")
    summary_output = gr.Textbox(label="AI-Generated Insights", lines=12)

    with gr.Row():
        chart1_output = gr.Plot(label="Chart 1")
        chart2_output = gr.Plot(label="Chart 2")

    analyze_btn.click(
        fn=analyze_datasets,
        inputs=[file1, file2],
        outputs=[overview_output, summary_output, chart1_output, chart2_output],
    )

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    app.launch(share=share)
