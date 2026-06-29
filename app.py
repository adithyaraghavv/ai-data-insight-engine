import os
import io
import tempfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.enums import TA_CENTER

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


def parse_key_columns(raw: str):
    if not raw or not raw.strip():
        return None
    return [c.strip() for c in raw.split(",") if c.strip()]


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_null_comparison(df1, df2):
    cols = list(set(df1.columns) & set(df2.columns))
    nulls1 = [int(df1[c].isnull().sum()) for c in cols]
    nulls2 = [int(df2[c].isnull().sum()) for c in cols]
    if all(v == 0 for v in nulls1 + nulls2):
        return None
    x = range(len(cols))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.8), 4))
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
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(
        [df1[col1].dropna().tolist(), df2[col2].dropna().tolist()],
        patch_artist=True,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Dataset 1", "Dataset 2"])
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][1].set_facecolor("#4CAF50")
    ax.set_title(f"Distribution: {col1}")
    ax.set_ylabel("Value")
    plt.tight_layout()
    return fig


def chart_single_nulls(df):
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
    type_counts = df.dtypes.apply(lambda x: x.kind).value_counts()
    label_map = {
        "f": "Float", "i": "Integer", "O": "Object/String",
        "b": "Boolean", "M": "Datetime", "U": "Unicode",
    }
    labels = [label_map.get(k, k) for k in type_counts.index]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        type_counts.values, labels=labels, autopct="%1.0f%%",
        colors=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"],
    )
    ax.set_title("Column Data Types")
    plt.tight_layout()
    return fig


def chart_histogram(df, col):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col].dropna(), bins=20, color="#7E57C2", edgecolor="white")
    ax.set_title(f"Distribution: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def chart_correlation_heatmap(df, title="Correlation Heatmap"):
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr()
    size = max(5, corr.shape[1] * 0.8)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr.iloc[i, j]
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(val) > 0.5 else "black",
            )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def chart_missing_heatmap(df, title="Missing Values Map (red = missing)", max_rows=100):
    null_matrix = df.isnull()
    if not null_matrix.any().any():
        return None
    sample = null_matrix.iloc[:max_rows]
    fig, ax = plt.subplots(figsize=(max(6, sample.shape[1] * 0.5), max(3, min(6, sample.shape[0] * 0.06))))
    ax.imshow(sample.values, aspect="auto", cmap="Reds", interpolation="none")
    ax.set_xticks(range(len(sample.columns)))
    ax.set_xticklabels(sample.columns, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel(f"Columns  (up to {max_rows} rows shown)")
    ax.set_title(title)
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
    for col1, col2 in numeric_matches[:2]:
        charts.append(chart_boxplot(df1, df2, col1, col2))
    corr1 = chart_correlation_heatmap(df1, "Correlation Heatmap — Dataset 1")
    if corr1:
        charts.append(corr1)
    miss1 = chart_missing_heatmap(df1, "Missing Values — Dataset 1")
    if miss1:
        charts.append(miss1)
    miss2 = chart_missing_heatmap(df2, "Missing Values — Dataset 2")
    if miss2:
        charts.append(miss2)
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
    corr = chart_correlation_heatmap(df)
    if corr:
        charts.append(corr)
    miss = chart_missing_heatmap(df)
    if miss:
        charts.append(miss)
    return charts


# ── PDF Export ────────────────────────────────────────────────────────────────

def _fig_to_bytes(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf


def _md_to_plain(text: str) -> str:
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-|]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_pdf(overview_md: str, summary_text: str, charts: list) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], alignment=TA_CENTER, fontSize=18)
    h1_style = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13, textColor=colors.HexColor("#1565C0"))
    body_style = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=13)

    doc = SimpleDocTemplate(tmp.name, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    story = []

    story.append(Paragraph("AI Data Insight Report", title_style))
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("Dataset Overview", h1_style))
    story.append(Spacer(1, 2*mm))
    for line in _md_to_plain(overview_md).splitlines():
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 6*mm))

    story.append(Paragraph("AI-Generated Insights", h1_style))
    story.append(Spacer(1, 2*mm))
    for line in summary_text.splitlines():
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 6*mm))

    visible_charts = [f for f in charts if f is not None]
    if visible_charts:
        story.append(PageBreak())
        story.append(Paragraph("Visual Analysis", h1_style))
        story.append(Spacer(1, 4*mm))
        usable_w = A4[0] - 40*mm
        img_w = (usable_w - 5*mm) / 2
        row_buf = []
        for fig in visible_charts:
            buf = _fig_to_bytes(fig)
            img = RLImage(buf, width=img_w, height=img_w * 0.65)
            row_buf.append(img)
            if len(row_buf) == 2:
                from reportlab.platypus import Table
                t = Table([row_buf], colWidths=[img_w, img_w])
                story.append(t)
                story.append(Spacer(1, 4*mm))
                row_buf = []
        if row_buf:
            from reportlab.platypus import Table
            t = Table([row_buf + [Spacer(img_w, img_w * 0.65)]], colWidths=[img_w, img_w])
            story.append(t)

    doc.build(story)
    return tmp.name


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
            "## Dataset Overview", "",
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
            "## Dataset Overview", "",
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

ERROR_RETURN = ("", "", None, None, None, None, None, None, None)


def analyze_datasets(file1, file2, key_columns_raw=""):
    if file1 is None:
        return ("Please upload at least one CSV file.",) + ("",) + (None,) * 7

    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2) if file2 is not None else None
        key_columns = parse_key_columns(key_columns_raw)

        dq_report = ai_agent_data_quality(df1, df2, key_columns=key_columns)
        matched_columns = fuzzy_match_columns(df1.columns, df2.columns) if df2 is not None else {}
        stats = compute_stats(df1, df2, matched_columns)

        overview = build_overview_md(dq_report, matched_columns)
        summary = generate_ai_summary(dq_report, matched_columns, stats)

        charts = (
            generate_charts_two(df1, df2, matched_columns)
            if df2 is not None
            else generate_charts_single(df1)
        )

        pdf_path = generate_pdf(overview, summary, charts)

        padded = charts + [None] * (6 - len(charts))
        return overview, summary, pdf_path, padded[0], padded[1], padded[2], padded[3], padded[4], padded[5]

    except ValueError as e:
        return (f"Configuration error: {e}",) + ("",) + (None,) * 7
    except Exception as e:
        return (f"Error: {e}",) + ("",) + (None,) * 7


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

@keyframes fadeUp   { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
@keyframes cardPop  { from { opacity:0; transform:scale(0.95); }      to { opacity:1; transform:scale(1); } }
@keyframes glow     { 0%,100%{box-shadow:0 0 20px rgba(34,197,94,.15);} 50%{box-shadow:0 0 36px rgba(34,197,94,.32);} }
@keyframes spinRing { to { transform: rotate(360deg); } }

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box !important; }
body, .gradio-container {
    background: #0e0e0e !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #fff !important;
    -webkit-font-smoothing: antialiased !important;
}
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; padding: 32px 24px !important; }

/* ── Hero ── */
#hero {
    position: relative;
    padding: 60px 40px 52px;
    margin-bottom: 20px;
    text-align: center;
    animation: fadeUp 0.55s cubic-bezier(.22,1,.36,1) both;
    overflow: hidden;
    border-radius: 28px;
    background: #161616;
    border: 1px solid #242424;
}
#hero::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 50% -10%, rgba(34,197,94,.13) 0%, transparent 70%),
        radial-gradient(ellipse 30% 30% at 80% 80%, rgba(59,130,246,.07) 0%, transparent 60%);
    pointer-events: none;
}
#hero .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34,197,94,.1);
    border: 1px solid rgba(34,197,94,.25);
    color: #22c55e;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 20px;
}
#hero h1 {
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    color: #fff !important;
    letter-spacing: -0.045em !important;
    line-height: 1.05 !important;
    margin: 0 0 16px !important;
}
#hero h1 em { color: #22c55e; font-style: normal; }
#hero p {
    color: #4b5563 !important;
    font-size: 1rem !important;
    max-width: 480px !important;
    margin: 0 auto !important;
    line-height: 1.6 !important;
}

/* ── Feature cards ── */
#feature-cards {
    display: grid !important;
    grid-template-columns: repeat(4,1fr) !important;
    gap: 10px !important;
    margin-bottom: 20px !important;
    animation: fadeUp 0.55s cubic-bezier(.22,1,.36,1) 0.08s both;
}
.feat-card {
    border-radius: 18px !important;
    padding: 22px 18px 18px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    line-height: 1.35 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    cursor: default !important;
    transition: transform 0.22s cubic-bezier(.22,1,.36,1), box-shadow 0.22s !important;
    animation: cardPop 0.45s cubic-bezier(.22,1,.36,1) both !important;
    min-height: 110px !important;
}
.feat-card:hover { transform: translateY(-5px) !important; }
.feat-card .icon { font-size: 1.5rem !important; }
.feat-card-green  { background:#22c55e !important; color:#052e16 !important; animation-delay:.04s !important; box-shadow:0 8px 32px rgba(34,197,94,.2) !important; }
.feat-card-green:hover  { box-shadow:0 16px 40px rgba(34,197,94,.35) !important; }
.feat-card-blue   { background:#3b82f6 !important; color:#dbeafe !important; animation-delay:.08s !important; box-shadow:0 8px 32px rgba(59,130,246,.2) !important; }
.feat-card-blue:hover   { box-shadow:0 16px 40px rgba(59,130,246,.35) !important; }
.feat-card-coral  { background:#f97316 !important; color:#431407 !important; animation-delay:.12s !important; box-shadow:0 8px 32px rgba(249,115,22,.2) !important; }
.feat-card-coral:hover  { box-shadow:0 16px 40px rgba(249,115,22,.35) !important; }
.feat-card-violet { background:#a855f7 !important; color:#f5f3ff !important; animation-delay:.16s !important; box-shadow:0 8px 32px rgba(168,85,247,.2) !important; }
.feat-card-violet:hover { box-shadow:0 16px 40px rgba(168,85,247,.35) !important; }

/* ── Upload section label ── */
.upload-label {
    font-size: 0.68rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #374151 !important;
    margin-bottom: 8px !important;
}

/* ── Upload cards ── */
#upload-row { animation: fadeUp 0.55s cubic-bezier(.22,1,.36,1) 0.14s both; gap: 12px !important; }
#upload-row .block {
    background: #161616 !important;
    border: 1.5px dashed #262626 !important;
    border-radius: 20px !important;
    min-height: 160px !important;
    transition: all 0.22s cubic-bezier(.22,1,.36,1) !important;
}
#upload-row .block:hover {
    border-color: #22c55e !important;
    background: #161f17 !important;
    box-shadow: 0 0 0 4px rgba(34,197,94,.08), 0 12px 40px rgba(0,0,0,.35) !important;
    transform: translateY(-3px) !important;
}
#upload-row label span {
    color: #374151 !important;
    font-size: 0.68rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
#upload-row .upload-text, #upload-row p { color: #374151 !important; font-size: 0.88rem !important; }
#upload-row svg { stroke: #2d2d2d !important; width: 28px !important; height: 28px !important; }

/* ── Key columns input ── */
#key-cols { animation: fadeUp 0.55s cubic-bezier(.22,1,.36,1) 0.18s both; }
#key-cols .block {
    background: #161616 !important;
    border: 1.5px solid #242424 !important;
    border-radius: 20px !important;
    padding: 4px 4px !important;
    transition: all 0.22s !important;
}
#key-cols .block:focus-within {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 4px rgba(34,197,94,.08) !important;
}
#key-cols input, #key-cols textarea {
    background: transparent !important;
    border: none !important;
    color: #f3f4f6 !important;
    font-size: 0.97rem !important;
    font-weight: 500 !important;
    padding: 14px 16px !important;
}
#key-cols input::placeholder, #key-cols textarea::placeholder { color: #2d2d2d !important; }
#key-cols label span {
    color: #374151 !important;
    font-size: 0.68rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
#key-cols .info { color: #2d2d2d !important; font-size: 0.75rem !important; margin-top: 4px !important; }

/* ── Analyze button ── */
#analyze-btn { animation: fadeUp 0.55s cubic-bezier(.22,1,.36,1) 0.22s both; }
#analyze-btn button {
    background: #22c55e !important;
    border: none !important;
    border-radius: 18px !important;
    font-size: 0.98rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em !important;
    color: #052e16 !important;
    height: 58px !important;
    transition: all 0.22s cubic-bezier(.22,1,.36,1) !important;
    box-shadow: 0 4px 24px rgba(34,197,94,.28) !important;
    animation: glow 3s ease-in-out infinite !important;
}
#analyze-btn button:hover {
    background: #16a34a !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(34,197,94,.45) !important;
}
#analyze-btn button:active { transform: translateY(-1px) !important; }

/* ── Tabs ── */
.tabs { animation: fadeUp 0.55s cubic-bezier(.22,1,.36,1) 0.26s both; }
.tab-nav {
    background: #161616 !important;
    border-radius: 16px !important;
    padding: 5px !important;
    border: 1px solid #242424 !important;
    margin-bottom: 14px !important;
    gap: 3px !important;
}
.tab-nav button {
    color: #4b5563 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border: none !important;
    background: transparent !important;
    padding: 10px 22px !important;
    transition: all 0.18s !important;
    letter-spacing: 0.01em !important;
}
.tab-nav button:hover:not(.selected) {
    color: #9ca3af !important;
    background: #1e1e1e !important;
}
.tab-nav button.selected {
    background: #22c55e !important;
    color: #052e16 !important;
    font-weight: 800 !important;
    box-shadow: 0 2px 16px rgba(34,197,94,.3) !important;
}

/* ── Tab panel ── */
.tabitem {
    background: #161616 !important;
    border: 1px solid #242424 !important;
    border-radius: 20px !important;
    padding: 30px !important;
}

/* ── Markdown ── */
.prose h2, .md h2 {
    color: #22c55e !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em !important;
    border-bottom: 1px solid #1e1e1e !important;
    padding-bottom: 10px !important;
    margin: 0 0 16px !important;
}
.prose h3, .md h3 { color: #e5e7eb !important; font-size: 0.88rem !important; font-weight: 700 !important; margin: 16px 0 8px !important; }
.prose, .md { color: #6b7280 !important; line-height: 1.8 !important; }
.prose table { width:100% !important; border-collapse:collapse !important; }
.prose th {
    background: #111 !important;
    color: #22c55e !important;
    font-weight: 800 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 14px 16px !important;
    border-bottom: 1px solid #1e1e1e !important;
    text-align: left !important;
}
.prose td { padding: 12px 16px !important; border-bottom: 1px solid #1a1a1a !important; color: #d1d5db !important; font-size: 0.9rem !important; }
.prose tr:last-child td { border-bottom: none !important; }
.prose tr:hover td { background: rgba(34,197,94,.04) !important; }
.prose code { background: #1a1a1a !important; color: #22c55e !important; border-radius: 6px !important; padding: 2px 8px !important; font-size: 0.83em !important; }
.prose li { color: #6b7280 !important; margin-bottom: 5px !important; padding-left: 4px !important; }
.prose strong { color: #f9fafb !important; font-weight: 700 !important; }

/* ── AI insights ── */
#ai-insights textarea {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 14px !important;
    color: #d1d5db !important;
    font-size: 0.92rem !important;
    line-height: 1.85 !important;
    padding: 20px !important;
    resize: vertical !important;
}
#ai-insights label span { color: #22c55e !important; font-weight: 800 !important; font-size: 0.72rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }

/* ── Charts ── */
.plot-container {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    transition: transform 0.22s cubic-bezier(.22,1,.36,1), box-shadow 0.22s !important;
}
.plot-container:hover { transform: translateY(-4px) !important; box-shadow: 0 12px 40px rgba(0,0,0,.5) !important; }

/* ── Export ── */
#export-section h3 { color: #22c55e !important; font-size: 1rem !important; font-weight: 800 !important; margin-bottom: 16px !important; }
#export-file .file-preview {
    background: #111 !important;
    border: 1px solid #1e1e1e !important;
    border-radius: 14px !important;
    color: #22c55e !important;
    transition: border-color 0.2s !important;
}
#export-file .file-preview:hover { border-color: #22c55e !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0e0e0e; }
::-webkit-scrollbar-thumb { background: #242424; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #22c55e; }
"""

with gr.Blocks(title="AI Data Insight Engine", css=CSS) as app:

    gr.HTML("""
    <div id="hero">
        <div class="badge">⚡ Powered by LLaMA 3.3 · 70B</div>
        <h1>What are you<br><em>analysing today?</em></h1>
        <p>Drop your CSV datasets below — get data quality reports, AI-generated insights, visual charts and a PDF export in seconds.</p>
    </div>
    <div id="feature-cards">
        <div class="feat-card feat-card-green">
            <span class="icon">🔍</span>
            <span>Data Quality<br>Report</span>
        </div>
        <div class="feat-card feat-card-blue">
            <span class="icon">🤖</span>
            <span>AI-Generated<br>Insights</span>
        </div>
        <div class="feat-card feat-card-coral">
            <span class="icon">📊</span>
            <span>Visual<br>Charts</span>
        </div>
        <div class="feat-card feat-card-violet">
            <span class="icon">📥</span>
            <span>PDF<br>Export</span>
        </div>
    </div>
    """)

    with gr.Row(elem_id="upload-row"):
        file1 = gr.File(label="Dataset 1 (CSV)", file_types=[".csv"])
        file2 = gr.File(label="Dataset 2 (CSV) — optional", file_types=[".csv"])

    with gr.Row(elem_id="key-cols"):
        key_cols_input = gr.Textbox(
            label="Key Columns (optional, comma-separated)",
            placeholder="e.g.  EmployeeID, Name",
            info="Enables deeper duplicate and uniqueness checks on these columns.",
        )

    with gr.Row(elem_id="analyze-btn"):
        analyze_btn = gr.Button("⚡  Analyze", variant="primary", size="lg")

    with gr.Tabs():
        with gr.Tab("📊  Overview"):
            overview_output = gr.Markdown()
        with gr.Tab("🤖  AI Insights"):
            summary_output = gr.Textbox(
                label="AI-Generated Insights",
                lines=20,
                elem_id="ai-insights",
            )
        with gr.Tab("📈  Charts"):
            with gr.Row():
                chart1_output = gr.Plot()
                chart2_output = gr.Plot()
            with gr.Row():
                chart3_output = gr.Plot()
                chart4_output = gr.Plot()
            with gr.Row():
                chart5_output = gr.Plot()
                chart6_output = gr.Plot()
        with gr.Tab("📥  Export"):
            gr.HTML('<div id="export-section"><h3>Download your full report as a PDF</h3></div>')
            pdf_download = gr.File(label="PDF Report", elem_id="export-file")

    analyze_btn.click(
        fn=analyze_datasets,
        inputs=[file1, file2, key_cols_input],
        outputs=[
            overview_output, summary_output, pdf_download,
            chart1_output, chart2_output,
            chart3_output, chart4_output,
            chart5_output, chart6_output,
        ],
    )

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    app.launch(share=share)
