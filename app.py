```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from fuzzywuzzy import fuzz
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import io
import base64

# Initialize Groq LLM
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.environ.get("GROQ_API_KEY", "your-groq-key-here")
    )

def fuzzy_match_columns(cols1, cols2):
    """Match columns between two datasets using fuzzy matching"""
    matches = {}
    for col1 in cols1:
        best_match = None
        best_score = 0
        for col2 in cols2:
            score = fuzz.ratio(col1.lower(), col2.lower())
            if score > best_score:
                best_score = score
                best_match = col2
        if best_score > 60:
            matches[col1] = best_match
    return matches

def generate_charts(df1, df2, matched_columns):
    """Generate comparison charts for matched columns"""
    charts = []
    numeric_matches = {k: v for k, v in matched_columns.items()
                      if pd.api.types.is_numeric_dtype(df1[k])}

    for col1, col2 in list(numeric_matches.items())[:3]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['Dataset 1', 'Dataset 2'],
               [df1[col1].mean(), df2[col2].mean()],
               color=['#2196F3', '#4CAF50'])
        ax.set_title(f'Average {col1} Comparison')
        ax.set_ylabel('Mean Value')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        charts.append(fig)
        plt.close()

    return charts

def generate_ai_summary(df1, df2, matched_columns, stats):
    """Use LLM to generate natural language summary"""
    llm = get_llm()

    prompt = f"""You are a data analyst AI. Analyze these two datasets and provide clear insights.

Dataset 1 shape: {df1.shape}
Dataset 2 shape: {df2.shape}
Matched columns: {matched_columns}
Key statistics: {stats}

Provide:
1. A brief overview of both datasets
2. Key differences found
3. Notable trends or anomalies
4. Business insights in simple language

Keep it clear and concise."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def analyze_datasets(file1, file2):
    """Main analysis function"""
    if file1 is None or file2 is None:
        return "⚠️ Please upload both CSV files!", None, None, None

    try:
        df1 = pd.read_csv(file1.name)
        df2 = pd.read_csv(file2.name)

        # Fuzzy column matching
        matched_columns = fuzzy_match_columns(df1.columns, df2.columns)

        # Generate statistics
        stats = {}
        for col1, col2 in matched_columns.items():
            if pd.api.types.is_numeric_dtype(df1[col1]):
                stats[col1] = {
                    'df1_mean': round(df1[col1].mean(), 2),
                    'df2_mean': round(df2[col2].mean(), 2),
                    'df1_std': round(df1[col1].std(), 2),
                    'df2_std': round(df2[col2].std(), 2),
                }

        # Generate AI summary
        summary = generate_ai_summary(df1, df2, matched_columns, stats)

        # Generate charts
        charts = generate_charts(df1, df2, matched_columns)

        # Dataset overview
        overview = f"""
## 📊 Dataset Overview

| Metric | Dataset 1 | Dataset 2 |
|--------|-----------|-----------|
| Rows | {df1.shape[0]} | {df2.shape[0]} |
| Columns | {df1.shape[1]} | {df2.shape[1]} |
| Matched Columns | {len(matched_columns)} | {len(matched_columns)} |

## 🔗 Matched Columns
{chr(10).join([f'- **{k}** → **{v}**' for k, v in matched_columns.items()])}
        """

        return overview, summary, charts[0] if charts else None, charts[1] if len(charts) > 1 else None

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="AI Data Insight Engine") as app:

    gr.Markdown("""
    # 🔍 AI-Powered Data Insight & Summary Engine
    **Upload two CSV datasets → Get instant AI-generated insights, comparisons & visual charts**
    """)

    with gr.Row():
        file1 = gr.File(label="📂 Upload Dataset 1 (CSV)", file_types=[".csv"])
        file2 = gr.File(label="📂 Upload Dataset 2 (CSV)", file_types=[".csv"])

    analyze_btn = gr.Button("🔍 Analyze Datasets", variant="primary", size="lg")

    with gr.Row():
        overview_output = gr.Markdown(label="Dataset Overview")

    with gr.Row():
        summary_output = gr.Textbox(label="🤖 AI-Generated Insights", lines=10)

    with gr.Row():
        chart1_output = gr.Plot(label="📊 Chart 1")
        chart2_output = gr.Plot(label="📊 Chart 2")

    analyze_btn.click(
        fn=analyze_datasets,
        inputs=[file1, file2],
        outputs=[overview_output, summary_output, chart1_output, chart2_output]
    )

app.launch(share=True)
