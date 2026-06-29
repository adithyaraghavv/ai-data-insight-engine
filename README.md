# AI-Powered Data Insight & Summary Engine

An intelligent data analysis tool that examines one or two CSV datasets and automatically generates data quality reports, visual charts, and natural language summaries — replacing manual analysis workflows.

![Demo 1](demoimg1.png)
![Demo 2](demoimg2.png)
![Demo 3](demoimg3.png)

---

## How It Works

1. Upload one or two CSV datasets
2. AI performs fuzzy column alignment & schema matching
3. Data quality checks run (nulls, duplicates, anomalies, schema mismatches)
4. Statistical analysis is computed across matched columns
5. LLaMA 3.3 70B generates a human-readable summary via Groq
6. Visual charts are generated (box plots, distributions, null heatmaps)

---

## Key Features

- **Fuzzy Column Alignment** — matches columns even with mismatched names
- **Data Quality Report** — detects nulls, duplicates, negative anomalies, schema mismatches
- **Visual Charts** — box plots, histograms, null comparisons, data type breakdowns
- **Natural Language Summaries** — complex stats explained in plain English by LLaMA 3.3 70B
- **Single or Two Dataset Mode** — works with just one CSV or a pair for comparison
- **Automated Pipeline** — zero manual effort needed

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Gradio | Web UI |
| LangChain + Groq | LLM pipeline (LLaMA 3.3 70B) |
| Pandas | Data processing |
| Matplotlib | Visual charts |
| FuzzyWuzzy | Column name matching |
| python-dotenv | API key management |

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/adithyaraghavv/ai-data-insight-engine
cd ai-data-insight-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key (free at console.groq.com)
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 4. Run
python app.py
```

The app opens at **http://localhost:7860**

---

## Use Cases

- Compare sales data across two time periods
- Analyze differences between two customer datasets
- Audit data quality between source and target systems
- Generate automated reports from raw CSV data
- Detect anomalies and schema issues before loading data to a database
