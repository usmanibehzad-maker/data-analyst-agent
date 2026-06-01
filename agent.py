"""
agent.py — All 8 tools + LangGraph ReAct agent assembly.
This is the core of the Junior Data Analyst Agent.
"""

import os
import io
import sys
import re
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — no display needed
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from sqlalchemy import create_engine, text, inspect as sql_inspect

from safety import is_safe_query, run_sandboxed, truncate_output, summarize_dataframe

load_dotenv()

# ── Shared State ────────────────────────────────────────────────────────────────
# Global state for the duration of one analysis run.
# Intentionally simple — one DataFrame, one connection, one output dir.

_state = {
    "current_dataframe": None,       # pd.DataFrame | None
    "database_engine": None,         # SQLAlchemy engine | None
    "output_directory": "output",    # base path for charts/reports/logs
    "run_log": [],                   # audit trail entries
}


def _log(action: str, detail: str):
    """Append to the run audit log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "detail": detail[:500],  # truncate long details in log
    }
    _state["run_log"].append(entry)


def init_run(output_directory: str = "output"):
    """Call this before each analysis run to reset state."""
    _state["current_dataframe"] = None
    _state["database_engine"] = None
    _state["output_directory"] = output_directory
    _state["run_log"] = []
    os.makedirs(f"{output_directory}/reports", exist_ok=True)
    os.makedirs(f"{output_directory}/charts", exist_ok=True)
    os.makedirs(f"{output_directory}/logs", exist_ok=True)


# ── Tool 1: connect_database ────────────────────────────────────────────────────

@tool
def connect_database(connection_string: str) -> str:
    """
    Connect to a SQL database using a SQLAlchemy connection string.
    Supported: sqlite:///path, postgresql://user:pass@host/db, mysql+pymysql://user:pass@host/db
    Call this first when analyzing a database source.
    """
    try:
        engine = create_engine(connection_string, connect_args={"connect_timeout": 10} if "postgresql" in connection_string or "mysql" in connection_string else {})
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        _state["database_engine"] = engine
        db_name = connection_string.split("/")[-1].split("?")[0]
        db_type = connection_string.split(":")[0].split("+")[0]

        _log("connect_database", f"Connected to {db_name} ({db_type})")
        return f"Connected to {db_name} ({db_type}). Ready to query."

    except Exception as e:
        error_msg = str(e)
        # Never echo credentials back
        if "@" in connection_string:
            safe_str = connection_string.split("@")[-1]
        else:
            safe_str = connection_string.split("/")[-1]
        _log("connect_database_error", error_msg[:200])
        return f"Connection failed to {safe_str}. Error: {error_msg[:200]}"


# ── Tool 2: list_tables ─────────────────────────────────────────────────────────

@tool
def list_tables() -> str:
    """
    List all tables in the connected database with approximate row counts.
    Call this after connect_database to see what data is available.
    """
    engine = _state["database_engine"]
    if engine is None:
        return "No database connected. Call connect_database first."

    try:
        inspector = sql_inspect(engine)
        table_names = inspector.get_table_names()

        if not table_names:
            return "No tables found in the database."

        lines = ["Tables in database:"]
        with engine.connect() as conn:
            for table in table_names:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM `{table}`" if "mysql" in str(engine.url) else f'SELECT COUNT(*) FROM "{table}"'))
                    count = result.scalar()
                    lines.append(f"  - {table}: ~{count:,} rows")
                except Exception:
                    lines.append(f"  - {table}: (row count unavailable)")

        output = "\n".join(lines)
        _log("list_tables", output)
        return output

    except Exception as e:
        return f"Error listing tables: {str(e)[:200]}"


# ── Tool 3: inspect_table ───────────────────────────────────────────────────────

@tool
def inspect_table(table_name: str) -> str:
    """
    Inspect a table's schema: columns, data types, null counts, and sample rows.
    ALWAYS call this before running a SQL query on a new table.
    """
    engine = _state["database_engine"]
    if engine is None:
        return "No database connected. Call connect_database first."

    try:
        inspector = sql_inspect(engine)
        available = inspector.get_table_names()

        if table_name not in available:
            return f"Table '{table_name}' not found. Available tables: {', '.join(available)}"

        columns = inspector.get_columns(table_name)
        pk = inspector.get_pk_constraint(table_name)
        pk_cols = pk.get("constrained_columns", [])

        lines = [f"Table: {table_name}", "Columns:"]
        col_names = []
        for col in columns:
            pk_flag = "  PRIMARY KEY" if col["name"] in pk_cols else ""
            nullable = "" if col.get("nullable", True) else "  NOT NULL"
            lines.append(f"  - {col['name']:<25} {str(col['type']):<20}{nullable}{pk_flag}")
            col_names.append(col["name"])

        # Null counts + sample
        with engine.connect() as conn:
            quote = "`" if "mysql" in str(engine.url) else '"'
            sample_q = f"SELECT * FROM {quote}{table_name}{quote} LIMIT 5"
            sample_df = pd.read_sql(sample_q, conn)

            null_counts = sample_df.isnull().sum()
            lines.append("\nNull counts (sample of 5 rows):")
            for col, count in null_counts.items():
                lines.append(f"  - {col}: {count}")

            lines.append("\nSample (first 5 rows):")
            lines.append(sample_df.to_string(index=False))

        output = "\n".join(lines)
        _log("inspect_table", f"Inspected {table_name}")
        return truncate_output(output)

    except Exception as e:
        return f"Error inspecting table '{table_name}': {str(e)[:200]}"


# ── Tool 4: run_sql ─────────────────────────────────────────────────────────────

@tool
def run_sql(query: str) -> str:
    """
    Execute a SELECT query against the connected database.
    Results are stored as the current DataFrame for further analysis.
    Only SELECT and WITH queries are allowed — this agent is read-only.
    """
    engine = _state["database_engine"]
    if engine is None:
        return "No database connected. Call connect_database first."

    # Layer 1 safety check
    safe, reason = is_safe_query(query)
    if not safe:
        _log("run_sql_refused", reason)
        return reason

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        _state["current_dataframe"] = df
        _log("run_sql", f"Query returned {df.shape[0]} rows × {df.shape[1]} cols")

        summary = summarize_dataframe(df)
        return f"Query returned {df.shape[0]:,} rows × {df.shape[1]} columns.\n\n{summary}"

    except Exception as e:
        _log("run_sql_error", str(e)[:200])
        return f"SQL error: {str(e)[:300]}"


# ── Tool 5: load_file ───────────────────────────────────────────────────────────

@tool
def load_file(file_path: str) -> str:
    """
    Load a CSV, Excel (.xlsx/.xls), or JSON file as the working DataFrame.
    Call this first when analyzing a file source.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > 500:
        return f"File too large ({size_mb:.0f}MB). Please provide a subset under 500MB."

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        else:
            return f"Unsupported format: {ext}. Supported: .csv, .xlsx, .xls, .json"

        _state["current_dataframe"] = df
        _log("load_file", f"Loaded {file_path}: {df.shape[0]} rows × {df.shape[1]} cols")

        summary = summarize_dataframe(df)
        return f"Loaded {os.path.basename(file_path)}: {df.shape[0]:,} rows × {df.shape[1]} columns.\n\n{summary}"

    except Exception as e:
        return f"Error loading file: {str(e)[:300]}"


# ── Tool 6: run_python ──────────────────────────────────────────────────────────

@tool
def run_python(code: str) -> str:
    """
    Execute pandas/numpy code against the current DataFrame.
    The variable `df` refers to the current working DataFrame.
    Available: df, pd, np.
    If you reassign `df`, the new DataFrame becomes the working DataFrame.
    Use this for analysis, transformations, aggregations, and computations.
    """
    df = _state["current_dataframe"]
    if df is None:
        return "No DataFrame loaded. Call load_file or run_sql first."

    output, updated_df = run_sandboxed(code, df)

    if updated_df is not None:
        _state["current_dataframe"] = updated_df
        _log("run_python", f"Code executed. DataFrame updated to {updated_df.shape}")
        output += f"\n\n[DataFrame updated: now {updated_df.shape[0]:,} rows × {updated_df.shape[1]} columns]"
    else:
        _log("run_python", "Code executed.")

    return truncate_output(output)


# ── Tool 7: plot_chart ──────────────────────────────────────────────────────────

@tool
def plot_chart(code: str, filename: str) -> str:
    """
    Generate a matplotlib chart and save it as a PNG.
    Available variables: df, pd, np, plt.
    The filename should be descriptive (no extension needed).
    Charts are saved to output/charts/<filename>.png.
    """
    df = _state["current_dataframe"]
    if df is None:
        return "No DataFrame loaded. Call load_file or run_sql first."

    output_dir = _state["output_directory"]
    safe_filename = re.sub(r'[^\w\-_]', '_', filename)
    chart_path = f"{output_dir}/charts/{safe_filename}.png"

    # Build sandbox namespace with plt
    import matplotlib.pyplot as plt
    namespace = {
        '__builtins__': {'print': print, 'range': range, 'len': len, 'enumerate': enumerate,
                         'zip': zip, 'list': list, 'str': str, 'int': int, 'float': float,
                         'round': round, 'min': min, 'max': max, 'sorted': sorted,
                         'True': True, 'False': False, 'None': None},
        'df': df.copy(),
        'pd': pd,
        'np': np,
        'plt': plt,
    }

    try:
        plt.figure(figsize=(10, 6))
        exec(compile(code, '<chart>', 'exec'), namespace)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close('all')

        _log("plot_chart", f"Saved {chart_path}")
        return f"Chart saved: {chart_path}"

    except Exception as e:
        plt.close('all')
        _log("plot_chart_error", str(e)[:200])
        return f"Chart error: {str(e)[:300]}"


# ── Tool 8: write_report ────────────────────────────────────────────────────────

@tool
def write_report(findings: str, filename: str) -> str:
    """
    Save the final analytical report as a markdown file.
    This should be the LAST tool called in every analysis.
    The findings parameter should contain the full markdown report.
    Reports are saved to output/reports/<filename>.md
    """
    output_dir = _state["output_directory"]
    safe_filename = re.sub(r'[^\w\-_]', '_', filename)
    report_path = f"{output_dir}/reports/{safe_filename}.md"

    try:
        # Add metadata header
        header = f"<!-- Generated by Junior Data Analyst Agent | {datetime.now().strftime('%Y-%m-%d %H:%M')} -->\n\n"
        full_content = header + findings

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        word_count = len(findings.split())
        _log("write_report", f"Saved {report_path} ({word_count} words)")
        return f"Report saved: {report_path} ({word_count:,} words)"

    except Exception as e:
        return f"Error saving report: {str(e)[:200]}"


# ── Agent Assembly ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior data analyst. You receive a data source (database or file)
and a question. Your job is to understand the data, analyze it rigorously,
and write a clear report with findings and supporting charts.

== Process for database sources ==
1. Call connect_database first with the connection string you were given.
2. Call list_tables to see what is available.
3. Call inspect_table on each table you intend to query, BEFORE querying it.
   Never query a table whose schema you have not inspected.
4. Use run_sql with targeted SELECT queries. Always use WHERE filters and
   LIMIT clauses where appropriate. Never SELECT * on tables over 10,000 rows
   without a LIMIT.
5. Once you have the data in a DataFrame, use run_python for further analysis,
   joins, transformations, and computations.

== Process for file sources ==
1. Call load_file with the file path.
2. Use run_python to inspect: df.head(), df.info(), df.describe(), null counts.
3. Use run_python for analysis.

== Both paths continue ==
6. Call plot_chart when a chart conveys more than a number can. Charts should
   have clear titles, axis labels, and a legend if multi-series.
7. Call write_report as the FINAL step. The report is the deliverable.

== Style for findings and reports ==
- State findings, not steps. Write "Revenue dropped 32% in March", not "I ran a groupby".
- Cite numbers precisely. Round only when it improves readability.
- When data is ambiguous, incomplete, or insufficient, say so explicitly. Do not guess.
- Surface anomalies — outliers, missing data, suspicious patterns — even when the user did not ask.
- Reference charts in the report by filename: "See revenue_trend.png for the full monthly view."

== Report structure ==
# [Descriptive title]

## Summary
[2-3 sentence executive summary with the headline finding]

## Key findings
[Bullet list, each with a specific numbered claim]

## Detail
[Section per finding with supporting numbers and chart references]

## Caveats
[Data quality issues, limitations, what could not be answered]

## Recommended next questions
[3-5 follow-ups that emerged from this analysis]

== HARD RULES — never violate ==
- NEVER write INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER, GRANT, REVOKE, or any modifying SQL.
- NEVER attempt to modify the source data in any way.
- If a user asks you to modify data, refuse — this agent is read-only by design.
- NEVER print or include more than 10 rows of raw data in any tool input or output.
- NEVER include raw PII in reports or charts. Aggregate, anonymize, or sample.
- ALWAYS inspect_table before querying a new table.
- ALWAYS load_file or inspect with run_python before analyzing a file.

== Stopping condition ==
Stop and call write_report when:
- You have a clear, specific, supported answer to the user's question, OR
- You have determined the data cannot answer the question.

Do not loop indefinitely. After 15 tool calls, write a report with what you have found.

Tone: Senior analyst briefing a CEO. Brief, accurate, useful, honest about uncertainty."""


TOOLS = [
    connect_database,
    list_tables,
    inspect_table,
    run_sql,
    load_file,
    run_python,
    plot_chart,
    write_report,
]


def build_agent():
    """Build and return the LangGraph ReAct agent."""
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=8096,
    )

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    return agent
