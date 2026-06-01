# Junior Data Analyst Agent
### Powered by Claude + LangGraph | Built by Behzad Usmani — UMC Tech AI

An autonomous AI agent that replaces a junior data analyst. Give it a database or file and a business question — it inspects the data, runs analysis, generates charts, and delivers a professional markdown report. No human in the loop.

---

## What it does

| Junior Analyst Task | This Agent |
|---|---|
| Connect to database | `connect_database` tool |
| Explore tables and schema | `list_tables` + `inspect_table` |
| Write SQL queries | `run_sql` (read-only, validated) |
| Clean and transform data | `run_python` (sandboxed pandas) |
| Build charts | `plot_chart` (matplotlib) |
| Write the report | `write_report` (markdown deliverable) |

---

## Architecture

```
User question + data source
         ↓
  LangGraph ReAct Agent
  (Claude Sonnet backbone)
         ↓
  Reason → Select Tool → Observe → Repeat
         ↓
  8 Tools: 5 data ingestion + 3 analysis/output
         ↓
  output/reports/*.md + output/charts/*.png
```

**Single-agent ReAct** — not multi-agent. Data analysis is sequential reasoning, not parallel coordination. The agent loops through reason→act→observe until the analysis is complete.

---

## Safety

Three layers of read-only enforcement:
1. **Code-level**: `run_sql` rejects any non-SELECT query with a regex guard
2. **Prompt-level**: System prompt hard rules forbid write operations
3. **Operational**: Designed to work with read-only database credentials

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key" > .env

# 3. Create demo data
python data/create_demo_data.py

# 4a. Run database analysis
python run.py --db "sqlite:///data/sample.sqlite" --question "Which customer segments are most profitable and why?"

# 4b. Run file analysis
python run.py --file data/sales_messy.csv --question "What is driving revenue variance across regions and sales reps?"

# 5. Add --verbose to see every agent step
python run.py --db "sqlite:///data/sample.sqlite" --question "What happened to revenue in March 2025?" --verbose
```

---

## Output

After every run:
```
output/
├── reports/    ← Markdown report (the deliverable)
├── charts/     ← PNG charts referenced in the report
└── logs/       ← Audit trail of every tool call
```

---

## Built with

- **LangGraph** — ReAct agent loop with state management
- **Claude Sonnet** — reasoning and analysis
- **pandas + matplotlib** — data processing and visualization
- **SQLAlchemy** — database connectivity (PostgreSQL, MySQL, SQLite)

---

*Built by Behzad Usmani | UMC Tech AI*
