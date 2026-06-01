"""
safety.py — SQL safety guard + Python sandbox utilities.
All defensive code lives here, isolated for clarity and testability.
"""

import re
import io
import sys
import traceback
import pandas as pd
import numpy as np

# ── SQL Safety ─────────────────────────────────────────────────────────────────

SQL_SAFE_PATTERN = re.compile(r'^\s*(SELECT|WITH)\s+', re.IGNORECASE)

FORBIDDEN_KEYWORDS = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|GRANT|REVOKE|CREATE|REPLACE|MERGE|EXEC|EXECUTE|CALL)\b',
    re.IGNORECASE
)


def is_safe_query(query: str) -> tuple[bool, str]:
    """
    Returns (True, "") if query is safe to execute.
    Returns (False, reason) if it should be rejected.
    """
    stripped = query.strip()

    if not SQL_SAFE_PATTERN.match(stripped):
        return False, (
            "Refused: query must start with SELECT or WITH. "
            "This agent is read-only by design. "
            "Write operations require a different tool and explicit human approval."
        )

    if FORBIDDEN_KEYWORDS.search(stripped):
        match = FORBIDDEN_KEYWORDS.search(stripped)
        return False, (
            f"Refused: detected forbidden keyword '{match.group()}'. "
            "This agent only performs read-only SELECT analysis."
        )

    return True, ""


# ── Python Sandbox ──────────────────────────────────────────────────────────────

FORBIDDEN_PYTHON = re.compile(
    r'\b(import|__import__|exec|eval|open|os\.|sys\.|subprocess|socket|urllib|requests)\b'
)


def build_sandbox_namespace(df: pd.DataFrame) -> dict:
    """
    Returns a restricted globals dict for exec().
    Only df, pd, np, and safe builtins are available.
    """
    safe_builtins = {
        'print': print,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'round': round,
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'sorted': sorted,
        'type': type,
        'isinstance': isinstance,
        'hasattr': hasattr,
        'getattr': getattr,
        'None': None,
        'True': True,
        'False': False,
    }

    return {
        '__builtins__': safe_builtins,
        'df': df.copy() if df is not None else pd.DataFrame(),
        'pd': pd,
        'np': np,
    }


def run_sandboxed(code: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame | None]:
    """
    Execute code in a restricted namespace.
    Returns (output_string, updated_df_or_None).
    updated_df is set if the code reassigned `df`.
    """
    # Check for forbidden patterns
    if FORBIDDEN_PYTHON.search(code):
        match = FORBIDDEN_PYTHON.search(code)
        return f"Refused: '{match.group()}' is not allowed in analysis code.", None

    namespace = build_sandbox_namespace(df)
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    try:
        exec(compile(code, '<analysis>', 'exec'), namespace)
        output = stdout_capture.getvalue()

        # Check if df was reassigned
        new_df = namespace.get('df', None)
        if new_df is not None and not new_df.equals(df):
            updated_df = new_df
        else:
            updated_df = None

        # Also capture the last expression value if any
        # Try to get the last line as an expression
        lines = code.strip().split('\n')
        last_line = lines[-1].strip()
        try:
            last_val = eval(compile(last_line, '<expr>', 'eval'), namespace)
            if last_val is not None and str(last_val) not in output:
                output += str(last_val)
        except Exception:
            pass

        return output.strip() if output.strip() else "Code executed successfully (no output).", updated_df

    except Exception as e:
        return f"Error: {traceback.format_exc()}", None
    finally:
        sys.stdout = old_stdout


# ── Output Truncation ───────────────────────────────────────────────────────────

def truncate_output(text: str, max_chars: int = 3000) -> str:
    """
    Keep tool responses under a reasonable length.
    Truncates with a note if over the limit.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[... truncated — {len(text) - max_chars} chars omitted for context efficiency ...]"


def summarize_dataframe(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Return a compact summary of a DataFrame — never dump raw rows in bulk.
    """
    if df is None or df.empty:
        return "DataFrame is empty."

    lines = []
    lines.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")

    # Dtypes
    dtype_summary = df.dtypes.value_counts()
    dtype_str = ", ".join([f"{count} {dtype}" for dtype, count in dtype_summary.items()])
    lines.append(f"Types: {dtype_str}")

    # Null counts (only show columns with nulls)
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        null_str = ", ".join([f"{col}: {count:,}" for col, count in nulls.items()])
        lines.append(f"Nulls: {null_str}")
    else:
        lines.append("Nulls: none")

    # Sample rows
    lines.append(f"\nFirst {min(max_rows, len(df))} rows:")
    lines.append(df.head(max_rows).to_string(index=False))

    return "\n".join(lines)
