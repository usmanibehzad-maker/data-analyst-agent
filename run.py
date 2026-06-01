"""
run.py — Entry point for the Junior Data Analyst Agent.

Usage:
  # Analyze a database:
  python run.py --db "sqlite:///data/sample.sqlite" --question "Which products are most profitable?"

  # Analyze a file:
  python run.py --file data/sales_messy.csv --question "What is driving the Q2 revenue drop?"
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Junior Data Analyst Agent — powered by Claude + LangGraph"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", metavar="CONNECTION_STRING",
                        help="SQLAlchemy connection string for database analysis")
    source.add_argument("--file", metavar="FILE_PATH",
                        help="Path to CSV, Excel, or JSON file for file analysis")

    parser.add_argument("--question", required=True,
                        help="The analytical question to answer")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each agent step as it happens")

    return parser.parse_args()


def save_audit_log(run_log: list, output_dir: str, run_id: str):
    """Save the audit log for this run."""
    log_path = f"{output_dir}/logs/{run_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        for entry in run_log:
            f.write(f"[{entry['timestamp']}] {entry['action']}\n")
            f.write(f"  {entry['detail']}\n\n")
    return log_path


def main():
    args = parse_args()

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment or .env file.")
        sys.exit(1)

    # Import after env check
    from agent import build_agent, init_run, _state

    # Generate run ID
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_analyst"

    # Initialize state
    init_run(output_directory=args.output)

    # Build the agent
    print("\n" + "=" * 60)
    print("  JUNIOR DATA ANALYST AGENT")
    print("  Powered by Claude + LangGraph")
    print("=" * 60)

    if args.db:
        print(f"\n  Source:   Database")
        print(f"  DB:       {args.db.split('/')[-1].split('?')[0]}")
    else:
        print(f"\n  Source:   File")
        print(f"  File:     {args.file}")

    print(f"  Question: {args.question}")
    print(f"  Run ID:   {run_id}")
    print("\n" + "=" * 60)
    print("\nAgent is thinking...\n")

    # Build input message
    if args.db:
        user_message = (
            f"Database connection string: {args.db}\n\n"
            f"Question: {args.question}"
        )
    else:
        user_message = (
            f"File path: {args.file}\n\n"
            f"Question: {args.question}"
        )

    # Run the agent
    agent = build_agent()

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )

        # Extract final message
        messages = result.get("messages", [])
        final_message = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                if hasattr(msg, "type") and msg.type == "ai":
                    final_message = msg.content
                    break

        # Print tool calls if verbose
        if args.verbose:
            print("\n--- AGENT STEPS ---")
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"\n[TOOL] {tc['name']}")
                        args_preview = str(tc.get('args', {}))[:200]
                        print(f"  Args: {args_preview}")
                elif hasattr(msg, "type") and msg.type == "tool":
                    content_preview = str(msg.content)[:300]
                    print(f"  Result: {content_preview}")

        # Find output files
        report_files = []
        chart_files = []
        if os.path.exists(f"{args.output}/reports"):
            report_files = [f for f in os.listdir(f"{args.output}/reports") if f.endswith(".md")]
        if os.path.exists(f"{args.output}/charts"):
            chart_files = [f for f in os.listdir(f"{args.output}/charts") if f.endswith(".png")]

        # Save audit log
        log_path = save_audit_log(_state["run_log"], args.output, run_id)

        # Print summary
        print("\n" + "=" * 60)
        print("  ANALYSIS COMPLETE")
        print("=" * 60)

        if report_files:
            print("\n  Reports generated:")
            for f in report_files:
                print(f"    {args.output}/reports/{f}")

        if chart_files:
            print("\n  Charts generated:")
            for f in chart_files:
                print(f"    {args.output}/charts/{f}")

        print(f"\n  Audit log: {log_path}")

        if final_message:
            print(f"\n  Agent summary:\n")
            print(f"  {final_message[:500]}")

        print("\n" + "=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
