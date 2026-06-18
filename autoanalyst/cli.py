"""`autoanalyst` command-line entry point.

    autoanalyst -q "What was the survival rate by class?" --csv samples/titanic.csv

Loads the file, runs the agent, and streams each step (the code it wrote, the
output, a chart marker) to the terminal, followed by the final answer.
"""
from __future__ import annotations

import argparse
import os
import sys

# ANSI styling — Windows Terminal / PowerShell / Jupyter all handle these.
_C = {
    "dim": "\033[2m", "bold": "\033[1m", "cyan": "\033[36m", "green": "\033[32m",
    "red": "\033[31m", "magenta": "\033[35m", "reset": "\033[0m",
}


def _c(text: str, *styles: str) -> str:
    return "".join(_C[s] for s in styles) + text + _C["reset"]


def _print_step(step) -> None:
    print(_c(f"── step {step.n} · run_python ──", "cyan", "bold"))
    print(step.code.rstrip())
    if step.stdout.strip():
        print(_c(step.stdout.rstrip(), "dim"))
    if step.error:
        print(_c(step.error.rstrip(), "red"))
    if step.charts:
        print(_c(f"[chart rendered — {len(step.charts)} figure(s)]", "magenta"))
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="autoanalyst",
        description="Ask a question about a CSV/Excel file; an LLM agent writes "
        "and runs the analysis itself.",
    )
    parser.add_argument("-q", "--question", required=True, help="what to find out")
    parser.add_argument("--csv", "--data", dest="data", required=True,
                        help="path to a .csv or .xlsx file")
    parser.add_argument("--model", default=None, help="override the Groq model id")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args(argv)

    # Load a local .env if one is present (GROQ_API_KEY).
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Put it in a .env file or your environment "
              "(free key at https://console.groq.com).", file=sys.stderr)
        return 2

    # Imported here so `--help` and the key check don't pay the import cost.
    from .agent import DEFAULT_MODEL, Analyst
    from .dataio import load_table

    try:
        df, schema = load_table(args.data)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(_c("AutoAnalyst", "bold", "green") +
          f" — {df.shape[0]:,} rows × {df.shape[1]} cols from {args.data}")
    print(_c(f"Q: {args.question}", "bold") + "\n")

    analyst = Analyst(df, schema, model=args.model or DEFAULT_MODEL,
                      max_steps=args.max_steps)
    result = analyst.run(args.question, on_step=_print_step)

    print(_c("▶ Answer", "bold", "green"))
    print(result.answer)
    print(_c(f"\n({result.steps_used} step(s), stopped: {result.stopped})", "dim"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
