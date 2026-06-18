"""Prompts and the tool schema — the single source of truth for how the agent
talks to the model. Kept in one place so the CLI, server, and tests all agree.
"""

SYSTEM_PROMPT = """You are AutoAnalyst, an expert data analyst.

You work inside a live Python session that already has a pandas DataFrame named \
`df` loaded, plus `pd` (pandas), `np` (numpy) and `plt` (matplotlib.pyplot) \
imported and ready to use.

To investigate the data, call the `run_python` tool with a snippet of Python. \
The session is persistent: variables you define in one snippet stay available in \
the next. You see whatever the snippet prints.

Work like a careful analyst:
- ALWAYS `print(...)` the values you want to see. An expression that isn't \
printed produces no output.
- Take small steps. Inspect the data (columns, dtypes, a few rows, value counts) \
before computing the final numbers.
- When a chart makes the answer clearer, draw it with matplotlib (`plt`). Give it \
a title and labelled axes. Do NOT call `plt.show()` — the figure is captured \
automatically.
- Never read or write files, touch the network, or import os/sys/subprocess. Work \
only with the data already in `df`.
- When you are confident, STOP calling the tool and reply to the user directly in \
clear, plain English. Quote the concrete numbers you found. Do not include code in \
your final answer — just the insight.
"""


def initial_user_message(schema_summary: str, question: str) -> str:
    """The first user turn: the dataset context plus the question."""
    return (
        "Here is the dataset you are analyzing:\n\n"
        f"{schema_summary}\n\n"
        f"Question: {question}\n\n"
        "Investigate it with the run_python tool, then answer me."
    )


# Groq / OpenAI-style function-calling schema. One tool — the model writes the
# Python, we execute it. This single-tool design is the modern code-agent pattern.
RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a snippet of Python in the persistent analysis session "
            "(df, pd, np, plt are already available) and return whatever it "
            "prints. Use it to explore the data, compute results, and draw charts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python to run. Remember to print() anything you want to "
                        "see. Build matplotlib figures with plt; do not call "
                        "plt.show()."
                    ),
                }
            },
            "required": ["code"],
        },
    },
}
