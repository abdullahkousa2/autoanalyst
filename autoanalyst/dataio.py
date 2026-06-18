"""Loading tabular data and describing it for the model.

`schema_summary` builds the compact context block the agent sees before it
writes any code: shape, columns + dtypes, a few sample rows, and a light numeric
summary. The bundled-sample registry lives here too (metadata only — the file
paths are resolved by whoever owns the samples directory).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: str | Path) -> tuple["pd.DataFrame", str]:
    """Read a CSV/Excel file into a DataFrame and build its schema summary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such data file: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df, schema_summary(df, name=path.stem)


def schema_summary(df: "pd.DataFrame", name: str = "dataset", head: int = 5) -> str:
    """A compact, model-friendly description of a DataFrame."""
    rows, cols = df.shape
    lines = [
        f"Dataset: {name}",
        f"Shape: {rows:,} rows x {cols} columns",
        "",
        "Columns (name : dtype):",
    ]
    lines += [f"  - {col} : {dtype}" for col, dtype in df.dtypes.items()]

    with pd.option_context("display.max_columns", None, "display.width", 200):
        lines += ["", f"First {head} rows:", df.head(head).to_string()]
        numeric = df.select_dtypes("number")
        if not numeric.empty:
            lines += ["", "Numeric summary:", numeric.describe().round(3).to_string()]
    return "\n".join(lines)


# --- curated sample datasets (metadata only) -----------------------------------
# Files are produced by scripts/make_samples.py into the repo's samples/ dir.
SAMPLES: dict[str, dict] = {
    "titanic": {
        "label": "Titanic — passenger survival",
        "file": "titanic.csv",
        "description": "891 passengers of the RMS Titanic: who survived, their "
        "class, sex, age, fare and port of embarkation.",
        "questions": [
            "What was the overall survival rate, and how did it differ by passenger class?",
            "Did women and children really survive more often than men? Show it with a chart.",
            "Is there a relationship between the fare a passenger paid and their survival?",
        ],
    },
    "tips": {
        "label": "Restaurant tips",
        "file": "tips.csv",
        "description": "244 restaurant bills: total bill, tip, party size, day, "
        "time and whether the customer was a smoker.",
        "questions": [
            "What is the average tip percentage, and does it change by day of the week?",
            "Do larger parties tip a higher or lower percentage? Show the trend.",
            "Compare average tips for lunch versus dinner.",
        ],
    },
    "penguins": {
        "label": "Palmer penguins",
        "file": "penguins.csv",
        "description": "344 penguins across 3 species: bill and flipper "
        "measurements, body mass, sex and island.",
        "questions": [
            "How do the three species differ in body mass? Show it with a chart.",
            "Is flipper length correlated with body mass across all penguins?",
            "Which island has the most species diversity?",
        ],
    },
    "ecommerce_sales": {
        "label": "E-commerce sales",
        "file": "ecommerce_sales.csv",
        "description": "A year of online orders: date, category, region, quantity, "
        "unit price, revenue and customer segment.",
        "questions": [
            "Which product category generated the most revenue? Show the breakdown.",
            "How did total monthly revenue trend over the year?",
            "Which region has the highest average order value?",
        ],
    },
}


def sample_registry() -> list[dict]:
    """Public list of samples (id + metadata), safe to send to the frontend."""
    return [{"id": sid, **{k: v for k, v in meta.items() if k != "file"}}
            for sid, meta in SAMPLES.items()]
