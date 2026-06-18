"""Build the bundled sample datasets into ../samples/.

Fetches three well-known public datasets (Titanic, restaurant tips, Palmer
penguins) and generates a synthetic but realistic e-commerce sales table. Run
once; the CSVs are committed so the app and CI never need network access.

    python scripts/make_samples.py
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

# Stable raw-CSV sources (the seaborn-data / datasciencedojo mirrors).
URLS = {
    "titanic.csv":
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "tips.csv":
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
    "penguins.csv":
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
}


def fetch_public() -> None:
    SAMPLES.mkdir(exist_ok=True)
    for name, url in URLS.items():
        print(f"  fetching {name} …", end=" ", flush=True)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        (SAMPLES / name).write_bytes(data)
        print(f"{len(data):,} bytes")


def make_ecommerce(n: int = 2400, seed: int = 7) -> None:
    """A year of synthetic online orders with believable structure."""
    rng = np.random.default_rng(seed)
    categories = {
        "Electronics": (120, 600), "Home & Kitchen": (20, 180),
        "Clothing": (15, 90), "Books": (8, 40), "Sports": (25, 200),
        "Beauty": (10, 70),
    }
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office"]

    cat_names = list(categories)
    cat_weights = np.array([0.28, 0.22, 0.2, 0.1, 0.12, 0.08])
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=n), unit="D"
    )
    cats = rng.choice(cat_names, size=n, p=cat_weights)
    lo = np.array([categories[c][0] for c in cats])
    hi = np.array([categories[c][1] for c in cats])
    unit_price = np.round(rng.uniform(lo, hi), 2)
    qty = rng.integers(1, 6, size=n)

    df = pd.DataFrame({
        "order_id": [f"ORD-{100000 + i}" for i in range(n)],
        "order_date": dates,
        "category": cats,
        "region": rng.choice(regions, size=n),
        "customer_segment": rng.choice(segments, size=n, p=[0.6, 0.25, 0.15]),
        "quantity": qty,
        "unit_price": unit_price,
    })
    df["revenue"] = np.round(df["quantity"] * df["unit_price"], 2)
    df = df.sort_values("order_date").reset_index(drop=True)
    df.to_csv(SAMPLES / "ecommerce_sales.csv", index=False)
    print(f"  generated ecommerce_sales.csv ({len(df):,} rows)")


if __name__ == "__main__":
    print(f"writing samples to {SAMPLES}")
    fetch_public()
    make_ecommerce()
    print("done.")
