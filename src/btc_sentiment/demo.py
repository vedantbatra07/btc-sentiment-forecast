"""Deterministic fixtures for integration tests, never a substitute for X data."""
import numpy as np
import pandas as pd
from .storage import upsert, metadata


def populate(con):
    rng = np.random.default_rng(41)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    hourly = pd.date_range(start, periods=181*24, freq="h")
    prices = 40000*np.exp(np.cumsum(rng.normal(0, .003, len(hourly)+1)))
    rows = [dict(timestamp=t.isoformat(), open=float(prices[i]), close=float(prices[i+1]),
        high=float(max(prices[i:i+2])*1.001), low=float(min(prices[i:i+2])*.999),
        retrieved_at=t.isoformat(), source="SYNTHETIC_TEST_FIXTURE") for i,t in enumerate(hourly)]
    upsert(con,"prices",rows)
    posts, coverage = [], []
    for day in pd.date_range(start, periods=181, freq="D"):
        count = int(rng.integers(5,15))
        coverage.append(dict(day=day.isoformat(), run_id="SYNTHETIC", status="SUCCEEDED", returned_count=count,
            retained_count=count, cap=200, capped=0, retrieved_at=day.isoformat()))
        for j in range(count):
            text = ("Bitcoin looks excellent and promising" if rng.random() > .5 else "Bitcoin looks terrible and risky") + f" example {j}"
            posts.append(dict(id=f"fixture-{day.date()}-{j}",timestamp=(day+pd.Timedelta(hours=j)).isoformat(),
                text=text, likes=0,reposts=0,views=0,followers=0,author_id=str(j),retrieved_at=day.isoformat(),raw_json="{}"))
    upsert(con,"posts",posts)
    upsert(con,"coverage",coverage)
    metadata(con,"data_mode","synthetic")
