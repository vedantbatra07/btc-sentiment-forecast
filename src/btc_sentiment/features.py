import re
import json

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .storage import read

PRICE_FEATURES = ["return_1", "return_2", "return_3", "return_7", "volatility_7", "volatility_14", "mean_return_7", "range_1"]
SENTIMENT_FEATURES = ["sentiment_1", "sentiment_3", "sentiment_7", "sentiment_change", "positive_share", "negative_share", "log_posts", "post_change", "no_posts"]


def daily_prices(hourly):
    if hourly.empty:
        raise ValueError("No prices in database")
    h = hourly.copy()
    h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
    h = h.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    now = pd.Timestamp.now(tz="UTC")
    h = h[h.index + pd.Timedelta(hours=1) <= now]
    d = h.resample("D").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), hours=("close", "count"))
    # Never fill a missing market candle. Partial days are retained only as gaps.
    d.loc[d.hours != 24, ["open", "high", "low", "close"]] = np.nan
    execution = h[h.index.hour == 1]["open"].copy()
    execution.index = execution.index.floor("D")
    d["execution_open"] = execution
    return d


def daily_sentiment(posts, coverage, index):
    out = pd.DataFrame(index=index)
    known = pd.Series(False, index=index)
    capped = pd.Series(False, index=index)
    if not coverage.empty:
        c = coverage.copy()
        c["day"] = pd.to_datetime(c.day, utc=True)
        c = c.set_index("day")
        known = c.status.eq("SUCCEEDED").reindex(index, fill_value=False)
        capped = c.capped.eq(1).reindex(index, fill_value=False)
    out["covered"] = known
    out["capped"] = capped
    out["post_count"] = 0.0
    out["sentiment"] = np.nan
    out["positive_share"] = np.nan
    out["negative_share"] = np.nan
    if not posts.empty:
        p = posts.copy().drop_duplicates("id")
        p["timestamp"] = pd.to_datetime(p.timestamp, utc=True)
        p["day"] = p.timestamp.dt.floor("D")
        # Copied text should not get an unlimited number of votes within a day.
        p["normalized"] = p.text.map(lambda x: re.sub(r"https?://\S+", "", str(x)).strip().lower())
        p = p.sort_values("timestamp").drop_duplicates(["day", "normalized"])
        scorer = SentimentIntensityAnalyzer()
        p["score"] = p.text.map(lambda x: scorer.polarity_scores(re.sub(r"https?://\S+|@\w+", "", x).replace("#", ""))["compound"])
        p["positive"] = p.score >= 0.05
        p["negative"] = p.score <= -0.05
        g = p.groupby("day").agg(post_count=("score", "size"), sentiment=("score", "mean"),
            positive_share=("positive", "mean"), negative_share=("negative", "mean"))
        for col in g:
            out[col] = g[col].reindex(index)
    zero = known & out.post_count.fillna(0).eq(0)
    out.loc[zero, ["post_count", "sentiment", "positive_share", "negative_share"]] = 0
    out.loc[~known, ["post_count", "sentiment", "positive_share", "negative_share"]] = np.nan
    return out


def build_dataset(con):
    prices = read(con, "prices")
    request = con.execute("SELECT value FROM provenance WHERE key='price_request'").fetchone()
    if request:
        window = json.loads(request[0])
        stamp = pd.to_datetime(prices.timestamp, utc=True)
        prices = prices[(stamp >= pd.Timestamp(window['start'])) & (stamp < pd.Timestamp(window['end']))]
    daily = daily_prices(prices)
    # Add the next origin for a live forecast, without inventing a price bar.
    last_complete = daily.close.last_valid_index()
    if last_complete is None:
        raise ValueError("No complete UTC day with 24 hourly candles")
    idx = pd.date_range(daily.index.min(), last_complete + pd.Timedelta(days=1), freq="D")
    daily = daily.reindex(idx)
    sentiment = daily_sentiment(read(con, "posts"), read(con, "coverage"), idx)
    joined = daily.join(sentiment)
    x = pd.DataFrame(index=idx)
    r = np.log(daily.close / daily.close.shift(1))
    for lag in (1, 2, 3, 7):
        x[f"return_{lag}"] = r.shift(lag)
    x["volatility_7"] = r.shift(1).rolling(7).std()
    x["volatility_14"] = r.shift(1).rolling(14).std()
    x["mean_return_7"] = r.shift(1).rolling(7).mean()
    x["range_1"] = np.log(daily.high / daily.low).shift(1)
    x["sentiment_1"] = sentiment.sentiment.shift(1)
    x["sentiment_3"] = sentiment.sentiment.shift(1).rolling(3).mean()
    x["sentiment_7"] = sentiment.sentiment.shift(1).rolling(7).mean()
    x["sentiment_change"] = x.sentiment_1 - sentiment.sentiment.shift(2)
    x["positive_share"] = sentiment.positive_share.shift(1)
    x["negative_share"] = sentiment.negative_share.shift(1)
    x["log_posts"] = np.log1p(sentiment.post_count).shift(1)
    x["post_change"] = x.log_posts - np.log1p(sentiment.post_count).shift(2)
    x["no_posts"] = sentiment.post_count.shift(1).eq(0).astype(float)
    x.loc[sentiment.post_count.shift(1).isna(), "no_posts"] = np.nan
    x["anchor"] = daily.close.shift(1)
    for horizon in range(6):
        x[f"target_{horizon}"] = np.log(daily.close.shift(-horizon) / x.anchor)
    # SQL join is materialized explicitly for inspection and reproducibility.
    joined.rename_axis("day").reset_index().to_sql("daily_joined", con, if_exists="replace", index=False)
    return x, joined
