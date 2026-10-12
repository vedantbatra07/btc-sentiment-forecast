"""Public prices, bounded Apify runs, and current Tavily news."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .storage import connect, metadata, upsert


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def session():
    s = requests.Session()
    # GETs may retry. A paid POST is never retried automatically.
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def response_json(response):
    if not response.ok:
        # Never include a request URL, body or headers: they may contain secrets.
        raise RuntimeError(f"Data service returned HTTP {response.status_code}")
    return response.json()


def fetch_prices(con, start, end, http=None):
    """Read hourly BTC-USD candles; end is exclusive. No incomplete bars."""
    http = http or session()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    now = pd.Timestamp.now(tz="UTC")
    cursor = start
    total = 0
    while cursor < end:
        stop = min(cursor + pd.Timedelta(hours=250), end)
        data = response_json(http.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
            params={"granularity": 3600, "start": cursor.isoformat(), "end": stop.isoformat()}, timeout=45))
        if not isinstance(data, list):
            raise ValueError("Unexpected Coinbase response")
        rows = []
        for item in data:
            ts, low, high, op, close, _volume = item
            stamp = pd.Timestamp(ts, unit="s", tz="UTC")
            if cursor <= stamp < stop and stamp + pd.Timedelta(hours=1) <= now:
                if not (0 < low <= min(op, close) <= max(op, close) <= high):
                    raise ValueError("Invalid OHLC candle")
                rows.append(dict(timestamp=stamp.isoformat(), open=op, high=high, low=low,
                    close=close, retrieved_at=utcnow(), source="coinbase-exchange/BTC-USD"))
        upsert(con, "prices", rows)
        total += len(rows)
        cursor = stop
        time.sleep(0.12)
    metadata(con, "price_request", {"start": start.isoformat(), "end": end.isoformat(), "rows": total})
    return total


def normalize_post(item, fetched_at):
    if not isinstance(item, dict) or not item.get("id") or not item.get("text") or not item.get("createdAt"):
        raise ValueError("Post is missing id, text, or createdAt; check actor schema")
    ts = pd.to_datetime(item["createdAt"], utc=True, errors="raise")
    author = item.get("author") or {}
    def count(value):
        if value is None or value == "":
            return None
        return max(0, int(value))
    return dict(id=str(item["id"]), timestamp=ts.isoformat(), text=item["text"],
        likes=count(item.get("likeCount")), reposts=count(item.get("retweetCount")),
        views=count(item.get("viewCount")), followers=count(author.get("followers")),
        author_id=str(author.get("id") or ""), retrieved_at=fetched_at,
        raw_json=json.dumps(item, ensure_ascii=False))


def fetch_posts(con, start, end, max_charge_usd, per_day=200, http=None, write_config=True):
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise ValueError("Set APIFY_TOKEN in your local .env before scraping")
    if per_day < 50 or max_charge_usd <= 0:
        raise ValueError("Use at least 50 posts/day and a positive maximum charge")
    http = http or session()
    actor = os.getenv("APIFY_ACTOR_ID", "xquik/x-tweet-scraper")
    headers = {"Authorization": f"Bearer {token}"}
    days = pd.date_range(start, end, freq="D", inclusive="left")
    pending = [d for d in days if not con.execute("SELECT 1 FROM coverage WHERE day=? AND status='SUCCEEDED'", (d.isoformat(),)).fetchone()]
    if not pending:
        return 0
    budget_each = max_charge_usd / len(pending)
    if write_config:
        metadata(con, "apify_config", {"actor_id": actor, "per_day": per_day, "max_charge_usd": max_charge_usd,
            "query": "(BTC OR $BTC OR Bitcoin) lang:en -filter:retweets", "sampling": "Latest, daily capped"})
    for day in pending:
        stop = day + pd.Timedelta(days=1)
        existing = con.execute("SELECT run_id,status FROM coverage WHERE day=?", (day.isoformat(),)).fetchone()
        if existing and existing[0]:
            run_id = existing[0]
        else:
            payload = {"searchTerms": [f"(BTC OR $BTC OR Bitcoin) lang:en -filter:retweets since:{day.date()} until:{stop.date()}"],
                "maxItems": per_day, "sort": "Latest"}
            if actor == "xquik/x-tweet-scraper":
                payload.pop("sort")
                payload["queryType"] = "Latest"
                payload["mode"] = "search"
            elif actor.startswith("apidojo/"):
                payload["disableMaximization"] = True
            run = response_json(http.post(f"https://api.apify.com/v2/acts/{actor.replace('/', '~')}/runs",
                headers=headers, params={"maxTotalChargeUsd": budget_each}, json=payload, timeout=60))["data"]
            run_id = run["id"]
            upsert(con, "coverage", [dict(day=day.isoformat(), run_id=run_id, status="RUNNING",
                returned_count=0, retained_count=0, cap=per_day, capped=0, retrieved_at=utcnow())])
        deadline = time.monotonic() + 1800
        while True:
            run = response_json(http.get(f"https://api.apify.com/v2/actor-runs/{run_id}", headers=headers, timeout=45))["data"]
            status = run["status"]
            if status == "SUCCEEDED":
                break
            if status in {"FAILED", "ABORTED", "TIMED-OUT"}:
                raise RuntimeError(f"Apify run {run_id} ended {status}. Review it before retrying; no new run was started.")
            if time.monotonic() > deadline:
                raise RuntimeError(f"Run {run_id} is still pending; rerun to resume polling without a new paid run")
            time.sleep(5)
        items, offset = [], 0
        while True:
            batch = response_json(http.get(f"https://api.apify.com/v2/datasets/{run['defaultDatasetId']}/items",
                headers=headers, params={"format": "json", "clean": "true", "offset": offset, "limit": 1000}, timeout=60))
            items.extend(batch)
            if len(batch) < 1000:
                break
            offset += len(batch)
        rows = [normalize_post(item, utcnow()) for item in items]
        rows = [r for r in rows if day <= pd.Timestamp(r["timestamp"]) < stop]
        # Some actors return a whole final page beyond maxItems.
        rows = sorted(rows, key=lambda r: r["timestamp"], reverse=True)[:per_day]
        metadata(con, f"apify_run_{day.date()}", {"run_id": run_id, "actor_id": actor,
            "build_id": run.get("buildId"), "build_number": run.get("buildNumber"),
            "dataset_id": run.get("defaultDatasetId"), "usage_total_usd": run.get("usageTotalUsd"),
            "charged_event_counts": run.get("chargedEventCounts"), "status": status})
        upsert(con, "posts", rows)
        upsert(con, "coverage", [dict(day=day.isoformat(), run_id=run_id, status=status,
            returned_count=len(items), retained_count=len(rows), cap=per_day,
            capped=int(len(items) >= per_day), retrieved_at=utcnow())])
        print(f"X {day.date()}: {len(rows)} posts, capped={len(items) >= per_day}", flush=True)
    return len(pending)


def fetch_posts_parallel(db, start, end, max_charge_usd, per_day=50, workers=4):
    """One SQLite connection per worker; fixed per-day caps sum to the invocation cap."""
    if not 1 <= workers <= 4:
        raise ValueError("Use one to four workers")
    days = pd.date_range(start, end, freq="D", inclusive="left")
    with connect(db) as con:
        pending = [d for d in days if not con.execute("SELECT 1 FROM coverage WHERE day=? AND status='SUCCEEDED'",(d.isoformat(),)).fetchone()]
        if not pending:
            return 0
        cap = max_charge_usd / len(pending)
        metadata(con,"apify_config",{"actor_id":os.getenv("APIFY_ACTOR_ID","xquik/x-tweet-scraper"),
            "per_day":per_day,"invocation_max_charge_usd":max_charge_usd,"per_run_cap_usd":cap,
            "start":pd.Timestamp(start).isoformat(),"end":pd.Timestamp(end).isoformat(),"workers":workers,
            "query":"(BTC OR $BTC OR Bitcoin) lang:en -filter:retweets","sampling":"Latest, daily capped"})
    def run(day):
        with connect(db) as con:
            return fetch_posts(con,day,day+pd.Timedelta(days=1),cap,per_day,write_config=False)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run,d) for d in pending]
        try:
            for job in as_completed(jobs):
                job.result()
        except Exception:
            for job in jobs:
                job.cancel()
            raise
    return len(pending)


def fetch_news(con, http=None):
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("Set TAVILY_API_KEY in your local .env before fetching news")
    http = http or session()
    payload = {"query": "Bitcoin BTC market recent news", "topic": "news", "time_range": "week",
        "max_results": 8, "search_depth": "basic", "include_answer": False, "include_raw_content": False}
    result = response_json(http.post("https://api.tavily.com/search", headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=60))
    rows = []
    for item in result.get("results", []):
        if not item.get("url"):
            continue
        published = pd.to_datetime(item.get("published_date"), utc=True, errors="coerce")
        rows.append(dict(url=item["url"], title=item.get("title", ""), content=item.get("content", ""),
            published_at=None if pd.isna(published) else published.isoformat(), retrieved_at=utcnow(), relevance=item.get("score")))
    upsert(con, "news", rows)
    metadata(con, "tavily_request", {"query": payload, "retrieved_at": utcnow(), "count": len(rows)})
    return len(rows)
