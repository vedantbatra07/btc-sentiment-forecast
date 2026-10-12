import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import joblib
import pandas as pd
from dotenv import load_dotenv

from .storage import connect, read
from .ingest import fetch_prices, fetch_posts, fetch_news
from .features import build_dataset
from .model import walk_forward, forecast_latest
from .metrics import forecast_metrics, backtest


def evaluate(db, out, price_only=False, demo=False):
    from .report import write_reports
    with connect(db) as con:
        synthetic = con.execute("SELECT value FROM provenance WHERE key='data_mode'").fetchone()
        if synthetic and json.loads(synthetic[0]) == "synthetic" and not demo:
            raise ValueError("Synthetic database: use the demo command, never live evaluation")
        x, joined = build_dataset(con)
        if not price_only:
            coverage = joined.covered.iloc[:-1]
            if not coverage.all():
                raise ValueError(f"X coverage is incomplete ({int(coverage.sum())}/{len(coverage)} days); finish scraping or run --price-only")
            if read(con,"news").empty and not demo:
                raise ValueError("Fetch Tavily news before producing the full submission report")
        oos, audit = walk_forward(x, with_sentiment=not price_only)
        forecast, models = forecast_latest(x, oos, with_sentiment=not price_only)
        metrics = forecast_metrics(oos)
        risk, curves = backtest(oos, joined)
        mode = "synthetic demo" if demo else "price-only" if price_only else "live"
        write_reports(out, joined, oos, forecast, metrics, risk, curves, audit, read(con,"news"), mode)
        provenance = dict(con.execute("SELECT key,value FROM provenance"))
    modeldir = Path("models")/("demo" if demo else "price-only" if price_only else "live")
    modeldir.mkdir(parents=True,exist_ok=True)
    joblib.dump(models,modeldir/"ridge.joblib")
    manifest = {"mode": mode, "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "python": platform.python_version(), "database_sha256": hashlib.sha256(Path(db).read_bytes()).hexdigest(),
        "packages": {p: importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","vaderSentiment","requests","matplotlib","joblib"]},
        "provenance": {k:json.loads(v) for k,v in provenance.items()},
        "model": "StandardScaler + Ridge(alpha=10)", "seed": "deterministic estimator", "horizons": list(range(6)),
        "train_min": 60, "test_block_days": 14, "cost_bps": 10, "signal_threshold_log_return": .002}
    (Path(out)/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"Wrote {mode} reports to {out}; {oos.origin.nunique()} test origins.")


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description="Bitcoin X sentiment forecasting")
    parser.add_argument("--db", default="data/bitcoin.sqlite")
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prices",help="Fetch 180 days of public hourly Coinbase candles")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--end", help="Exclusive UTC day, YYYY-MM-DD; default today's UTC date")
    p = sub.add_parser("posts",help="Fetch X posts; incurs Apify charges")
    p.add_argument("--days",type=int,default=180)
    p.add_argument("--end")
    p.add_argument("--per-day",type=int,default=200)
    p.add_argument("--max-charge-usd",type=float,required=True)
    sub.add_parser("news",help="Fetch current Tavily news")
    p = sub.add_parser("evaluate")
    p.add_argument("--price-only",action="store_true")
    p.add_argument("--out")
    sub.add_parser("demo",help="Run a separate synthetic integration test")
    args = parser.parse_args(argv)
    if args.command == "demo":
        from .demo import populate
        db = Path("data/demo.sqlite")
        if db.exists():
            db.unlink()
        with connect(db) as con:
            populate(con)
        evaluate(db,"reports/demo",demo=True)
    elif args.command == "evaluate":
        evaluate(args.db,args.out or ("reports/price_only" if args.price_only else "reports/live"),args.price_only)
    else:
        with connect(args.db) as con:
            if args.command == "news":
                print(f"Stored {fetch_news(con)} news items")
            else:
                end = pd.Timestamp(args.end,tz="UTC") if args.end else pd.Timestamp.now(tz="UTC").floor("D")
                if end != end.floor("D") or end > pd.Timestamp.now(tz="UTC").floor("D") or args.days < 1:
                    raise ValueError("Use a completed UTC date and a positive number of days")
                start = end-pd.Timedelta(days=args.days)
                if args.command == "prices":
                    print(f"Stored {fetch_prices(con,start,end)} hourly prices")
                else:
                    fetch_posts(con,start,end,args.max_charge_usd,args.per_day)


if __name__ == "__main__":
    try:
        main()
    except (ValueError,RuntimeError) as exc:
        sys.exit(str(exc))
