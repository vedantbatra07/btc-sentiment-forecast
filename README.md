# Bitcoin: X sentiment and price forecasting

A small, reproducible research pipeline for six daily Bitcoin price forecasts. It collects X posts through Apify, stores hourly BTC-USD prices and recent Tavily news in SQLite, and compares a sentiment-plus-price model with price-only and persistence baselines.

## Current status

The real-data run is complete: 9,000 X posts across 180 days, 4,315 hourly Coinbase candles and eight Tavily news results. All 16 tests pass. Start with the [evaluation report](reports/live/evaluation.md), [current sentiment outlook](reports/live/sentiment_outlook.md) and [six forecasts](reports/live/forecast.csv).

On 79 out-of-sample days, the sentiment model's n+0 MAE is $1,172.98 versus $971.59 for persistence. The sentiment trading rule returns 9.94% after 10 bps per-side costs, versus 0.64% for the price-only rule and 32.34% for buy-and-hold. Sentiment therefore does not establish a forecasting advantage in this sample. These are retrospective research results, not a live trading record.

The X sample is capped at 50 posts per day and typically spans only 7.7 minutes near the day end. One price day has five missing hours and is excluded together with affected feature windows. These limitations matter more than small differences in annualized performance ratios.

## Run locally

Use Python 3.12. Commands below assume you are in the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
cp .env.example .env
```

Set `APIFY_TOKEN` and `TAVILY_API_KEY` in `.env`. Keep that file private. The actor ID is `xquik/x-tweet-scraper`, configurable through `APIFY_ACTOR_ID`. Before collecting, inspect the actor's current plan restrictions and pricing. Free-plan restrictions can prevent the required historical collection.

Choose one fixed exclusive end date for prices and posts so that a run crossing midnight does not produce mismatched windows:

```bash
btc-sentiment prices --days 180 --end 2026-10-12
btc-sentiment posts --days 180 --end 2026-10-12 --per-day 50 --max-charge-usd 2.70 --workers 4
btc-sentiment news
btc-sentiment evaluate
python -m pytest -q
```

The chosen actor also bills platform usage separately; reserve additional headroom and check the account usage. The example permits up to $2.70 of actor charges across new runs in that invocation. It is not a price quote or a platform-wide budget; verify your account's limits. Never run it without agreeing to the spend. Each day is a separate bounded search, avoiding a single query that returns only the most recent days. Runs persist their IDs before polling, and completed days are skipped on restart. A failed run is not silently replaced with another paid run. A network failure during initial run creation can leave an ambiguous result: check the Apify console before retrying.

To verify the pipeline without credentials:

```bash
btc-sentiment demo
```

This uses a clearly marked synthetic database and writes only to `reports/demo/`. It is an integration test, not a market backtest. To evaluate real prices while X collection is pending:

```bash
btc-sentiment evaluate --price-only
```

## Forecast timing

`n` is the UTC day beginning at the forecast origin. At 00:05 UTC, the pipeline uses complete price bars and posts through n−1. It predicts the closes of n, n+1, …, n+5. Thus n+0 is a genuine next-close forecast, not the known current price. The anchor is n−1's close. This resolves the otherwise ambiguous n+0 requirement explicitly.

UTC is the storage and grouping timezone. `America/New_York` is provided for display; this respects EST/EDT rather than applying a fixed five-hour offset all year. See [methodology](docs/METHODOLOGY.md).

## What gets modeled

- Price: lagged log returns, trailing volatility, recent mean return and prior high/low range. Exchange trading volume is not used.
- X: daily mean VADER sentiment, trailing sentiment, sentiment change, positive/negative shares and sampled post counts. Only prior days enter a forecast.
- News: current context for the outlook, never a historical predictor.
- Engagement and followers: extracted and stored, but excluded from historical features because a retrospective count can include future activity.

The model is a standardized Ridge regression with fixed regularization. Six separate targets estimate cumulative log returns; forecasts convert them back to USD. With only 180 days, a transparent regularized model is a more defensible starting point than a large neural model. No claim is made that it must beat a random walk.

## Validation

Expanding training windows, 14-day chronological test blocks, at least 60 training rows and horizon-specific label purging. All preprocessing is fitted inside each training fold. The price-only ablation uses the same eligible rows as the sentiment model. Hyperparameters and the trading rule are fixed before testing. The report shows every horizon's MAE, RMSE, MAPE and direction accuracy, plus Sharpe, Sortino and drawdown from one daily long/cash strategy with execution costs.

## Outputs

| File | Purpose |
| --- | --- |
| `reports/live/evaluation.md` | Full real-data evaluation once collection is complete |
| `reports/live/sentiment_outlook.md` | One-page current outlook and recent news links |
| `reports/live/forecast.csv` | Six latest price forecasts and empirical residual bands |
| `reports/live/predictions_oos.csv` | Every out-of-sample prediction |
| `reports/live/fold_audit.csv` | Training and label timing audit |
| `reports/live/risk_metrics.csv` | Risk metrics and cost sensitivity |
| `reports/live/sentiment_price.png` | Time-aligned sentiment, sampled post count and price |
| `reports/live/forecast_matrix.png` | Error matrix by forecast origin and horizon |
| `reports/live/equity.png` | Net strategy and buy-and-hold curves |
| `reports/live/run_manifest.json` | Package versions, input hash and collection settings |
| `reports/live/data_quality.json` | Coverage, missing fields and within-day sampling span |
| `reports/live/latest_forecast.png` | All six price forecasts with empirical residual bands |
| `data/bitcoin.sqlite` | Local raw inputs, coverage ledger and joined daily table |

Raw posts, database files, tokens and saved models are excluded from Git. Model files are local and should only be loaded from trusted sources. The public repository contains code and aggregate evaluation artifacts.

## Repository layout

`ingest.py` handles external sources; `storage.py` defines the database; `features.py` constructs lagged inputs; `model.py` implements walk-forward fitting; `metrics.py` defines trading and risk calculations; `report.py` writes charts and Markdown. Tests check temporal leakage, label purging, gap handling, costs, timezones and API adapters.

## Sources

- [Apify actor input and output](https://apify.com/xquik/x-tweet-scraper)
- [Apify run endpoint and charge cap](https://docs.apify.com/api/v2/actors-runs-post)
- [Coinbase hourly candle API](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
- [Tavily search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)
- [VADER implementation and research reference](https://github.com/cjhutto/vaderSentiment)
- [Scikit-learn time-series cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

See [submission notes](docs/SUBMISSION.md) for the deliverable checklist and credential handoff.
