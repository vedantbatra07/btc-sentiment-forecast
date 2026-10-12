import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(frame):
    def fmt(v):
        if pd.isna(v):
            return "N/A"
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join("---" for _ in frame.columns) + " |\n" + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in frame.itertuples(index=False, name=None))


def write_reports(out, joined, oos, forecast, metrics, risk, curves, audit, news, mode):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in {"predictions_oos": oos, "forecast": forecast, "forecast_metrics": metrics,
        "risk_metrics": risk, "equity": curves, "fold_audit": audit}.items():
        frame.to_csv(out/f"{name}.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    axes[0].plot(joined.index, joined.close, color="#133F69")
    axes[0].set_ylabel("BTC-USD close")
    axes[1].plot(joined.index, joined.sentiment, alpha=.4, label="Daily sample", color="#D48026")
    axes[1].plot(joined.index, joined.sentiment.rolling(7).mean(), label="7-day mean", color="#9B4F08")
    axes[1].axhline(0, color="gray", linewidth=.7)
    axes[1].set_ylabel("VADER compound")
    axes[1].legend(loc="upper left")
    axes[2].bar(joined.index, joined.post_count, color="#58988C")
    axes[2].set_ylabel("Sampled unique posts")
    axes[2].set_xlabel("UTC day")
    if not joined.sentiment.notna().any():
        for ax in axes[1:]:
            ax.clear()
            ax.text(.5,.5,"X data not collected for this run",ha="center",va="center",transform=ax.transAxes,color="#666666")
            ax.set_yticks([])
        axes[2].set_xlabel("UTC day")
    fig.suptitle(f"Bitcoin price and X sentiment — {mode}")
    fig.savefig(out/"sentiment_price.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    for name, group in curves.groupby("model"):
        ax.plot(pd.to_datetime(group.origin), group.equity, label=name)
    ax.set(title="Out-of-sample equity after costs", ylabel="Growth of $1", xlabel="UTC forecast day")
    ax.legend()
    fig.savefig(out/"equity.png", dpi=150)
    plt.close(fig)
    chosen = "sentiment_price" if "sentiment_price" in oos.model.unique() else "price_only"
    g = oos[oos.model == chosen].copy()
    g["absolute_pct_error"] = 100*abs(g.predicted_price-g.actual_price)/g.actual_price
    matrix = g.pivot(index="horizon", columns="origin", values="absolute_pct_error")
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ticks = np.unique(np.linspace(0, len(matrix.columns)-1, 7).astype(int))
    ax.set_xticks(ticks, [matrix.columns[t].strftime("%d %b") for t in ticks])
    ax.set_yticks(range(6), [f"n+{h}" for h in range(6)])
    ax.set(title="Forecast error matrix", xlabel="Out-of-sample origin (UTC)", ylabel="Target horizon")
    fig.colorbar(im, ax=ax, label="Absolute percentage error")
    fig.savefig(out/"forecast_matrix.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4),layout="constrained")
    positions = np.arange(6)
    ax.plot(positions,forecast.predicted_price,marker="o",color="#133F69",label="Forecast")
    if forecast.lower_empirical_90.notna().all():
        ax.fill_between(positions,forecast.lower_empirical_90,forecast.upper_empirical_90,alpha=.18,color="#133F69",label="Empirical residual band")
    ax.axhline(forecast.anchor_price.iloc[0],linestyle="--",color="#666666",label="Last known close")
    ax.set_xticks(positions,forecast.horizon)
    ax.set(title=f"Six closing-price forecasts — origin {forecast.origin.iloc[0][:10]}",ylabel="BTC-USD",xlabel="Daily horizon")
    ax.legend()
    fig.savefig(out/"latest_forecast.png",dpi=150)
    plt.close(fig)
    coverage = float(joined.covered.mean())
    incomplete = joined[joined.hours.notna() & joined.hours.ne(24)]
    gaps = ", ".join(str(d.date()) for d in incomplete.index) or "none"
    status = "Real X sentiment evaluation" if mode == "live" else ("Price-only benchmark — X features excluded" if mode == "price-only" else "Synthetic integration test — not market evidence")
    row = metrics[(metrics.model==chosen) & metrics.horizon.eq(0)].iloc[0]
    persistence = metrics[(metrics.model=='persistence') & metrics.horizon.eq(0)].iloc[0]
    gain = 100*(1-row.mae_usd/persistence.mae_usd)
    direction = 'lower' if gain >= 0 else 'higher'
    interpretation = f"The {chosen} model's n+0 MAE is ${row.mae_usd:,.2f}, compared with ${persistence.mae_usd:,.2f} for persistence. Its error is {abs(gain):.1f}% {direction} than the no-change baseline."
    report = f"""# Bitcoin forecasting evaluation

**Status: {status}.** Generated {pd.Timestamp.now(tz='UTC').isoformat()}.

{interpretation} The model and trading rule were not retuned to improve this result.

## Experiment

The model predicts six daily closing prices. At 00:05 UTC on day n, it uses prices and sampled posts through the end of n−1. The n+0 target is the close of day n, not an already observed spot price. The remaining targets are closes of n+1 through n+5. Stored timestamps are UTC; forecast exports also show America/New_York, including daylight-saving changes.

The first test origin is {oos.origin.min().date()}; the last is {oos.origin.max().date()}. There are {oos.origin.nunique()} distinct test origins. Training expands every 14 calendar days, with at least 60 training rows. Each horizon excludes every training label ending after its fold cutoff. Features and scaling are fit on training rows only. Ridge regularization is fixed at 10 before evaluation. The price-only ablation uses the same training rows and origins as the sentiment model. Persistence predicts the last observed close.

## Forecast accuracy

{markdown_table(metrics)}

MAE/RMSE are in USD. MAPE is in percent. Return MAE uses log returns. Direction accuracy is a fraction; persistence has no directional call. Overlapping multi-day targets are not independent observations. No significance or reliable future-profit claim follows from this short sample.

## Trading evaluation

Only the n+0 forecast drives the strategy. A predicted log return above 0.002 creates a long position; otherwise the strategy holds cash. This threshold was fixed before evaluation. Orders are modeled at the 01:00 UTC hourly open, 55 minutes after the forecast cutoff. Holdings earn the return to the next day's 01:00 open. There is no shorting or leverage. Costs apply to every position change and the final liquidation; sensitivity results show 0, 10 and 25 basis points per side. Hourly open execution is an approximation, not an order-book simulation.

{markdown_table(risk)}

Sharpe = sqrt(365) × mean daily net return / sample standard deviation. Sortino uses the root mean square of negative daily returns, including zero for non-negative days. Both use a zero risk-free rate and a zero minimum acceptable return. Undefined ratios are N/A. Maximum drawdown is the worst equity decline from a prior peak, including the initial $1. Ratios are annualized, but the test does not span a year. Buy-and-hold covers the identical test dates and pays entry/exit costs.

Total return and maximum drawdown are fractions: −0.10 denotes a 10% loss. The complete joined daily dataset and model features are materialized in SQLite for inspection.

## Data quality and limitations

Coverage flags are available for {coverage:.1%} of joined dates (including the next forecast day). X searches are daily capped samples ordered by recency, not the full population. Sample volume must not be described as total platform volume. Duplicated IDs and copied text within a day are removed. Missing scraping days remain missing rather than becoming neutral sentiment. Historical text is a retrospective snapshot: deleted/edited posts and search selection still introduce survivorship and selection bias. This is historical out-of-sample research, not a point-in-time archive or a live trading record.

Price days with fewer than 24 hourly candles: {gaps}. Those daily OHLC values and affected feature windows are excluded, not interpolated.

Likes, reposts, views and follower counts are stored for inspection but excluded from forecasting: their current values were not necessarily known at historical forecast time. Tavily news supplements the current outlook only and is excluded from all historical features. VADER measures general English tone, not reliably bullish/bearish investment intent; sarcasm and crypto jargon require manual review. 180 days provide a limited number of independent market regimes.

## Reproduction and artifacts

See README.md for exact commands, docs/METHODOLOGY.md for timing assumptions, and run_manifest.json for the environment and database hash. fold_audit.csv records the final training label boundary for every fold. predictions_oos.csv contains every untouched test prediction. forecast.csv contains the latest six outputs. Empirical bands use the 90th percentile of prior out-of-sample absolute log-return errors after their labels mature; these are descriptive residual bands, not guaranteed prediction intervals.

![Sentiment and price](sentiment_price.png)
![Equity](equity.png)
![Horizon error matrix](forecast_matrix.png)
![Latest forecasts](latest_forecast.png)
"""
    (out/"evaluation.md").write_text(report)
    if mode != "live":
        outlook = "# Bitcoin sentiment outlook\n\nThis run does not evaluate real X sentiment. No bullish or bearish outlook can be supported from price-only or synthetic inputs. News alone cannot substitute for the required X sample.\n\nAfter collecting the real data, `btc-sentiment evaluate` generates the one-page summary with recent sample counts, sentiment changes and linked news context.\n"
    else:
        last = joined.index[joined.covered].max()
        recent = joined.loc[last-pd.Timedelta(days=6):last]
        prior = joined.loc[last-pd.Timedelta(days=13):last-pd.Timedelta(days=7)]
        mean = recent.sentiment.mean()
        change = mean-prior.sentiment.mean()
        shift = 'softened' if change < 0 else 'strengthened'
        previous_close = joined.loc[last-pd.Timedelta(days=7),'close']
        price_change = recent.close.iloc[-1]/previous_close-1
        tone = "positive" if mean >= .05 else "negative" if mean <= -.05 else "mixed / neutral"
        headlines = []
        limit = pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=7)
        for _, row in news.sort_values(['published_at','relevance'],ascending=False,na_position='last').iterrows():
            published = pd.to_datetime(row.published_at, utc=True, errors="coerce")
            if pd.notna(published) and limit <= published <= pd.Timestamp.now(tz="UTC"):
                title = str(row.title).replace("[", "(").replace("]", ")").replace("\n", " ")
                headlines.append(f"- [{title}]({row.url}) — {published.date()}")
        outlook = f"""# Bitcoin sentiment outlook

As of {pd.Timestamp.now(tz='UTC').strftime('%d %B %Y, %H:%M UTC')}; latest sampled day: {last.date()}.

**Observed tone: {tone}.** The last seven days contain {int(recent.post_count.sum())} distinct sampled posts across {int(recent.covered.sum())} covered days. Mean daily VADER sentiment is {mean:.3f}, versus {prior.sentiment.mean():.3f} in the previous seven days. This is an equally weighted daily tone measure, not a price target or a recommendation. {int(recent.capped.sum())} recent days reached the requested sampling cap.

**Price context.** Bitcoin's latest completed UTC close is ${recent.close.iloc[-1]:,.2f}, a {price_change:+.2%} change over seven days on Coinbase. Price and sentiment are different measurements; agreement between them does not establish causality.

**Interpretation.** Sampled tone has {shift} by {abs(change):.3f} points from the previous week. VADER measures general wording, so positive tone is not necessarily bullish investment intent. These recency-capped samples cover only part of each day. Read this snapshot alongside the price-only comparison in evaluation.md; sentiment alone does not justify a trade.

**Recent news context.** These Tavily results are separate from the model inputs and do not establish a causal explanation for sentiment changes.

{chr(10).join(headlines[:4]) if headlines else 'No news with a verified publication date in the last seven days was returned.'}

**What to watch.** Check whether the seven-day shift persists across additional daily samples, review a sample of positive and negative posts manually, and compare realized prices with all six forecast horizons. Avoid treating overlapping forecast horizons as six independent confirmations. News and sentiment are snapshots and may change after collection.
"""
    (out/"sentiment_outlook.md").write_text(outlook)
