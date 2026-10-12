# Research design

## Information timing

For origin D at 00:05 UTC, every feature uses observations from days strictly earlier than D. Target h is `log(close[D+h] / close[D-1])`. Its label becomes available at midnight starting D+h+1. A training row is eligible only if that label boundary is at or before the training cutoff. The 14-day test block is predicted using one frozen model per horizon. Later folds may use earlier test outcomes only once those outcomes have matured, as they would in an expanding-window deployment.

Example: a model trained at 00:05 on 12 October can use the n+5 label for a 6 October origin, whose final close is 11 October. It cannot use that label for a 7 October origin, whose close is still in the future.

Daily Coinbase OHLC comes from 24 completed hourly candles. Missing hours invalidate the daily bar; no interpolation or future fill occurs. The latest forecast refuses missing inputs. Historical rows without the required lag window are excluded. The report discloses short histories and missing bars.

## Historical scraping limitations

Daily bounded X queries combine BTC, $BTC and Bitcoin, request English posts and exclude reposts. IDs are deduplicated globally; normalized copied text is deduplicated within a day. The sample is ordered by recency and capped. It is not a census or a random sample. The cap creates strong within-day recency bias. Counts measure the retained sample, not total X activity. Results should be stress-tested with denser, intraday-stratified collection when more budget is available.

Posts are grouped by creation time for historical research, while retrieval time is stored separately. This assumes retrieved text approximates what existed then. Deletions, edits, moderation and search ranking can violate that assumption. The backtest is therefore a retrospective, temporally split study, not evidence of a fully point-in-time data feed. A production system should archive posts continuously as they arrive and apply an ingestion-latency cutoff.

Current engagement and follower counts are not historical snapshots. They are stored to meet the extraction requirement but excluded from model weights, filters and features. Future popularity must not decide the importance of an old post.

## Sentiment

VADER is deterministic and designed for social text. URLs and handles are removed; words in hashtags are retained. Compound scores are averaged daily. Positive and negative shares use the library's conventional ±0.05 thresholds. Neutral tone is different from missing data: a successfully searched day with zero posts receives zero tone plus an explicit no-post indicator; an uncollected day remains missing. The full evaluation requires coverage of every requested day.

VADER is not a financial intent classifier. Financial sarcasm, slang, mixed sentiment and posts that mention Bitcoin in passing are known weaknesses. A useful next step is a human-labelled sample of bullish, bearish and neutral posts. Fine-tuning such a classifier on the test period would invalidate the existing evaluation.

## Model and uncertainty

Price features and sentiment features enter a StandardScaler followed by Ridge(alpha=10). The regularization and feature set are fixed; there is no retrospective hyperparameter selection. The same eligible historical rows train the price-only ablation. Persistence is the last known close at every horizon.

Latest forecasts include an empirical 90th-percentile absolute residual band from matured historical out-of-sample errors when at least 20 residuals exist. Overlapping multi-day outcomes and changing market regimes mean nominal 90% coverage is not guaranteed. Bands are omitted with insufficient residual history.

## Trading and metrics

The n+0 signal is long when predicted log return exceeds 0.002; otherwise it is cash. It executes at the 01:00 hourly open, after the 00:05 forecast cutoff, and rebalances at 01:00 the next day. This deliberately avoids execution at an already observed prior close. Execution data are not features. The forecast target and trading holding interval differ by one hour and should not be conflated.

Daily simple returns include position-change costs and final liquidation. Buy-and-hold spans the same dates. Costs are 10 basis points per side in the main result, with 0 and 25 basis-point scenarios. There is no borrowing, leverage, shorting or funding charge. Cash earns zero. No strategy is selected by maximizing reported Sharpe. If test dates or execution prices are missing, the strategy evaluation fails instead of silently bridging a gap.

Sharpe uses the sample standard deviation and sqrt(365) annualization. Sortino uses the root mean square of downside relative to zero, with the full observation count in the denominator. Undefined ratios remain missing. Drawdown includes starting capital so an initial loss is counted.

## News

Tavily retrieves recent news and stores both publication and retrieval timestamps. News appears in the outlook with links and dates, but is excluded from historical training. Search relevance is not sentiment and is never interpreted as a bullishness score. Exa is not used because the brief specifically requests Tavily for news.

## Interpretation

Report weak or negative results plainly. Compare forecast errors with persistence and the trading rule with buy-and-hold after costs. A high annualized ratio over a short sample is unstable. Multi-day targets overlap, so ordinary independent-sample significance calculations would overstate evidence. The next credible validation step is a frozen live paper-trading period with newly collected posts, not repeated tuning on these same test dates.
