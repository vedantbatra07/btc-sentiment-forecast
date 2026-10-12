# Submission checklist

The assessment requests a repository URL, provider tokens/actor IDs and an evaluation report. Do not submit a price-only or synthetic report as if it were the completed sentiment study.

- Repository: share the published project URL.
- Evaluation: use `reports/live/evaluation.md` after real X and Tavily collection succeeds.
- Outlook: use `reports/live/sentiment_outlook.md`.
- Forecasts and charts: included alongside the report.
- Actor: the actual `APIFY_ACTOR_ID` is recorded in `reports/live/run_manifest.json`, together with collection settings. The default is `xquik/x-tweet-scraper`.
- Tavily: uses the standard `/search` endpoint; no actor ID applies.
- Exa: not used.
- Tokens: provide dedicated, limited-budget credentials to the reviewer through a private channel. Do not put tokens in the public repository, report or screenshots. Rotate them after review.

Before submission, run tests, verify the report describes real X data, check all six horizons, open every chart, and confirm the GitHub URL works for a signed-out reviewer. Read the methodology so you can explain timing, sampling limitations, the baselines and the trading rule in an interview.

The pipeline does not send email automatically.
