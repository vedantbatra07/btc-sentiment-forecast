import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import PRICE_FEATURES, SENTIMENT_FEATURES


def estimator():
    # Fixed before evaluation; no full-sample hyperparameter search.
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def eligible_training(x, features, horizon, cutoff):
    label_end = x.index + pd.Timedelta(days=horizon + 1)
    return x.loc[(label_end <= cutoff) & x[features + [f"target_{horizon}"]].notna().all(axis=1)]


def walk_forward(x, with_sentiment=True, min_train=60, block_days=14):
    groups = {"price_only": PRICE_FEATURES}
    if with_sentiment:
        groups["sentiment_price"] = PRICE_FEATURES + SENTIMENT_FEATURES
    common = PRICE_FEATURES + (SENTIMENT_FEATURES if with_sentiment else [])
    valid = x[common].notna().all(axis=1)
    labels_complete = x[[f'target_{h}' for h in range(6)]].notna().all(axis=1)
    candidates = x.index[valid & labels_complete]
    first = next((d for d in candidates if len(eligible_training(x, common, 5, d)) >= min_train), None)
    if first is None:
        raise ValueError("Not enough complete history for a 60-row training window plus six-day label purge")
    rows, audit = [], []
    # Same origins for every horizon/model; no overlap between test blocks.
    end = candidates.max()
    for cutoff in pd.date_range(first, end, freq=f"{block_days}D"):
        test = x.loc[(x.index >= cutoff) & (x.index < cutoff + pd.Timedelta(days=block_days)) & valid & labels_complete]
        if test.empty:
            continue
        for horizon in range(6):
            for name, features in groups.items():
                # For the ablation, train on identical rows as the sentiment model.
                train = eligible_training(x, common, horizon, cutoff)
                if len(train) < min_train:
                    raise ValueError("Insufficient training rows in fold")
                fitted = estimator().fit(train[features], train[f"target_{horizon}"])
                pred = fitted.predict(test[features])
                audit.append(dict(model=name, horizon=horizon, cutoff=cutoff.isoformat(), train_count=len(train),
                    train_first=train.index.min().isoformat(), train_last=train.index.max().isoformat(),
                    latest_label_end=(train.index.max() + pd.Timedelta(days=horizon+1)).isoformat(), test_count=len(test)))
                for (origin, row), estimate in zip(test.iterrows(), pred):
                    rows.append(dict(origin=origin, horizon=horizon, model=name, anchor=row.anchor,
                        actual_return=row[f"target_{horizon}"], predicted_return=float(estimate),
                        actual_price=row.anchor * np.exp(row[f"target_{horizon}"]), predicted_price=row.anchor * np.exp(estimate)))
            for origin, row in test.iterrows():
                rows.append(dict(origin=origin, horizon=horizon, model="persistence", anchor=row.anchor,
                    actual_return=row[f"target_{horizon}"], predicted_return=0.0,
                    actual_price=row.anchor * np.exp(row[f"target_{horizon}"]), predicted_price=row.anchor))
    result = pd.DataFrame(rows).sort_values(["origin", "horizon", "model"])
    return result, pd.DataFrame(audit)


def forecast_latest(x, oos, with_sentiment=True):
    name = "sentiment_price" if with_sentiment else "price_only"
    cols = PRICE_FEATURES + (SENTIMENT_FEATURES if with_sentiment else [])
    origin = x.index.max()
    row = x.loc[[origin]]
    if row[cols].isna().any(axis=None):
        raise ValueError("Latest forecast inputs are incomplete; refusing to issue a stale forecast")
    predictions, fitted_models = [], {}
    for h in range(6):
        train = eligible_training(x, cols, h, origin)
        if len(train) < 60:
            raise ValueError("Insufficient live training history")
        fitted = estimator().fit(train[cols], train[f"target_{h}"])
        estimate = float(fitted.predict(row[cols])[0])
        prior = oos[(oos.model == name) & (oos.horizon == h) & (oos.origin + pd.Timedelta(days=h+1) <= origin)]
        # Descriptive empirical residual band; no distribution-free coverage claim.
        q = float(np.quantile(abs(prior.actual_return - prior.predicted_return), 0.9, method="higher")) if len(prior) >= 20 else np.nan
        anchor = float(row.anchor.iloc[0])
        predictions.append(dict(origin=origin.isoformat(), cutoff_utc=(origin+pd.Timedelta(minutes=5)).isoformat(),
            cutoff_new_york=(origin+pd.Timedelta(minutes=5)).tz_convert("America/New_York").isoformat(),
            horizon=f"n+{h}", target_close_utc=(origin+pd.Timedelta(days=h+1)).isoformat(),
            model=name, anchor_price=anchor, predicted_price=anchor*np.exp(estimate),
            lower_empirical_90=anchor*np.exp(estimate-q), upper_empirical_90=anchor*np.exp(estimate+q),
            residual_sample=len(prior)))
        fitted_models[h] = fitted
    return pd.DataFrame(predictions), fitted_models
