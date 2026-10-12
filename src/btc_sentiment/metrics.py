import numpy as np
import pandas as pd


def risk_metrics(returns):
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Risk metrics require at least two finite returns above -100%")
    equity = np.r_[1.0, np.cumprod(1+r)]
    drawdown = equity / np.maximum.accumulate(equity) - 1
    sd = r.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(r, 0)**2))
    return {"days": len(r), "total_return": float(equity[-1]-1),
        "sharpe": float(np.sqrt(365)*r.mean()/sd) if sd > 1e-12 else None,
        "sortino": float(np.sqrt(365)*r.mean()/downside) if downside > 1e-12 else None,
        "max_drawdown": float(drawdown.min())}


def forecast_metrics(oos):
    rows = []
    for (name, h), g in oos.groupby(["model", "horizon"]):
        error = g.predicted_price-g.actual_price
        rows.append(dict(model=name, horizon=h, observations=len(g), mae_usd=float(abs(error).mean()),
            rmse_usd=float(np.sqrt(np.mean(error**2))), mape_pct=float(100*np.mean(abs(error)/g.actual_price)),
            return_mae=float(np.mean(abs(g.predicted_return-g.actual_return))),
            direction_accuracy=None if name == "persistence" else float(np.mean(np.sign(g.predicted_return)==np.sign(g.actual_return)))))
    return pd.DataFrame(rows)


def backtest(oos, joined, cost_bps=10):
    """One daily long/cash position. Forecast at 00:05, execute at 01:00 UTC."""
    realized = joined.execution_open.shift(-1)/joined.execution_open-1
    one = oos[oos.horizon == 0].copy()
    curves, stats = [], []
    for name, g in one.groupby("model"):
        if name == "persistence":
            continue
        g = g.sort_values("origin").set_index("origin")
        # A fixed threshold of 20 bps, chosen before evaluation, covers a round trip at base costs.
        desired = g.predicted_return.gt(0.002).astype(float)
        ret = realized.reindex(g.index)
        valid = ret.notna()
        if not valid.all():
            raise ValueError("Execution price missing in an out-of-sample day")
        # Missing test dates would make position carry ambiguous; reject instead of hiding gaps.
        if len(g) > 1 and not (g.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
            raise ValueError("Backtest origins must be contiguous daily observations")
        for bps in sorted(set([0, cost_bps, 25])):
            turnover = desired.diff().abs()
            turnover.iloc[0] = abs(desired.iloc[0])
            net = desired*ret - turnover*bps/10000
            net.iloc[-1] -= desired.iloc[-1]*bps/10000  # close the final holding
            stats.append(dict(model=name, cost_bps=bps, **risk_metrics(net)))
            if bps == cost_bps:
                curves.append(pd.DataFrame(dict(origin=g.index, model=name, position=desired.values,
                    gross_return=(desired*ret).values, net_return=net.values, equity=(1+net).cumprod().values)))
        if name == sorted(one.model.unique()[one.model.unique() != "persistence"])[0]:
            benchmark = ret.copy()
            benchmark.iloc[0] -= cost_bps/10000
            benchmark.iloc[-1] -= cost_bps/10000
            stats.append(dict(model="buy_hold", cost_bps=cost_bps, **risk_metrics(benchmark)))
            curves.append(pd.DataFrame(dict(origin=g.index, model="buy_hold", position=1.0,
                gross_return=ret.values, net_return=benchmark.values, equity=(1+benchmark).cumprod().values)))
    return pd.DataFrame(stats), pd.concat(curves, ignore_index=True)
