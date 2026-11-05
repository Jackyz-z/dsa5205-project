"""Forecast and trading metrics. Inputs are backtest frames and forecast series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def sharpe_ratio(net_ret: pd.Series, rf_ret: pd.Series, periods_per_year: int) -> float:
    excess = net_ret - rf_ret
    std = excess.std()
    if not std > 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def annualized_return(net_ret: pd.Series, periods_per_year: int) -> float:
    return float((1 + net_ret).prod() ** (periods_per_year / len(net_ret)) - 1)


def max_drawdown(net_ret: pd.Series) -> float:
    wealth = (1 + net_ret).cumprod()
    # Include the starting value so a loss in the first period counts as a drawdown.
    peak = wealth.cummax().clip(lower=1.0)
    return float((wealth / peak - 1).min())


def capm_alpha(net_ret: pd.Series, spy_ret: pd.Series, rf_ret: pd.Series, periods_per_year: int, lags: int) -> dict[str, float]:
    """Regress strategy excess returns on SPY excess returns with Newey-West errors."""
    excess = (net_ret - rf_ret).to_numpy()
    market = sm.add_constant((spy_ret - rf_ret).to_numpy())
    fit = sm.OLS(excess, market).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "alpha_annual": float(fit.params[0] * periods_per_year),
        "alpha_t": float(fit.tvalues[0]),
        "beta": float(fit.params[1]),
    }


def trading_metrics(backtest: pd.DataFrame, periods_per_year: int, lags: int) -> dict[str, float]:
    net = backtest["net_ret"]
    rf = backtest["rf_ret"]
    sharpe = sharpe_ratio(net, rf, periods_per_year)
    daily_sharpe = sharpe / np.sqrt(periods_per_year)
    metrics = {
        "ann_return": annualized_return(net, periods_per_year),
        "ann_vol": float(net.std() * np.sqrt(periods_per_year)),
        "sharpe": sharpe,
        "sharpe_se": float(np.sqrt((1 + 0.5 * daily_sharpe**2) / len(net)) * np.sqrt(periods_per_year)),
        "max_drawdown": max_drawdown(net),
        "ann_turnover": float(backtest["trade"].mean() * periods_per_year),
        "avg_weight": float(backtest["weight"].mean()),
        "days_traded_pct": float((backtest["trade"] > 0).mean()),
        "n_days": float(len(net)),
    }
    if backtest["weight"].std() > 0:
        metrics.update(capm_alpha(net, backtest["spy_ret"], rf, periods_per_year, lags))
    else:
        metrics.update({"alpha_annual": float("nan"), "alpha_t": float("nan"), "beta": float("nan")})
    return metrics


def oos_r2(y_true: pd.Series, y_hat: pd.Series, y_benchmark: pd.Series) -> float:
    """Campbell-Thompson out-of-sample R^2 relative to a benchmark forecast."""
    sse = ((y_true - y_hat) ** 2).sum()
    sse_benchmark = ((y_true - y_benchmark) ** 2).sum()
    return float(1 - sse / sse_benchmark)


def cumulative_oos_r2(y_true: pd.Series, y_hat: pd.Series, y_benchmark: pd.Series) -> pd.Series:
    return 1 - ((y_true - y_hat) ** 2).cumsum() / ((y_true - y_benchmark) ** 2).cumsum()


def forecast_metrics(y_true: pd.Series, y_hat: pd.Series, y_benchmark: pd.Series) -> dict[str, float]:
    return {
        "oos_r2_vs_prevailing_mean": oos_r2(y_true, y_hat, y_benchmark),
        "oos_r2_vs_zero": oos_r2(y_true, y_hat, pd.Series(0.0, index=y_true.index)),
        "direction_accuracy": float((np.sign(y_hat) == np.sign(y_true)).mean()),
    }
