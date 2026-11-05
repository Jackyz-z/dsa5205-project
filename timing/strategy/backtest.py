"""Daily backtest. Each row covers the holding period that follows decision date t."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timing.strategy.positions import target_weights


def simulate(target: pd.Series, returns: pd.DataFrame, band: float, cost_bps: float) -> pd.DataFrame:
    """Trade at the next open when the target differs from the drifted weight by more than `band`.

    `returns` must hold spy_ret and rf_ret on the same dates. Rows with no target (the
    warm-up) are dropped. The first position is bought from cash and pays its cost.
    """
    target = target.dropna()
    spy_ret = returns.loc[target.index, "spy_ret"].to_numpy()
    rf_ret = returns.loc[target.index, "rf_ret"].to_numpy()
    n = len(target)
    weight = np.empty(n)
    trade = np.empty(n)
    gross = np.empty(n)

    held = 0.0
    for i, desired in enumerate(target.to_numpy()):
        if i == 0 or abs(desired - held) > band:
            weight[i] = desired
            trade[i] = abs(desired - held)
        else:
            weight[i] = held
            trade[i] = 0.0
        gross[i] = weight[i] * spy_ret[i] + (1 - weight[i]) * rf_ret[i]
        held = weight[i] * (1 + spy_ret[i]) / (1 + gross[i])

    cost = trade * cost_bps / 10_000
    return pd.DataFrame(
        {
            "target": target.to_numpy(),
            "weight": weight,
            "trade": trade,
            "cost": cost,
            "gross_ret": gross,
            "net_ret": gross - cost,
            "spy_ret": spy_ret,
            "rf_ret": rf_ret,
        },
        index=target.index,
    )


def backtest_forecasts(
    y_hat: pd.Series,
    returns: pd.DataFrame,
    strategy_cfg: dict[str, Any],
    cost_bps: float,
) -> pd.DataFrame:
    target = target_weights(y_hat, strategy_cfg)
    return simulate(target, returns, strategy_cfg["no_trade_band"], cost_bps)


def buy_and_hold(returns: pd.DataFrame) -> pd.DataFrame:
    """SPY held throughout; dividends are captured because prices are adjusted."""
    return pd.DataFrame(
        {
            "target": 1.0,
            "weight": 1.0,
            "trade": 0.0,
            "cost": 0.0,
            "gross_ret": returns["spy_ret"],
            "net_ret": returns["spy_ret"],
            "spy_ret": returns["spy_ret"],
            "rf_ret": returns["rf_ret"],
        },
        index=returns.index,
    )
