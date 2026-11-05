"""Convert forecasts to target SPY weights with a single rule shared by all models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MIN_FORECAST_STD = 1e-12


def forecast_z_score(y_hat: pd.Series, lookback: int) -> pd.Series:
    """Standardize today's forecast using only the previous `lookback` forecasts."""
    history = y_hat.shift(1).rolling(lookback, min_periods=lookback)
    mean = history.mean()
    std = history.std()
    z = (y_hat - mean) / std.where(std > MIN_FORECAST_STD)
    # With no variation in past forecasts there is no signal, so stay neutral.
    return z.where(~(mean.notna() & ~(std > MIN_FORECAST_STD)), 0.0)


def target_weights(y_hat: pd.Series, strategy_cfg: dict[str, Any]) -> pd.Series:
    z = forecast_z_score(y_hat, strategy_cfg["z_lookback"])
    target = strategy_cfg["neutral_weight"] + strategy_cfg["z_scale"] * z
    return target.clip(strategy_cfg["min_weight"], strategy_cfg["max_weight"]).where(z.notna(), np.nan)
