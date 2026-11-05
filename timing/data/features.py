"""Feature definitions. Every value on row t uses data available at the close of day t."""

from __future__ import annotations

import numpy as np
import pandas as pd

CORE_FEATURES = [
    "f_overnight",
    "f_intraday",
    "f_ret_5",
    "f_ret_21",
    "f_ret_63",
    "f_mom_252",
    "f_ma200_gap",
    "f_rv_21",
    "f_volume_ratio",
    "f_vix_log",
    "f_vix_chg_5",
    "f_vix_minus_rv",
    "f_term_spread",
    "f_long_rate_chg_5",
]
AR_FEATURES = ["b_lag_y"]

FEATURE_SETS = {"core": CORE_FEATURES, "ar": AR_FEATURES}


def align_to_calendar(series: pd.Series, calendar: pd.DatetimeIndex, limit: int) -> pd.Series:
    """Align a series to SPY trading days using only the last value known on each day."""
    union = series.index.union(calendar)
    return series.reindex(union).ffill(limit=limit).reindex(calendar)


def build_features(
    spy: pd.DataFrame,
    vix_close: pd.Series,
    long_rate_close: pd.Series,
    short_rate_close: pd.Series,
) -> pd.DataFrame:
    """Compute the 14 core features on the SPY calendar. Inputs must already be aligned."""
    close = spy["Close"]
    open_ = spy["Open"]
    volume = spy["Volume"]

    rv_21 = close.pct_change().rolling(21).std() * np.sqrt(252)

    features = pd.DataFrame(index=spy.index)
    features["f_overnight"] = open_ / close.shift(1) - 1
    features["f_intraday"] = close / open_ - 1
    features["f_ret_5"] = close.pct_change(5)
    features["f_ret_21"] = close.pct_change(21)
    features["f_ret_63"] = close.pct_change(63)
    features["f_mom_252"] = close.pct_change(252)
    features["f_ma200_gap"] = close / close.rolling(200).mean() - 1
    features["f_rv_21"] = rv_21
    features["f_volume_ratio"] = volume.rolling(5).mean() / volume.rolling(63).mean()
    features["f_vix_log"] = np.log(vix_close)
    features["f_vix_chg_5"] = np.log(vix_close).diff(5)
    features["f_vix_minus_rv"] = vix_close / 100 - rv_21
    features["f_term_spread"] = long_rate_close - short_rate_close
    features["f_long_rate_chg_5"] = long_rate_close.diff(5)
    return features[CORE_FEATURES]
