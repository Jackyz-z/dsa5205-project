"""Assemble the modelling table. This file defines the data layer's public contract.

Each row is a decision date t (the close of day t):
    f_*        features known at the close of t
    b_lag_y    open(t-1) -> open(t) excess return, known at the close of t
    spy_ret    open(t+1) -> open(t+2) SPY return
    rf_ret     cash return over the same holding period
    y          spy_ret - rf_ret
    label_end  t+2, the date on which the label becomes known
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from timing.config import Paths, stable_hash, write_json
from timing.data.download import load_raw
from timing.data.features import AR_FEATURES, CORE_FEATURES, align_to_calendar, build_features

LABEL_COLUMNS = ["spy_ret", "rf_ret", "y", "entry_date", "label_end"]


def build_labels(open_: pd.Series, short_rate_close: pd.Series) -> pd.DataFrame:
    """Build next-open to following-open labels for every decision date."""
    dates = pd.Series(open_.index, index=open_.index)
    entry_date = dates.shift(-1)
    label_end = dates.shift(-2)
    spy_ret = open_.shift(-2) / open_.shift(-1) - 1
    # ^IRX is quoted in annual percent. The rate known at t accrues over the holding period.
    held_days = (label_end - entry_date).dt.days
    rf_ret = short_rate_close / 100 * held_days / 360
    return pd.DataFrame(
        {
            "spy_ret": spy_ret,
            "rf_ret": rf_ret,
            "y": spy_ret - rf_ret,
            "entry_date": entry_date,
            "label_end": label_end,
        }
    )


def assemble_dataset(
    spy: pd.DataFrame,
    vix_close: pd.Series,
    long_rate_close: pd.Series,
    short_rate_close: pd.Series,
    max_ffill_days: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Combine features and labels on the SPY calendar. Returns the table and fill records."""
    calendar = spy.index
    filled: dict[str, list[str]] = {}
    aligned = {}
    for name, series in (
        ("vix", vix_close),
        ("long_rate", long_rate_close),
        ("short_rate", short_rate_close),
    ):
        missing = calendar.difference(series.dropna().index)
        filled[name] = [d.date().isoformat() for d in missing]
        aligned[name] = align_to_calendar(series, calendar, max_ffill_days)

    features = build_features(spy, aligned["vix"], aligned["long_rate"], aligned["short_rate"])
    labels = build_labels(spy["Open"], aligned["short_rate"])
    table = features.join(labels)
    table["b_lag_y"] = table["y"].shift(2)
    table = table[CORE_FEATURES + AR_FEATURES + LABEL_COLUMNS].dropna()
    table.index.name = "date"
    return table, filled


def build_and_save(cfg: dict[str, Any], paths: Paths) -> pd.DataFrame:
    tickers = cfg["data"]["tickers"]
    table, filled = assemble_dataset(
        load_raw(paths.raw, tickers["spy"]),
        load_raw(paths.raw, tickers["vix"])["Close"],
        load_raw(paths.raw, tickers["long_rate"])["Close"],
        load_raw(paths.raw, tickers["short_rate"])["Close"],
        cfg["data"]["max_ffill_days"],
    )
    paths.dataset.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.dataset, float_format="%.12g")
    manifest = {
        "rows": len(table),
        "first_decision_date": table.index.min().date().isoformat(),
        "last_decision_date": table.index.max().date().isoformat(),
        "data_hash": stable_hash(table.to_csv(float_format="%.10g", lineterminator="\n")),
        "forward_filled_dates": filled,
    }
    write_json(paths.manifest, manifest)
    return table


def load_dataset(paths: Paths) -> pd.DataFrame:
    if not paths.dataset.exists():
        raise FileNotFoundError(f"{paths.dataset} not found. Run: python -m timing.cli data")
    return pd.read_csv(
        paths.dataset,
        index_col="date",
        parse_dates=["date", "entry_date", "label_end"],
    )
