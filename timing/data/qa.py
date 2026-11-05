"""Data-quality checks, including the Appendix B adjustment verification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from timing.config import Paths, write_json
from timing.data.download import load_raw

STOOQ_URL = "https://stooq.com/q/d/l/?s=spy.us&i=d"


def manual_back_adjust(close: pd.Series, dividends: pd.Series, method: str = "appendix") -> pd.Series:
    """Appendix B.2: A_t = P_t * product of g_k over k > t, using dividend factors.

    Yahoo's unadjusted Close is already split-adjusted, so only dividends remain.
    The "appendix" factor is 1 / (1 + D / P_prev), as printed in the project brief. The
    "subtractive" factor is 1 - D / P_prev, the common vendor convention. Both only
    approximate the total return (P + D) / P_prev, which is why QA reports both gaps.
    """
    factor = pd.Series(1.0, index=close.index)
    has_dividend = dividends > 0
    ratio = dividends[has_dividend] / close.shift(1)[has_dividend]
    factor[has_dividend] = 1 / (1 + ratio) if method == "appendix" else 1 - ratio
    future_product = factor[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return close * future_product


def check_price_integrity(frame: pd.DataFrame) -> dict[str, Any]:
    open_equals_close = frame.index[frame["Open"] == frame["Close"]]
    high_ok = frame["High"] >= frame[["Open", "Close"]].max(axis=1) * (1 - 1e-6)
    low_ok = frame["Low"] <= frame[["Open", "Close"]].min(axis=1) * (1 + 1e-6)
    return {
        "rows": len(frame),
        "first_date": frame.index.min().date().isoformat(),
        "last_date": frame.index.max().date().isoformat(),
        "is_sorted_unique": bool(frame.index.is_monotonic_increasing and frame.index.is_unique),
        "nan_counts": {k: int(v) for k, v in frame.isna().sum().items()},
        "non_positive_open_or_close": int(((frame["Open"] <= 0) | (frame["Close"] <= 0)).sum()),
        "high_low_violations": int((~high_ok).sum() + (~low_ok).sum()),
        "zero_volume_days": int((frame["Volume"] == 0).sum()),
        "open_equals_close_days": len(open_equals_close),
        "open_equals_close_by_year": {
            str(year): int(count)
            for year, count in pd.Series(1, index=open_equals_close)
            .groupby(open_equals_close.year)
            .sum()
            .items()
        },
    }


def check_adjustments(raw: pd.DataFrame, adjusted: pd.DataFrame) -> dict[str, Any]:
    """Appendix B.4: return continuity plus one representative dividend event."""
    vendor = raw["Adj Close"].pct_change()
    gap = (manual_back_adjust(raw["Close"], raw["Dividends"]).pct_change() - vendor).abs().dropna()
    subtractive = manual_back_adjust(raw["Close"], raw["Dividends"], method="subtractive")
    subtractive_gap = (subtractive.pct_change() - vendor).abs().dropna()
    auto_gap = (adjusted["Close"].pct_change() - vendor).abs().dropna()

    event_date = raw.index[raw["Dividends"] > 0][-1]
    pos = raw.index.get_loc(event_date)
    prev_close = raw["Close"].iloc[pos - 1]
    formula_tr = (raw["Close"].iloc[pos] + raw["Dividends"].iloc[pos]) / prev_close - 1
    vendor_tr = raw["Adj Close"].iloc[pos] / raw["Adj Close"].iloc[pos - 1] - 1
    return {
        "manual_vs_vendor_mean_abs_return_gap": float(gap.mean()),
        "manual_vs_vendor_max_abs_return_gap": float(gap.max()),
        "subtractive_vs_vendor_mean_abs_return_gap": float(subtractive_gap.mean()),
        "subtractive_vs_vendor_max_abs_return_gap": float(subtractive_gap.max()),
        "auto_adjust_vs_adj_close_mean_abs_return_gap": float(auto_gap.mean()),
        "spy_split_events": int((raw["Stock Splits"] > 0).sum()),
        "dividend_events": int((raw["Dividends"] > 0).sum()),
        "dividend_event": {
            "date": event_date.date().isoformat(),
            "dividend": float(raw["Dividends"].iloc[pos]),
            "total_return_formula": float(formula_tr),
            "total_return_vendor": float(vendor_tr),
        },
    }


def check_split_event(raw: pd.DataFrame, split_date: str) -> dict[str, Any]:
    """Confirm the split lands on the expected date without a discontinuity in Close."""
    date = pd.Timestamp(split_date)
    pos = raw.index.get_loc(date)
    return {
        "date": split_date,
        "split_ratio": float(raw["Stock Splits"].iloc[pos]),
        "close_return_on_split_date": float(raw["Close"].iloc[pos] / raw["Close"].iloc[pos - 1] - 1),
    }


def check_against_stooq(adjusted: pd.DataFrame) -> dict[str, Any]:
    """Optional cross-check. A Stooq failure is recorded and never stops the pipeline."""
    try:
        stooq = pd.read_csv(STOOQ_URL, index_col="Date", parse_dates=["Date"])
    except Exception as exc:  # Any network or parsing failure means the check is skipped.
        return {"status": "skipped", "reason": repr(exc)}
    both = adjusted[["Close"]].join(stooq[["Close"]], rsuffix="_stooq", how="inner")
    gap = (both["Close"].pct_change() - both["Close_stooq"].pct_change()).abs().dropna()
    return {
        "status": "ok",
        "overlap_days": len(both),
        "mean_abs_return_gap": float(gap.mean()),
    }


def run_qa(cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    tickers = cfg["data"]["tickers"]
    spy_adjusted = load_raw(paths.raw, tickers["spy"], adjusted=True)
    spy_raw = load_raw(paths.raw, tickers["spy"], adjusted=False)
    split_raw = load_raw(paths.raw, cfg["data"]["split_check_ticker"], adjusted=False)
    report = {
        "spy": check_price_integrity(spy_adjusted),
        "adjustments": check_adjustments(spy_raw, spy_adjusted),
        "split_check": {
            "ticker": cfg["data"]["split_check_ticker"],
            **check_split_event(split_raw, cfg["data"]["split_check_date"]),
        },
        "rate_ranges": {
            name: {
                "min": float(np.min(load_raw(paths.raw, tickers[name])["Close"])),
                "max": float(np.max(load_raw(paths.raw, tickers[name])["Close"])),
            }
            for name in ("vix", "long_rate", "short_rate")
        },
        "stooq": check_against_stooq(spy_adjusted),
    }
    write_json(paths.qa_report, report)

    spy = report["spy"]
    if not spy["is_sorted_unique"] or spy["non_positive_open_or_close"] or any(spy["nan_counts"].values()):
        raise ValueError(f"SPY failed integrity checks: {spy}")
    return report
