"""Download raw series from Yahoo Finance and cache them as CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


def _safe_name(ticker: str) -> str:
    return ticker.replace("^", "")


def raw_path(raw_dir: Path, ticker: str, adjusted: bool) -> Path:
    suffix = "adjusted" if adjusted else "raw"
    return raw_dir / f"{_safe_name(ticker)}_{suffix}.csv"


def _download(ticker: str, start: str, end: str, adjusted: bool) -> pd.DataFrame:
    frame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=adjusted,
        actions=not adjusted,
        progress=False,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}")
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"
    return frame.sort_index()


def download_all(cfg: dict[str, Any], raw_dir: Path, refresh: bool = False) -> None:
    """Fetch every required series, skipping files that are already cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg["data"]
    requests = [(ticker, True) for ticker in data_cfg["tickers"].values()]
    requests.append((data_cfg["tickers"]["spy"], False))
    requests.append((data_cfg["split_check_ticker"], False))
    for ticker, adjusted in requests:
        path = raw_path(raw_dir, ticker, adjusted)
        if path.exists() and not refresh:
            print(f"cached   {path.name}")
            continue
        frame = _download(ticker, data_cfg["start"], data_cfg["end"], adjusted)
        frame.to_csv(path)
        print(f"fetched  {path.name}: {frame.index.min().date()} to {frame.index.max().date()}")


def load_raw(raw_dir: Path, ticker: str, adjusted: bool = True) -> pd.DataFrame:
    return pd.read_csv(raw_path(raw_dir, ticker, adjusted), index_col="date", parse_dates=["date"])
