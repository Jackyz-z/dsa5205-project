"""Synthetic market data, so tests never need the network or real test-period results."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from timing.config import PROJECT_ROOT, Paths, load_base_config, write_json
from timing.data.dataset import assemble_dataset


def make_market(n_days: int = 2200, seed: int = 7, start: str = "2000-01-03") -> dict[str, pd.DataFrame | pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    overnight = rng.normal(0.0002, 0.004, n_days)
    intraday = rng.normal(0.0002, 0.008, n_days)
    closes, opens = [], []
    previous_close = 100.0
    for gap, move in zip(overnight, intraday):
        day_open = previous_close * (1 + gap)
        day_close = day_open * (1 + move)
        opens.append(day_open)
        closes.append(day_close)
        previous_close = day_close
    spy = pd.DataFrame(
        {
            "Open": opens,
            "Close": closes,
            "High": np.maximum(opens, closes) * 1.002,
            "Low": np.minimum(opens, closes) * 0.998,
            "Volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
        },
        index=dates,
    )
    vix = pd.Series(np.exp(rng.normal(2.9, 0.2, n_days)), index=dates)
    long_rate = pd.Series(4 + np.cumsum(rng.normal(0, 0.02, n_days)), index=dates)
    short_rate = pd.Series(2 + np.cumsum(rng.normal(0, 0.01, n_days)), index=dates).clip(lower=-0.1)
    return {"spy": spy, "vix": vix, "long_rate": long_rate, "short_rate": short_rate}


def build_table(market: dict) -> pd.DataFrame:
    table, _ = assemble_dataset(market["spy"], market["vix"], market["long_rate"], market["short_rate"], 5)
    return table


@pytest.fixture
def market() -> dict:
    return make_market()


@pytest.fixture
def table(market) -> pd.DataFrame:
    return build_table(market)


@pytest.fixture
def sandbox(tmp_path):
    """A project copy with short windows and dates that suit the synthetic sample."""
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    paths = Paths(tmp_path)
    cfg = load_base_config(paths)
    cfg["splits"] = {
        "prediction_start": "2001-06-01",
        "validation_start": "2003-02-03",
        "validation_end": "2004-12-31",
        "test_start": "2005-01-03",
        "test_end": "2008-06-30",
    }
    cfg["evaluation"]["subperiods"] = {
        "validation": [["2003-02-03", "2003-12-31"], ["2004-01-01", "2004-12-31"]],
        "test": [["2005-01-03", "2006-12-31"], ["2007-01-01", "2008-06-30"]],
    }
    with open(paths.configs / "base.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    for model_path in (paths.configs / "models").glob("*.yaml"):
        with open(model_path, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        spec["grid"]["window"] = [60, 252]
        with open(model_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(spec, f)

    dataset = build_table(make_market())
    paths.dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(paths.dataset)
    write_json(paths.manifest, {"data_hash": "synthetic"})
    return paths, cfg, dataset
