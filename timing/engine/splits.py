"""Time splits. A row belongs to a split only if its whole holding period fits inside it."""

from __future__ import annotations

from typing import Any

import pandas as pd


def split_index(dataset: pd.DataFrame, split: str, splits_cfg: dict[str, Any]) -> pd.DatetimeIndex:
    """Return the decision dates evaluated in the 'validation' or 'test' split."""
    start = pd.Timestamp(splits_cfg[f"{split}_start"])
    end = pd.Timestamp(splits_cfg[f"{split}_end"])
    mask = (dataset.index >= start) & (dataset["label_end"] <= end)
    return dataset.index[mask]


def development_rows(dataset: pd.DataFrame, splits_cfg: dict[str, Any]) -> pd.DataFrame:
    """Drop every row whose label is not known by the end of validation."""
    return dataset[dataset["label_end"] <= pd.Timestamp(splits_cfg["validation_end"])]
