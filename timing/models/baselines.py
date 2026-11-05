"""Naive forecast baselines, run through the same pipeline as every other model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from timing.models.base import BaseModel
from timing.models.registry import register


@register("prevailing_mean")
class PrevailingMeanModel(BaseModel):
    """Forecasts the training window's average excess return."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PrevailingMeanModel":
        self._mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)


@register("ar1")
class AR1Model(BaseModel):
    """OLS of the label on its most recent realized value, `b_lag_y`."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._intercept = 0.0
        self._slope = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AR1Model":
        design = np.column_stack([np.ones(len(X)), X["b_lag_y"].to_numpy()])
        (self._intercept, self._slope), *_ = np.linalg.lstsq(design, y.to_numpy(), rcond=None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._intercept + self._slope * X["b_lag_y"].to_numpy()
