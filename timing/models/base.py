"""The interface every model implements. Models know nothing about costs or positions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """A forecaster of the next holding-period excess return.

    The rolling engine passes a training window that excludes unrealized labels, already
    preprocessed according to the model's YAML config. A model that needs internal
    validation (for example, early stopping) must split `X` and `y` in time order.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit on one training window and return self."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return a one-dimensional array with one forecast per row of X."""

    def diagnostics(self) -> dict[str, float] | None:
        """Optional values to record after each fit, such as coefficients."""
        return None
