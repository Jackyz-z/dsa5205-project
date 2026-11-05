"""Feature preprocessing, fitted on the training window only."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLIP_Z = 5.0


class Preprocessor:
    """Either passes features through or standardizes and clips them at +/- CLIP_Z."""

    def __init__(self, kind: str) -> None:
        if kind not in {"none", "standard"}:
            raise ValueError(f"Unknown preprocess option {kind!r}")
        self.kind = kind
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        if self.kind == "standard":
            self._mean = X.mean()
            self._std = X.std(ddof=0).replace(0.0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.kind == "none":
            return X
        return ((X - self._mean) / self._std).clip(-CLIP_Z, CLIP_Z)


def has_missing(X: pd.DataFrame) -> bool:
    return bool(np.isnan(X.to_numpy()).any())
