"""Random forest forecasts using the shared, time-filtered training window."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from timing.models.base import BaseModel
from timing.models.registry import register


@register("random_forest")
class RandomForestModel(BaseModel):
    """Average bootstrapped regression trees; depth and leaf size limit complexity.

    Bootstrap samples come only from the window supplied by the rolling engine.
    Out-of-bag scores are not used for time-series model selection. Use the shared
    forward validation instead. Single-worker prediction keeps summation order
    deterministic for the repository's exact reproducibility contract.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 20,
                 n_estimators: int = 200, max_features: float = 1.0,
                 seed: int = 0) -> None:
        super().__init__(seed)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.max_features = max_features
        self._model: RandomForestRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
            bootstrap=True, oob_score=False, random_state=self.seed, n_jobs=1,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Fit RandomForestModel before predicting.")
        return np.asarray(self._model.predict(X), dtype=float)

    # Keep the default diagnostics=None: the shared reporting code labels generic
    # diagnostics as regression coefficients, which tree importances are not.
