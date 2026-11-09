"""Boosted regression trees fitted only on the shared realized-label window."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from timing.models.base import BaseModel
from timing.models.registry import register


@register("xgboost")
class XGBoostModel(BaseModel):
    """Predict excess returns with a fixed number of squared-error boosting rounds.

    Hyperparameters are selected by the shared forward validation. No shuffled
    cross-validation, evaluation set, early stopping, or warm start is used.
    The initial forecast is the current training window's mean return.
    """

    def __init__(self, max_depth: int = 2, learning_rate: float = 0.03,
                 n_estimators: int = 200, min_child_weight: float = 20.0,
                 reg_lambda: float = 1.0, seed: int = 0) -> None:
        super().__init__(seed)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self._model: XGBRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self._model = XGBRegressor(
            objective="reg:squarederror", booster="gbtree", tree_method="hist",
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda, reg_alpha=0.0, gamma=0.0,
            subsample=1.0, colsample_bytree=1.0, base_score=float(y.mean()),
            random_state=self.seed, n_jobs=1, verbosity=0,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Fit XGBoostModel before predicting.")
        return np.asarray(self._model.predict(X), dtype=float)

    # Tree importance is not a regression coefficient; keep diagnostics=None.
