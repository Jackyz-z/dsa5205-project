"""LASSO regression for sparse return prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from timing.models.base import BaseModel
from timing.models.registry import register


@register("lasso")
class LassoModel(BaseModel):
    """Linear regression with an L1 penalty that can set coefficients to zero.

    With standardized features, ``alpha`` controls the common shrinkage scale across
    predictors.  A sparse solution matches the hypothesis that only a small subset of
    the candidate market signals is useful in any rolling estimation window.
    """

    def __init__(self, alpha: float, seed: int = 0) -> None:
        super().__init__(seed)
        self.alpha = alpha
        self._model: Lasso | None = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoModel":
        self._columns = list(X.columns)
        self._model = Lasso(
            alpha=self.alpha,
            fit_intercept=True,
            max_iter=100_000,
            selection="cyclic",
        )
        self._model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.to_numpy())

    def diagnostics(self) -> dict[str, float]:
        coefficients = dict(zip(self._columns, map(float, self._model.coef_)))
        coefficients["intercept"] = float(self._model.intercept_)
        coefficients["n_nonzero"] = float(np.count_nonzero(self._model.coef_))
        return coefficients
