"""Ridge regression with a penalty normalized by the window length."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from timing.models.base import BaseModel
from timing.models.registry import register


@register("ridge")
class RidgeModel(BaseModel):
    """Minimizes (1/T) * sum((y - a - X b)^2) + lam * ||b||^2 in closed form.

    The intercept is not penalized, so a large `lam` shrinks the forecast toward the
    window's average return, which is the prevailing-mean baseline.
    """

    def __init__(self, lam: float, seed: int = 0) -> None:
        super().__init__(seed)
        self.lam = lam
        self._model: Ridge | None = None
        self._columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        self._columns = list(X.columns)
        self._model = Ridge(alpha=self.lam * len(X), fit_intercept=True, solver="cholesky")
        self._model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.to_numpy())

    def diagnostics(self) -> dict[str, float]:
        coefficients = dict(zip(self._columns, map(float, self._model.coef_)))
        coefficients["intercept"] = float(self._model.intercept_)
        return coefficients
