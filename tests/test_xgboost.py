"""XGBoost behavior and temporal integration checks on synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from timing.config import PROJECT_ROOT, Paths, load_model_spec
from timing.data.features import CORE_FEATURES
from timing.engine.rolling import run_rolling
from timing.engine.splits import development_rows
from timing.models.xgboost import XGBoostModel


def test_predict_requires_fit(table):
    with pytest.raises(RuntimeError, match="before predicting"):
        XGBoostModel().predict(table[CORE_FEATURES].iloc[:3])


def test_learns_a_nonlinear_return_signal():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"signal": rng.uniform(-1, 1, 600)})
    y = pd.Series(np.where(X["signal"].abs() > 0.5, 0.02, -0.01))
    model = XGBoostModel().fit(X.iloc[:400], y.iloc[:400])
    prediction = model.predict(X.iloc[400:])
    mse = np.mean((prediction - y.iloc[400:].to_numpy()) ** 2)
    baseline = np.mean((y.iloc[:400].mean() - y.iloc[400:].to_numpy()) ** 2)
    assert mse < baseline * 0.2


def test_refit_discards_previous_window(table):
    X, y = table[CORE_FEATURES], table["y"]
    model = XGBoostModel(n_estimators=10, seed=42)
    model.fit(X.iloc[:120], y.iloc[:120])
    model.fit(X.iloc[120:240], y.iloc[120:240])
    fresh = XGBoostModel(n_estimators=10, seed=42).fit(X.iloc[120:240], y.iloc[120:240])
    np.testing.assert_array_equal(model.predict(X.iloc[240:250]), fresh.predict(X.iloc[240:250]))


def test_development_prefix_and_realized_labels(table):
    spec = load_model_spec(Paths(PROJECT_ROOT), "xgboost")
    params = {**spec.configs()[0], "window": 120, "n_estimators": 10}
    development = development_rows(table, {"validation_end": "2004-12-31"})
    full = run_rolling(table, spec, params, 21, "2001-06-01", 42).predictions
    prefix = run_rolling(development, spec, params, 21, "2001-06-01", 42).predictions
    pd.testing.assert_series_equal(prefix["y_hat"], full.loc[prefix.index, "y_hat"])
    for fit_date, train_start, train_end in full[
        ["fit_date", "train_start", "train_end"]
    ].drop_duplicates().itertuples(index=False):
        training = table.loc[train_start:train_end]
        assert len(training) == 120
        assert (training["label_end"] <= fit_date).all()
