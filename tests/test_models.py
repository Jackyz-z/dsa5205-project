"""Contract checks that run automatically for every registered model with a YAML config."""

import numpy as np
import pandas as pd
import pytest

import timing.models  # noqa: F401
from timing.config import PROJECT_ROOT, Paths, list_model_names, load_model_spec
from timing.data.features import CORE_FEATURES, FEATURE_SETS
from timing.engine.preprocess import Preprocessor
from timing.engine.rolling import run_rolling
from timing.models.registry import build_model, register, registered_names
from timing.models.ridge import RidgeModel

PATHS = Paths(PROJECT_ROOT)
SHORT_WINDOW = 120


def short_params(spec):
    params = dict(spec.configs()[0])
    params["window"] = SHORT_WINDOW
    return params


@pytest.fixture(params=list_model_names(PATHS))
def spec(request):
    return load_model_spec(PATHS, request.param)


def fit_predict(spec, table):
    params = {k: v for k, v in short_params(spec).items() if k != "window"}
    columns = FEATURE_SETS[spec.feature_set]
    train, test = table.iloc[:SHORT_WINDOW], table.iloc[SHORT_WINDOW:SHORT_WINDOW + 25]
    processor = Preprocessor(spec.preprocess).fit(train[columns])
    model = build_model(spec.name, params, seed=1)
    returned = model.fit(processor.transform(train[columns]), train["y"])
    return returned, model, model.predict(processor.transform(test[columns]))


def test_every_config_has_a_registered_model(spec):
    assert spec.name in registered_names()


def test_fit_returns_self_and_predict_returns_a_flat_array(spec, table):
    returned, model, forecasts = fit_predict(spec, table)
    assert returned is model
    assert isinstance(forecasts, np.ndarray)
    assert forecasts.shape == (25,)
    assert np.isfinite(forecasts).all()


def test_same_seed_gives_identical_forecasts(spec, table):
    _, _, first = fit_predict(spec, table)
    _, _, second = fit_predict(spec, table)
    np.testing.assert_array_equal(first, second)


def test_model_forecasts_ignore_the_future(spec, table):
    start = "2001-06-01"
    params = short_params(spec)
    baseline = run_rolling(table, spec, params, 21, start, 0).predictions
    cutoff = baseline.index[200]
    tampered = table.copy()
    tampered.loc[tampered["label_end"] > cutoff, ["y", "spy_ret"]] = 99.0
    future = tampered.index > cutoff
    columns = FEATURE_SETS[spec.feature_set]
    tampered.loc[future, columns] = -99.0
    changed = run_rolling(tampered, spec, params, 21, start, 0).predictions
    pd.testing.assert_series_equal(baseline.loc[:cutoff, "y_hat"], changed.loc[:cutoff, "y_hat"])


def test_ridge_matches_the_closed_form():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    y = pd.Series(X["a"] * 0.5 + rng.normal(size=100))
    lam = 0.7
    model = RidgeModel(lam=lam).fit(X, y)

    centered_X = X - X.mean()
    centered_y = y - y.mean()
    n = len(X)
    beta = np.linalg.solve(centered_X.T @ centered_X / n + lam * np.eye(3), centered_X.T @ centered_y / n)
    assert list(model.diagnostics().values())[:3] == pytest.approx(list(beta))


def test_large_ridge_penalty_shrinks_to_the_prevailing_mean(table):
    X, y = table[CORE_FEATURES].iloc[:200], table["y"].iloc[:200]
    processor = Preprocessor("standard").fit(X)
    forecasts = RidgeModel(lam=1e8).fit(processor.transform(X), y).predict(processor.transform(X))
    assert forecasts == pytest.approx(np.full(200, y.mean()), abs=1e-8)


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register("ridge")(RidgeModel)
