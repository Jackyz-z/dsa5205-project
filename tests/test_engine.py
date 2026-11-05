import numpy as np
import pandas as pd
import pytest

from timing.config import ModelSpec
from timing.data.features import CORE_FEATURES
from timing.engine.preprocess import Preprocessor
from timing.engine.rolling import run_rolling
from timing.engine.splits import development_rows, split_index

SPEC = ModelSpec("ridge", "candidate", "core", "standard", {"window": [120], "lam": [1.0]})
PARAMS = {"window": 120, "lam": 1.0}
START = "2001-06-01"


def run(frame):
    return run_rolling(frame, SPEC, PARAMS, refit_every=21, prediction_start=START, seed=0).predictions


def test_training_labels_are_realized_at_each_fit(table):
    predictions = run(table)
    for fit_date, train_end in predictions[["fit_date", "train_end"]].drop_duplicates().itertuples(index=False):
        assert table.loc[train_end, "label_end"] <= fit_date
    # The two decision dates before a fit have unrealized labels, so the gap is exact.
    fit_date, train_end = predictions[["fit_date", "train_end"]].iloc[0]
    positions = table.index
    assert positions.get_loc(fit_date) - positions.get_loc(train_end) == 2


def test_changing_the_future_does_not_change_past_forecasts(table):
    baseline = run(table)
    cutoff_pos = table.index.get_loc(baseline.index[300])
    fit_date = table.index[cutoff_pos]
    tampered = table.copy()
    future = tampered.index > fit_date
    # Labels of the last two decision dates are also unknown at the cutoff.
    unrealized = tampered["label_end"] > fit_date
    tampered.loc[unrealized, ["y", "spy_ret"]] = 99.0
    tampered.loc[future, CORE_FEATURES] = -99.0
    changed = run(tampered)
    pd.testing.assert_series_equal(baseline.loc[:fit_date, "y_hat"], changed.loc[:fit_date, "y_hat"])


def test_each_forecast_uses_a_fit_from_on_or_before_its_date(table):
    predictions = run(table)
    assert (predictions["fit_date"] <= predictions.index).all()
    assert predictions["fit_date"].nunique() > 1


def test_training_window_has_the_requested_length(table):
    predictions = run(table)
    row = predictions.iloc[0]
    length = table.index.get_loc(row["train_end"]) - table.index.get_loc(row["train_start"]) + 1
    assert length == PARAMS["window"]


def test_no_forecast_before_a_full_window_exists(table):
    spec = ModelSpec("ridge", "candidate", "core", "standard", {"window": [5000], "lam": [1.0]})
    with pytest.raises(ValueError, match="enough realized labels"):
        run_rolling(table, spec, {"window": 5000, "lam": 1.0}, 21, START, 0)


def test_development_forecasts_equal_the_prefix_of_the_full_run(table):
    splits = {"validation_end": "2004-12-31"}
    development = development_rows(table, splits)
    short = run(development)
    full = run(table)
    pd.testing.assert_series_equal(short["y_hat"], full.loc[short.index, "y_hat"])


def test_splits_do_not_overlap_in_holding_periods(table):
    splits = {
        "validation_start": "2002-01-01",
        "validation_end": "2004-12-31",
        "test_start": "2005-01-03",
        "test_end": "2008-06-30",
    }
    validation = split_index(table, "validation", splits)
    test = split_index(table, "test", splits)
    assert table.loc[validation, "label_end"].max() <= pd.Timestamp("2004-12-31")
    assert test.min() >= pd.Timestamp("2005-01-03")
    assert table.loc[test, "entry_date"].min() > table.loc[validation, "label_end"].max()


def test_development_rows_exclude_labels_after_validation(table):
    development = development_rows(table, {"validation_end": "2004-12-31"})
    assert development["label_end"].max() <= pd.Timestamp("2004-12-31")


def test_preprocessor_is_fitted_on_training_data_only():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    other = pd.DataFrame({"a": [1000.0]})
    processor = Preprocessor("standard").fit(train)
    assert processor.transform(train)["a"].mean() == pytest.approx(0.0)
    assert processor.transform(other)["a"].iloc[0] == 5.0


def test_preprocessor_handles_constant_columns():
    processor = Preprocessor("standard").fit(pd.DataFrame({"a": [2.0, 2.0]}))
    assert np.isfinite(processor.transform(pd.DataFrame({"a": [3.0]}))["a"].iloc[0])
