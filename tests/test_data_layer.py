import numpy as np
import pandas as pd
import pytest
from conftest import build_table, make_market

from timing.data.dataset import build_labels
from timing.data.features import CORE_FEATURES, FEATURE_SETS, align_to_calendar
from timing.data.qa import manual_back_adjust


def test_features_do_not_change_when_the_future_is_removed(market):
    full = build_table(market)
    cutoff = market["spy"].index[1800]
    truncated_market = {k: v.loc[:cutoff] for k, v in market.items()}
    truncated = build_table(truncated_market)
    common = truncated.index
    pd.testing.assert_frame_equal(full.loc[common, CORE_FEATURES], truncated.loc[common, CORE_FEATURES])
    # Rows whose label is complete in the truncated data must also match exactly.
    pd.testing.assert_series_equal(full.loc[common, "y"], truncated.loc[common, "y"])


def test_label_uses_next_open_to_following_open():
    dates = pd.bdate_range("2024-01-01", periods=5)  # Monday to Friday.
    open_ = pd.Series([100.0, 101.0, 103.0, 102.0, 104.0], index=dates)
    rate = pd.Series(3.6, index=dates)
    labels = build_labels(open_, rate)
    first = labels.iloc[0]
    assert first["spy_ret"] == pytest.approx(103.0 / 101.0 - 1)
    assert first["rf_ret"] == pytest.approx(0.036 * 1 / 360)
    assert first["y"] == pytest.approx(first["spy_ret"] - first["rf_ret"])
    assert first["entry_date"] == dates[1]
    assert first["label_end"] == dates[2]
    assert labels.iloc[-2:]["spy_ret"].isna().all()


def test_cash_accrues_over_weekends():
    dates = pd.bdate_range("2024-01-03", periods=4)  # Wed, Thu, Fri, Mon.
    labels = build_labels(pd.Series(100.0, index=dates), pd.Series(3.6, index=dates))
    assert labels.loc[dates[1], "rf_ret"] == pytest.approx(0.036 * 3 / 360)


def test_lagged_label_is_known_at_decision_time(table):
    assert (table["label_end"].shift(2).dropna() <= table.index[2:]).all()
    expected = table["y"].shift(2).dropna()
    pd.testing.assert_series_equal(expected, table["b_lag_y"].loc[expected.index], check_names=False)


def test_alignment_uses_last_known_value_only():
    calendar = pd.bdate_range("2024-01-01", periods=4)
    series = pd.Series([1.0, 3.0], index=[calendar[0], calendar[2]])
    aligned = align_to_calendar(series, calendar, limit=5)
    assert aligned.tolist() == [1.0, 1.0, 3.0, 3.0]


def test_alignment_stops_at_the_fill_limit():
    calendar = pd.bdate_range("2024-01-01", periods=4)
    aligned = align_to_calendar(pd.Series([1.0], index=[calendar[0]]), calendar, limit=1)
    assert aligned.iloc[1] == 1.0
    assert np.isnan(aligned.iloc[2])


def test_manual_back_adjustment_applies_factors_to_earlier_dates_only():
    dates = pd.bdate_range("2024-01-01", periods=3)
    close = pd.Series([100.0, 99.0, 100.0], index=dates)
    dividends = pd.Series([0.0, 2.0, 0.0], index=dates)

    appendix = manual_back_adjust(close, dividends)
    assert appendix.iloc[-1] == 100.0
    assert appendix.iloc[1] == 99.0
    assert appendix.iloc[0] == pytest.approx(100.0 / 1.02)

    subtractive = manual_back_adjust(close, dividends, method="subtractive")
    assert subtractive.iloc[1] == 99.0
    assert subtractive.iloc[0] == pytest.approx(100.0 * 0.98)


def test_feature_sets_exist_in_dataset(table):
    for columns in FEATURE_SETS.values():
        assert set(columns) <= set(table.columns)


def test_core_set_has_14_features_and_excludes_labels():
    assert len(CORE_FEATURES) == 14
    assert not {"y", "spy_ret", "rf_ret"} & set(CORE_FEATURES)


def test_dataset_has_no_missing_values():
    table = build_table(make_market(n_days=600))
    assert not table.isna().any().any()
