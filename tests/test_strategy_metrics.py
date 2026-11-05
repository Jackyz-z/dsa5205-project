import numpy as np
import pandas as pd
import pytest

from timing.evaluation.metrics import annualized_return, max_drawdown, oos_r2, sharpe_ratio
from timing.strategy.backtest import buy_and_hold, simulate
from timing.strategy.positions import forecast_z_score, target_weights

DATES = pd.bdate_range("2024-01-01", periods=3)
CONFIG = {
    "neutral_weight": 1.0,
    "z_scale": 0.5,
    "min_weight": 0.0,
    "max_weight": 1.5,
    "z_lookback": 3,
    "no_trade_band": 0.10,
}


def returns(spy, rf=0.0):
    return pd.DataFrame({"spy_ret": spy, "rf_ret": rf}, index=DATES)


def test_simulate_matches_a_hand_calculation():
    target = pd.Series([1.0, 0.5, 0.5], index=DATES)
    result = simulate(target, returns([0.10, 0.0, 0.0]), band=0.0, cost_bps=10.0)
    # Day 0: buy 1.0 from cash. Net = 0.10 - 1.0 * 0.001.
    assert result["net_ret"].iloc[0] == pytest.approx(0.10 - 0.001)
    # Day 1: weight stays 1.0 after drift with no cash, then sells to 0.5.
    assert result["trade"].iloc[1] == pytest.approx(0.5)
    assert result["cost"].iloc[1] == pytest.approx(0.5 * 0.001)
    assert result["trade"].iloc[2] == pytest.approx(0.0)


def test_drift_is_measured_before_comparing_to_the_band():
    target = pd.Series([0.5, 0.5, 0.5], index=DATES)
    result = simulate(target, returns([1.0, 0.0, 0.0]), band=0.10, cost_bps=0.0)
    # After SPY doubles, 0.5 drifts to 1.0 / 1.5 = 2/3, which is beyond the band.
    assert result["weight"].iloc[1] == pytest.approx(0.5)
    assert result["trade"].iloc[1] == pytest.approx(2 / 3 - 0.5)


def test_no_trade_inside_the_band():
    target = pd.Series([1.0, 1.05, 0.95], index=DATES)
    result = simulate(target, returns([0.0, 0.0, 0.0]), band=0.10, cost_bps=5.0)
    assert result["weight"].tolist() == [1.0, 1.0, 1.0]
    assert result["cost"].iloc[1:].sum() == 0.0


def test_cash_share_earns_the_cash_return():
    target = pd.Series([0.5, 0.5, 0.5], index=DATES)
    result = simulate(target, returns([0.0, 0.0, 0.0], rf=0.001), band=0.0, cost_bps=0.0)
    assert result["gross_ret"].iloc[0] == pytest.approx(0.0005)


def test_leverage_pays_the_cash_return():
    target = pd.Series([1.5, 1.5, 1.5], index=DATES)
    result = simulate(target, returns([0.01, 0.0, 0.0], rf=0.001), band=0.0, cost_bps=0.0)
    assert result["gross_ret"].iloc[0] == pytest.approx(1.5 * 0.01 - 0.5 * 0.001)


def test_warm_up_rows_are_dropped():
    target = pd.Series([np.nan, 1.0, 1.0], index=DATES)
    result = simulate(target, returns([0.5, 0.0, 0.0]), band=0.0, cost_bps=0.0)
    assert result.index.tolist() == DATES[1:].tolist()


def test_buy_and_hold_earns_spy_with_no_cost():
    result = buy_and_hold(returns([0.01, -0.02, 0.03]))
    assert result["net_ret"].tolist() == [0.01, -0.02, 0.03]
    assert result["cost"].sum() == 0.0


def test_z_score_uses_only_earlier_forecasts():
    y_hat = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0], dtype=float)
    z = forecast_z_score(y_hat, lookback=3)
    assert z.iloc[:3].isna().all()
    assert z.iloc[3] == pytest.approx((10.0 - 2.0) / 1.0)
    # Changing today's forecast must not change yesterday's z-score.
    changed = y_hat.copy()
    changed.iloc[4] = -500.0
    assert forecast_z_score(changed, 3).iloc[3] == z.iloc[3]


def test_constant_forecast_gives_a_neutral_weight():
    weights = target_weights(pd.Series([0.001] * 6), CONFIG)
    assert weights.iloc[:3].isna().all()
    assert (weights.iloc[3:] == 1.0).all()


def test_weights_stay_within_limits():
    y_hat = pd.Series([1.0, 2.0, 3.0, 1000.0, -1000.0])
    weights = target_weights(y_hat, CONFIG).dropna()
    assert weights.iloc[0] == 1.5
    assert weights.iloc[1] == 0.0


def test_a_higher_forecast_gives_a_higher_weight():
    history = [1.0, 2.0, 3.0]
    low = target_weights(pd.Series(history + [1.5]), CONFIG).iloc[-1]
    high = target_weights(pd.Series(history + [2.5]), CONFIG).iloc[-1]
    assert high > low


def test_oos_r2_values():
    y = pd.Series([1.0, -1.0])
    assert oos_r2(y, y, pd.Series([0.0, 0.0])) == 1.0
    assert oos_r2(y, pd.Series([0.0, 0.0]), pd.Series([0.0, 0.0])) == 0.0
    assert oos_r2(y, pd.Series([-1.0, 1.0]), pd.Series([0.0, 0.0])) == pytest.approx(-3.0)


def test_max_drawdown_includes_a_first_period_loss():
    assert max_drawdown(pd.Series([-0.5, 0.0])) == pytest.approx(-0.5)
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_annualized_return_and_sharpe():
    assert annualized_return(pd.Series([0.01] * 252), 252) == pytest.approx(1.01**252 - 1)
    net = pd.Series([0.01, 0.03])
    rf = pd.Series([0.0, 0.0])
    assert sharpe_ratio(net, rf, 252) == pytest.approx(0.02 / net.std() * np.sqrt(252))
    assert np.isnan(sharpe_ratio(pd.Series([0.01, 0.01]), rf, 252))
