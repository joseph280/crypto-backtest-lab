"""Tests for the cost model and the vectorbt engine."""

import numpy as np
import pandas as pd

from xortcut.backtest.costs import CostModel, funding_return, turnover
from xortcut.backtest.engine import position_to_signals, run_backtest


def test_funding_sign_and_proration():
    pos = pd.Series([1.0, -1.0, 0.0])
    rate = pd.Series([0.001, 0.001, 0.001])  # positive funding
    # Hourly funding on 1h bars: factor 1. Long pays (negative return), short
    # receives (positive), flat is zero.
    f = funding_return(pos, rate, interval_seconds=3600, settlement="hourly")
    assert f.iloc[0] < 0 and f.iloc[1] > 0 and f.iloc[2] == 0.0
    # 15m bars prorate the hourly rate to a quarter.
    f15 = funding_return(pos, rate, interval_seconds=900, settlement="hourly")
    assert np.isclose(f15.iloc[0], f.iloc[0] / 4.0)


def test_turnover_counts_first_position_as_a_trade():
    pos = pd.Series([1.0, 1.0, -1.0, 0.0])
    t = turnover(pos)
    assert t.iloc[0] == 1.0  # opening
    assert t.iloc[1] == 0.0  # held
    assert t.iloc[2] == 2.0  # flip long to short
    assert t.iloc[3] == 1.0  # close


def test_position_to_signals_roundtrip():
    pos = pd.Series([0, 1, 1, 0, -1, -1, 0.0])
    le, lx, se, sx = position_to_signals(pos)
    assert le.iloc[1] and lx.iloc[3]
    assert se.iloc[4] and sx.iloc[6]


def test_flat_position_has_no_trades_and_zero_return():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01)))
    pos = pd.Series(0.0, index=close.index)
    res = run_backtest(close, pos, interval="1h", funding_rate=pd.Series(0.001, index=close.index))
    assert res.n_trades == 0
    assert abs(res.total_return) < 1e-9


def test_costs_make_a_round_trip_lose_versus_gross():
    # A single long held one bar: net return must be below the raw price move
    # because fees and slippage are charged.
    close = pd.Series([100.0, 101.0, 101.0])
    pos = pd.Series([1.0, 0.0, 0.0])
    cost = CostModel.from_settings(prefer_maker=False)
    res = run_backtest(close, pos, interval="1h", prefer_maker=False)
    gross_move = 0.01  # 100 -> 101
    assert res.total_return < gross_move
    # The drag is at least the two-sided fee plus slippage.
    assert gross_move - res.total_return >= (cost.fee_frac + cost.slippage_frac)
