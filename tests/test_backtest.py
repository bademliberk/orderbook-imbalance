import numpy as np
import pandas as pd
import pytest

from obi.backtest import ExecConfig, run_backtest
from obi.costs import ZERO_FEES, FeeSchedule
from obi.synthetic import make_book

TAKER = dict(entry_mode="taker", exit_mode="taker")


def step_book():
    # price jumps from 100 to 110 at ts=50
    ts = np.arange(0, 200, 10)
    bid = np.where(ts < 50, 100.0, 110.0)
    return pd.DataFrame({"ts": ts, "bid": bid, "bid_qty": 5.0, "ask": bid + 1, "ask_qty": 5.0})


def signal_at(n, i, side=1):
    s = np.zeros(n, int)
    s[i] = side
    return s


def test_latency_means_entry_after_jump():
    bk = step_book()
    base = dict(fees=ZERO_FEES, max_hold_ms=50, order_qty=1, max_entry_slip_bps=None, depth_slip_bps=0, maker_timeout_ms=0, exit_timeout_ms=0, **TAKER)
    tr, _ = run_backtest(bk, signal_at(len(bk), 4), ExecConfig(latency_ms=0, **base))
    assert tr.loc[0, "entry_px"] == 101.0 and tr.loc[0, "exit_px"] == 110.0
    tr, _ = run_backtest(bk, signal_at(len(bk), 4), ExecConfig(latency_ms=15, **base))
    assert tr.loc[0, "entry_px"] == 111.0


def test_price_protection_rejects():
    bk = step_book()
    cfg = ExecConfig(fees=ZERO_FEES, latency_ms=15, max_hold_ms=50, order_qty=1, max_entry_slip_bps=5, maker_timeout_ms=0, exit_timeout_ms=0, **TAKER)
    tr, st = run_backtest(bk, signal_at(len(bk), 4), cfg)
    assert tr.empty and st["rejected"] == 1


def test_partial_fill_slippage():
    bk = step_book()
    cfg = ExecConfig(fees=ZERO_FEES, latency_ms=0, max_hold_ms=20, order_qty=10, depth_slip_bps=10, max_entry_slip_bps=None, maker_timeout_ms=0, exit_timeout_ms=0, **TAKER)
    tr, st = run_backtest(bk, signal_at(len(bk), 0), cfg)
    assert st["partial"] == 1
    assert tr.loc[0, "entry_px"] == pytest.approx((5 * 101 + 5 * 101 * 1.001) / 10)


def test_target_exit_and_fees():
    bk = step_book()
    fees = FeeSchedule(0.0, 0.0005)
    cfg = ExecConfig(fees=fees, latency_ms=0, target=0.05, stop=-0.05, max_hold_ms=150, order_qty=1, max_entry_slip_bps=None, maker_timeout_ms=0, exit_timeout_ms=0, **TAKER)
    tr, _ = run_backtest(bk, signal_at(len(bk), 0), cfg)
    assert tr.loc[0, "reason"] == "target"
    assert tr.loc[0, "net"] == pytest.approx(110 / 101 * (1 - 0.0005) - 1 - 0.0005)


def test_short_trade_net():
    bk = step_book()
    bk["bid"] = bk["bid"][::-1].to_numpy()
    bk["ask"] = bk["bid"] + 1
    cfg = ExecConfig(fees=ZERO_FEES, latency_ms=0, max_hold_ms=50, order_qty=1, max_entry_slip_bps=None, maker_timeout_ms=0, exit_timeout_ms=0, **TAKER)
    tr, _ = run_backtest(bk, signal_at(len(bk), len(bk) - 10, -1), cfg)
    assert tr.loc[0, "side"] == -1
    assert tr.loc[0, "net"] == pytest.approx(1 - tr.loc[0, "exit_px"] / tr.loc[0, "entry_px"])


def test_no_overlapping_positions():
    book, trades = make_book(n=20_000, seed=3)
    cfg = ExecConfig(fees=ZERO_FEES, latency_ms=5, max_hold_ms=200, max_entry_slip_bps=None, maker_timeout_ms=500, exit_timeout_ms=500)
    tr, _ = run_backtest(book, np.ones(len(book), int), cfg, trades=trades)
    assert len(tr) > 0
    assert (tr["t_entry"].to_numpy()[1:] >= tr["t_exit"].to_numpy()[:-1]).all()


def flat_book():
    ts = np.arange(0, 200, 10)
    return pd.DataFrame({"ts": ts, "bid": 100.0, "bid_qty": 3.0, "ask": 100.1, "ask_qty": 3.0})


def test_limit_entry_and_limit_exit_charge_maker_fees():
    bk = flat_book()
    trades = pd.DataFrame(
        {
            "ts": [15, 25, 35, 95, 105],
            "price": [100.0, 100.0, 100.0, 100.1, 100.1],
            "qty": [2.0, 1.5, 1.0, 3.0, 2.0],
            "side": [-1, -1, -1, 1, 1],
        }
    )
    fees = FeeSchedule(maker=0.00015, taker=0.00045)
    cfg = ExecConfig(fees=fees, latency_ms=0, maker_timeout_ms=40, exit_timeout_ms=60, max_hold_ms=50, order_qty=1.0)
    tr, st = run_backtest(bk, signal_at(len(bk), 0), cfg, trades=trades)
    # entry queue 3 + own 1 = 4 => filled at the third sell print (cum 4.5)
    assert st["filled"] == 1 and tr.loc[0, "t_entry"] == 35 and tr.loc[0, "entry_px"] == 100.0
    # exit posted at ask 100.1 at t=85, queue 4 => filled at t=105 (cum 5)
    assert tr.loc[0, "exit_type"] == "maker" and tr.loc[0, "t_exit"] == 105 and tr.loc[0, "exit_px"] == 100.1
    assert tr.loc[0, "net"] == pytest.approx(100.1 / 100.0 * (1 - 0.00015) - 1 - 0.00015)


def test_unfilled_limit_exit_falls_back_to_taker():
    bk = flat_book()
    trades = pd.DataFrame({"ts": [15], "price": [99.9], "qty": [0.1], "side": [-1]})  # trade-through fills the entry
    fees = FeeSchedule(maker=0.00015, taker=0.00045)
    cfg = ExecConfig(fees=fees, latency_ms=0, maker_timeout_ms=40, exit_timeout_ms=30, max_hold_ms=50, order_qty=1.0)
    tr, st = run_backtest(bk, signal_at(len(bk), 0), cfg, trades=trades)
    assert st["exit_fallback"] == 1
    assert tr.loc[0, "reason"] == "time_fallback" and tr.loc[0, "exit_type"] == "taker"
    assert tr.loc[0, "exit_px"] == 100.0  # sold at the bid
    assert tr.loc[0, "net"] == pytest.approx(100.0 / 100.0 * (1 - 0.00045) - 1 - 0.00015)


def test_unfilled_limit_entry_is_not_a_trade():
    bk = flat_book()
    trades = pd.DataFrame({"ts": [15], "price": [100.0], "qty": [1.0], "side": [-1]})
    cfg = ExecConfig(latency_ms=0, maker_timeout_ms=40, exit_timeout_ms=30, max_hold_ms=50, order_qty=1.0)
    tr, st = run_backtest(bk, signal_at(len(bk), 0), cfg, trades=trades)
    assert tr.empty and st["unfilled_maker"] == 1
