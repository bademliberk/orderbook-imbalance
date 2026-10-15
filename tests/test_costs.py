import numpy as np
import pytest

from obi import costs


def test_hyperliquid_base_fees():
    f = costs.HYPERLIQUID_BASE
    assert (f.maker, f.taker) == (0.00015, 0.00045)
    assert costs.round_trip_cost(f, "maker", "maker") == pytest.approx(0.0003)
    assert costs.round_trip_cost(f, "taker", "taker") == pytest.approx(0.0009)


def test_tiers_staking_referral_and_rebate():
    assert costs.hyperliquid_fees(3).maker == pytest.approx(0.00004)
    gold = costs.hyperliquid_fees(0, "gold")
    assert (gold.maker, gold.taker) == pytest.approx((0.00012, 0.00036))
    ref = costs.hyperliquid_fees(0, "diamond", referral=True)
    assert ref.taker == pytest.approx(0.00045 * 0.6 * 0.96)
    assert costs.hyperliquid_fees(4, maker_rebate=0.00002).maker == pytest.approx(-0.00002)


def test_breakeven_long_and_short_are_zero_net():
    fe, fx = 0.00015, 0.00045
    be = costs.breakeven_move(fe, fx)
    assert costs.long_net_return(100.0, 100.0 * (1 + be), fe, fx) == pytest.approx(0.0, abs=1e-15)
    # short: exit/entry must fall to (1 - fe) / (1 + fx)
    assert costs.short_net_return(100.0, 100.0 * (1 - fe) / (1 + fx), fe, fx) == pytest.approx(0.0, abs=1e-15)


def test_short_return_is_relative_to_entry_notional():
    # Linear perp: sell 1 at 100, buy back at 90 => +10 on 100 notional = 10%.
    assert costs.short_net_return(100.0, 90.0, 0.0, 0.0) == pytest.approx(0.10)


def test_net_matches_cashflows():
    entry, exit_, fe, fx = 100.0, 101.0, 0.00015, 0.00045
    long_pnl = (exit_ - entry) - fe * entry - fx * exit_
    short_pnl = (entry - exit_) - fe * entry - fx * exit_
    assert costs.long_net_return(entry, exit_, fe, fx) == pytest.approx(long_pnl / entry)
    assert costs.short_net_return(entry, exit_, fe, fx) == pytest.approx(short_pnl / entry)


def test_vectorised_net_return():
    r = costs.net_return(np.array([1, -1]), np.array([100.0, 100.0]), np.array([101.0, 99.0]), 0.0, 0.0)
    assert r == pytest.approx([0.01, 0.01])
