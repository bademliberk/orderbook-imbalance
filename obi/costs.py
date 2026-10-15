"""Trading-cost model for Hyperliquid perpetuals (linear, USDC-margined).

Returns are measured relative to entry notional. A long earns
``(exit - entry) * qty`` and a short earns ``(entry - exit) * qty``; fees are
``rate * notional`` on each fill, and the rate depends on whether the fill
added liquidity (maker, resting limit order) or removed it (taker).

Long  (buy at entry, sell at exit):
    net = (exit / entry) * (1 - f_exit) - 1 - f_entry
Short (sell at entry, buy back at exit):
    net = 1 - f_entry - (exit / entry) * (1 + f_exit)

Pass the actual fill prices: for limit orders that is the limit price (your own
side of the book), for market orders the opposite touch plus slippage. Spread is
already inside these prices and must not be subtracted again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeeSchedule:
    maker: float
    taker: float


# Hyperliquid perps fee tiers by 14-day weighted volume (docs, checked Oct 2026):
# tier: (min volume USD, taker, maker). Governance can change these.
HYPERLIQUID_TIERS = {
    0: (0, 0.00045, 0.00015),
    1: (5e6, 0.00040, 0.00012),
    2: (25e6, 0.00035, 0.00008),
    3: (100e6, 0.00030, 0.00004),
    4: (500e6, 0.00028, 0.0),
    5: (2e9, 0.00026, 0.0),
    6: (7e9, 0.00024, 0.0),
}
# HYPE staking discount by level (applied to the tier rate).
HYPE_STAKING_DISCOUNT = {"none": 0.0, "wood": 0.05, "bronze": 0.10, "silver": 0.15, "gold": 0.20, "platinum": 0.30, "diamond": 0.40}
REFERRAL_DISCOUNT = 0.04  # first $25M of volume


def hyperliquid_fees(tier: int = 0, staking: str = "none", referral: bool = False, maker_rebate: float = 0.0) -> FeeSchedule:
    """Effective fee schedule. ``maker_rebate`` (e.g. 0.00001) is the maker
    volume-share rebate, paid instead of a maker fee where it applies."""
    _, taker, maker = HYPERLIQUID_TIERS[tier]
    mult = (1 - HYPE_STAKING_DISCOUNT[staking]) * (1 - REFERRAL_DISCOUNT if referral else 1.0)
    maker = -maker_rebate if maker_rebate else maker * mult
    return FeeSchedule(maker=maker, taker=taker * mult)


HYPERLIQUID_BASE = hyperliquid_fees()  # 0.015% maker / 0.045% taker
ZERO_FEES = FeeSchedule(0.0, 0.0)


def long_net_return(entry, exit_, fee_entry: float, fee_exit: float):
    return np.asarray(exit_) / np.asarray(entry) * (1.0 - fee_exit) - 1.0 - fee_entry


def short_net_return(entry, exit_, fee_entry: float, fee_exit: float):
    return 1.0 - fee_entry - np.asarray(exit_) / np.asarray(entry) * (1.0 + fee_exit)


def net_return(side, entry, exit_, fee_entry, fee_exit):
    """Vectorised net return; ``side`` is +1 (long) or -1 (short)."""
    side = np.asarray(side)
    return np.where(
        side > 0,
        long_net_return(entry, exit_, fee_entry, fee_exit),
        short_net_return(entry, exit_, fee_entry, fee_exit),
    )


def breakeven_move(fee_entry: float, fee_exit: float) -> float:
    """Favourable move between the two fill prices a long needs to break even."""
    return (1.0 + fee_entry) / (1.0 - fee_exit) - 1.0


def round_trip_cost(fees: FeeSchedule, entry: str = "maker", exit_: str = "maker") -> float:
    """Approximate round-trip fee (fraction) for the given leg types."""
    return getattr(fees, entry) + getattr(fees, exit_)
