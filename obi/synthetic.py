"""Synthetic top-of-book and trade data with a known imbalance effect.

Used by tests: the probability that the next mid move is up is
``0.5 + 0.5 * edge * imbalance``, so analysis code should recover a monotone
relationship and backtests should show the effect is far too small to pay
taker fees unless ``tick`` is large relative to the price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_book(
    n: int = 200_000,
    start_ts: int = 1_711_411_200_000,
    price: float = 70_000.0,
    tick: float = 0.1,
    edge: float = 0.6,
    mean_dt_ms: float = 5.0,
    p_move: float = 0.05,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ts = start_ts + np.cumsum(rng.exponential(mean_dt_ms, n)).astype(np.int64)
    move = rng.random(n) < p_move
    # Queues are redrawn when the price changes and only jitter in between,
    # so the displayed imbalance persists until the next move.
    regime = np.cumsum(move)
    n_reg = regime[-1] + 1
    bq = rng.gamma(1.5, 2.0, n_reg)[regime] * rng.uniform(0.9, 1.1, n)
    aq = rng.gamma(1.5, 2.0, n_reg)[regime] * rng.uniform(0.9, 1.1, n)
    imb = (bq - aq) / (bq + aq)
    up = rng.random(n) < 0.5 + 0.5 * edge * imb
    # Move at event i is driven by the imbalance displayed at event i-1.
    step = np.zeros(n)
    step[1:] = np.where(move[1:], np.where(up[:-1], 1.0, -1.0), 0.0)
    bid = price + tick * np.cumsum(step)
    ask = bid + tick
    book = pd.DataFrame(
        {"ts": ts, "update_id": np.arange(n, dtype=np.int64), "bid": bid, "bid_qty": bq, "ask": ask, "ask_qty": aq}
    )
    # Trades: an aggressive trade on each up/down move, plus noise trades.
    trade_rows = np.flatnonzero(step != 0)
    side = np.sign(step[trade_rows]).astype(np.int8)
    tpx = np.where(side > 0, ask[trade_rows - 1], bid[trade_rows - 1])
    trades = pd.DataFrame(
        {"ts": ts[trade_rows], "price": tpx, "qty": rng.gamma(1.0, 0.5, len(trade_rows)), "side": side}
    )
    return book, trades
