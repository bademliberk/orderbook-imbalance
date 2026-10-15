"""Forward executable-return labels (plan step 2).

Timing model for a decision made on the book event at ``ts[i]``:

- the order reaches the exchange at ``ts[i] + latency_ms`` and trades against
  the book prevailing then (not the book at decision time);
- the exit order is sent at ``ts[i] + horizon_ms`` and fills against the book at
  ``ts[i] + horizon_ms + latency_ms``.

So both legs pay latency, and the horizon is measured from the decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import asof_index


def forward_rolling_extreme(ts: np.ndarray, x: np.ndarray, window_ms: int, how: str) -> np.ndarray:
    """max/min of ``x`` over events with ``ts[i] <= ts < ts[i] + window_ms``."""
    # Reverse time so a trailing window becomes a leading one. Rows sharing a
    # timestamp must stay in reverse order, so break ties by position.
    idx = pd.to_datetime(-ts[::-1], unit="ms")
    s = pd.Series(x[::-1], index=idx)
    roll = s.rolling(f"{window_ms}ms", closed="right")
    out = (roll.max() if how == "max" else roll.min()).to_numpy()[::-1]
    return out


def executable_labels(
    book: pd.DataFrame,
    horizons_ms: tuple[int, ...],
    latency_ms: int = 0,
    with_excursions: bool = True,
) -> pd.DataFrame:
    """Gross executable forward returns for long and short taker round trips.

    Columns per horizon ``h`` (all as fractions, gross of fees):

    - ``mid_ret_{h}``: mid-price change from decision to exit.
    - ``long_ret_{h}``: ``exit_bid / entry_ask - 1``.
    - ``short_ret_{h}``: ``1 - exit_ask / entry_bid`` (linear-perp short).
    - ``long_mfe_{h}`` / ``long_mae_{h}``: best/worst ``bid / entry_ask - 1``
      between entry and exit; ``short_mfe_{h}`` / ``short_mae_{h}`` likewise.

    Rows whose exit time is beyond the data are NaN.
    """
    ts = book["ts"].to_numpy(np.int64)
    bid = book["bid"].to_numpy(np.float64)
    ask = book["ask"].to_numpy(np.float64)
    mid = 0.5 * (bid + ask)
    n = len(ts)
    out = pd.DataFrame(index=book.index)

    entry = asof_index(ts, ts + latency_ms)
    entry_ask = ask[entry]
    entry_bid = bid[entry]
    # The next mid change in event time (direction label for imbalance studies).
    changes = np.flatnonzero(np.diff(mid) != 0) + 1
    nxt = np.searchsorted(changes, np.arange(n), side="right")
    has_next = nxt < len(changes)
    next_mid = np.where(has_next, mid[changes[np.minimum(nxt, len(changes) - 1)]], np.nan)
    out["next_mid_move"] = np.sign(next_mid - mid)

    for h in horizons_ms:
        exit_t = ts + h + latency_ms
        ex = asof_index(ts, exit_t)
        beyond = exit_t > ts[-1]
        long_r = bid[ex] / entry_ask - 1.0
        short_r = 1.0 - ask[ex] / entry_bid
        mid_r = mid[ex] / mid - 1.0
        for name, v in (("mid_ret", mid_r), ("long_ret", long_r), ("short_ret", short_r)):
            v = v.copy()
            v[beyond] = np.nan
            out[f"{name}_{h}"] = v
        if with_excursions:
            # Window of the position's life: from entry to exit fill.
            hold = h  # exit fills at ts+h+L, entry at ts+L
            max_bid = forward_rolling_extreme(ts, bid, hold + 1, "max")[entry]
            min_bid = forward_rolling_extreme(ts, bid, hold + 1, "min")[entry]
            max_ask = forward_rolling_extreme(ts, ask, hold + 1, "max")[entry]
            min_ask = forward_rolling_extreme(ts, ask, hold + 1, "min")[entry]
            for name, v in (
                ("long_mfe", max_bid / entry_ask - 1.0),
                ("long_mae", min_bid / entry_ask - 1.0),
                ("short_mfe", 1.0 - min_ask / entry_bid),
                ("short_mae", 1.0 - max_ask / entry_bid),
            ):
                v = v.copy()
                v[beyond] = np.nan
                out[f"{name}_{h}"] = v
    return out
