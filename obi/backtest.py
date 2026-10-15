"""Event-driven backtester for one Hyperliquid perpetual (plan step 3).

Execution model
---------------
Limit (maker) legs — the default for both entry and exit:

- A post-only (ALO) limit order is placed at our side's touch (bid for a buy,
  ask for a sell) when it reaches the exchange, ``latency_ms`` after the
  decision. It joins the queue behind ``queue_fraction`` of the displayed size.
- It fills when (a) a trade prints *through* our price, or (b) aggressive
  volume printed *at* our price exceeds the queue ahead of us plus our size.
  Public trades are the only evidence of fills, so this is conservative (it
  ignores cancellations ahead of us).
- Entry: unfilled after ``maker_timeout_ms`` -> cancelled, no trade.
- Exit: unfilled after ``exit_timeout_ms`` -> cancelled and closed with a
  market (taker) order (reason ``*_fallback``). The stop is not monitored while
  the exit order rests; the timeout bounds that exposure.

Market (taker) legs:

- Marketable IOC fills against the touch ``latency_ms`` later; size beyond the
  displayed top level fills ``depth_slip_bps`` worse; ``extra_slip_bps`` is
  added to every taker fill. Entries are rejected if the touch moved more than
  ``max_entry_slip_bps`` (price protection).

Exits trigger on profit target, stop loss or time limit. Targets are evaluated
on the mid price, stops on the executable side (bid for longs, ask for shorts)
and always exit with a market order. Funding is charged hourly, as on
Hyperliquid, for each funding time the position is open across.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import HYPERLIQUID_BASE, FeeSchedule
from .features import asof_index

FUNDING_INTERVAL_MS = 3600 * 1000


@dataclass(frozen=True)
class ExecConfig:
    fees: FeeSchedule = HYPERLIQUID_BASE
    latency_ms: int = 200
    entry_mode: str = "maker"  # "maker" (post-only limit) or "taker"
    exit_mode: str = "maker"
    maker_timeout_ms: int = 5_000
    exit_timeout_ms: int = 5_000
    queue_fraction: float = 1.0  # share of displayed size ahead of us when we join
    target: float | None = None  # net return (on mid) that triggers a profit exit
    stop: float | None = None  # net return (negative, executable side) that triggers a stop
    max_hold_ms: int = 60_000
    order_qty: float = 0.01
    extra_slip_bps: float = 0.0
    depth_slip_bps: float = 0.5
    max_entry_slip_bps: float | None = 2.0
    funding_rate: float = 0.0000125  # per hour (0.01% per 8 h), paid by longs when positive
    cooldown_ms: int = 0


def _taker_fill(side: int, px_touch: float, qty_touch: float, cfg: ExecConfig) -> tuple[float, float]:
    """Average taker fill price for an order of ``side`` and its slippage (fraction)."""
    q = cfg.order_qty
    at_touch = min(q, qty_touch)
    worse = px_touch * (1 + side * cfg.depth_slip_bps * 1e-4)
    avg = (at_touch * px_touch + (q - at_touch) * worse) / q
    avg *= 1 + side * cfg.extra_slip_bps * 1e-4
    return avg, side * (avg - px_touch) / px_touch


def _funding_times(t0: int, t1: int) -> int:
    return int(t1 // FUNDING_INTERVAL_MS - t0 // FUNDING_INTERVAL_MS)


def maker_fill_time(side, limit, queue_total, t_post, t_cancel, tts, tpx, tqty, tside):
    """Time a resting limit order of ``side`` (+1 buy, -1 sell) fills, or None."""
    j0 = np.searchsorted(tts, t_post, "right")
    j1 = np.searchsorted(tts, t_cancel, "right")
    if j1 <= j0:
        return None
    px, sd, q = tpx[j0:j1], tside[j0:j1], tqty[j0:j1]
    if side > 0:  # resting bid is hit by sell aggressors
        through = np.flatnonzero((sd < 0) & (px < limit))
        at_px = (sd < 0) & (px == limit)
    else:  # resting ask is lifted by buy aggressors
        through = np.flatnonzero((sd > 0) & (px > limit))
        at_px = (sd > 0) & (px == limit)
    depleted = np.flatnonzero(np.cumsum(np.where(at_px, q, 0.0)) >= queue_total)
    cands = [x[0] for x in (through, depleted) if len(x)]
    return int(tts[j0 + min(cands)]) if cands else None


def run_backtest(
    book: pd.DataFrame,
    signal: np.ndarray,
    cfg: ExecConfig,
    trades: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Simulate ``signal`` (+1 long, -1 short, 0 flat) on ``book``.

    Returns the trade log and order statistics.
    """
    ts = book["ts"].to_numpy(np.int64)
    bid = book["bid"].to_numpy(np.float64)
    ask = book["ask"].to_numpy(np.float64)
    bq = book["bid_qty"].to_numpy(np.float64)
    aq = book["ask_qty"].to_numpy(np.float64)
    mid = 0.5 * (bid + ask)
    valid = book["valid"].to_numpy(bool) if "valid" in book else np.ones(len(ts), bool)
    signal = np.asarray(signal)
    uses_maker = "maker" in (cfg.entry_mode, cfg.exit_mode)
    if uses_maker and trades is None:
        raise ValueError("limit-order legs need public trades for fill modelling")
    if trades is not None:
        tr_arrays = (
            trades["ts"].to_numpy(np.int64),
            trades["price"].to_numpy(np.float64),
            trades["qty"].to_numpy(np.float64),
            trades["side"].to_numpy(np.int8),
        )
    fees = cfg.fees
    at = lambda t: int(asof_index(ts, np.array([t]))[0])  # noqa: E731
    half_spread = lambda k: (ask[k] - bid[k]) / (2 * mid[k])  # noqa: E731

    candidates = np.flatnonzero((signal != 0) & valid)
    t_end = ts[-1]
    worst_case_ms = cfg.latency_ms * 3 + cfg.maker_timeout_ms + cfg.max_hold_ms + cfg.exit_timeout_ms
    log = []
    stats = {"decisions": 0, "submitted": 0, "filled": 0, "partial": 0, "rejected": 0, "unfilled_maker": 0, "exit_fallback": 0}
    next_free_ts = ts[0]
    ci = 0
    while ci < len(candidates):
        i = candidates[ci]
        ci += 1
        if ts[i] < next_free_ts:
            ci = int(np.searchsorted(candidates, np.searchsorted(ts, next_free_ts, "left"), "left"))
            continue
        t_dec = int(ts[i])
        if t_dec + worst_case_ms > t_end:
            break
        side = int(np.sign(signal[i]))
        stats["decisions"] += 1
        stats["submitted"] += 1
        t_arr = t_dec + cfg.latency_ms
        e = at(t_arr)

        # ---- entry ----
        if cfg.entry_mode == "taker":
            touch, touch_qty = (ask[e], aq[e]) if side > 0 else (bid[e], bq[e])
            ref = ask[i] if side > 0 else bid[i]
            if cfg.max_entry_slip_bps is not None and side * (touch - ref) / ref * 1e4 > cfg.max_entry_slip_bps:
                stats["rejected"] += 1
                next_free_ts = t_arr + cfg.cooldown_ms
                continue
            entry_px, entry_slip = _taker_fill(side, touch, touch_qty, cfg)
            stats["partial"] += int(touch_qty < cfg.order_qty)
            t_entry, fee_entry, entry_spread = t_arr, fees.taker, half_spread(e)
        else:
            limit = bid[e] if side > 0 else ask[e]
            ahead = (bq[e] if side > 0 else aq[e]) * cfg.queue_fraction
            t_cancel = t_arr + cfg.maker_timeout_ms
            fill_t = maker_fill_time(side, limit, ahead + cfg.order_qty, t_arr, t_cancel, *tr_arrays)
            if fill_t is None:
                stats["unfilled_maker"] += 1
                next_free_ts = t_cancel + cfg.latency_ms
                continue
            entry_px, entry_slip, t_entry, fee_entry = limit, 0.0, fill_t, fees.maker
            entry_spread = -half_spread(e)
            e = at(t_entry)
        stats["filled"] += 1

        # ---- exit trigger ----
        t_deadline = t_entry + cfg.max_hold_ms
        k_end = at(t_deadline)
        reason, t_trig = "time", t_deadline
        if cfg.target is not None or cfg.stop is not None:
            sl = slice(e + 1, k_end + 1)
            fx_t = fees.maker if cfg.exit_mode == "maker" else fees.taker
            if side > 0:
                mark_t = mid[sl] / entry_px * (1 - fx_t) - 1 - fee_entry
                mark_s = bid[sl] / entry_px * (1 - fees.taker) - 1 - fee_entry
            else:
                mark_t = 1 - fee_entry - mid[sl] / entry_px * (1 + fx_t)
                mark_s = 1 - fee_entry - ask[sl] / entry_px * (1 + fees.taker)
            hit_t = np.flatnonzero(mark_t >= cfg.target) if cfg.target is not None else np.array([], int)
            hit_s = np.flatnonzero(mark_s <= cfg.stop) if cfg.stop is not None else np.array([], int)
            first_t = hit_t[0] if len(hit_t) else np.inf
            first_s = hit_s[0] if len(hit_s) else np.inf
            if first_t < first_s:
                reason, t_trig = "target", int(ts[e + 1 + int(first_t)])
            elif first_s < np.inf:
                reason, t_trig = "stop", int(ts[e + 1 + int(first_s)])

        # ---- exit execution ----
        t_xarr = t_trig + cfg.latency_ms
        x = at(t_xarr)
        exit_px = None
        if cfg.exit_mode == "maker" and reason != "stop":
            limit = ask[x] if side > 0 else bid[x]
            ahead = (aq[x] if side > 0 else bq[x]) * cfg.queue_fraction
            fill_t = maker_fill_time(-side, limit, ahead + cfg.order_qty, t_xarr, t_xarr + cfg.exit_timeout_ms, *tr_arrays)
            if fill_t is not None:
                exit_px, exit_slip, t_exit, fee_exit = limit, 0.0, fill_t, fees.maker
                exit_spread, exit_type = -half_spread(x), "maker"
            else:
                stats["exit_fallback"] += 1
                reason += "_fallback"
                t_xarr = t_xarr + cfg.exit_timeout_ms + cfg.latency_ms
                x = at(t_xarr)
        if exit_px is None:
            x_touch, x_qty = (bid[x], bq[x]) if side > 0 else (ask[x], aq[x])
            exit_px, exit_slip = _taker_fill(-side, x_touch, x_qty, cfg)
            t_exit, fee_exit, exit_spread, exit_type = t_xarr, fees.taker, half_spread(x), "taker"

        if side > 0:
            gross = exit_px / entry_px - 1
            net = exit_px / entry_px * (1 - fee_exit) - 1 - fee_entry
        else:
            gross = 1 - exit_px / entry_px
            net = 1 - fee_entry - exit_px / entry_px * (1 + fee_exit)
        funding = side * cfg.funding_rate * _funding_times(t_entry, t_exit)
        net -= funding
        log.append(
            {
                "t_decision": t_dec,
                "side": side,
                "t_entry": t_entry,
                "entry_px": entry_px,
                "entry_type": cfg.entry_mode,
                "t_exit": t_exit,
                "exit_px": exit_px,
                "exit_type": exit_type,
                "reason": reason,
                "gross": gross,
                "fees": fee_entry + fee_exit * (1 + gross if side > 0 else 1 - gross),
                "spread_cost": entry_spread + exit_spread,
                "slippage": entry_slip + exit_slip,
                "funding": funding,
                "net": net,
                "hold_ms": t_exit - t_entry,
            }
        )
        next_free_ts = t_exit + cfg.cooldown_ms

    return pd.DataFrame(log), stats
