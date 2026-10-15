"""Causal microstructure features computed at each top-of-book event.

Every feature at row ``i`` uses only book events with index ``<= i`` and trades
with timestamp strictly earlier than ``ts[i]`` (same-millisecond trades may
arrive after the quote, so they are excluded to stay conservative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def window_sum(ts: np.ndarray, values: np.ndarray, window_ms: int) -> np.ndarray:
    """Sum of ``values`` over events in ``(ts[i] - window_ms, ts[i]]``."""
    cs = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    start = np.searchsorted(ts, ts - window_ms, side="right")
    return cs[np.arange(1, len(ts) + 1)] - cs[start]


def window_count(ts: np.ndarray, window_ms: int) -> np.ndarray:
    start = np.searchsorted(ts, ts - window_ms, side="right")
    return np.arange(1, len(ts) + 1) - start


def asof_index(ts: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the last event with ``ts <= query`` (-1 if none)."""
    return np.searchsorted(ts, query, side="right") - 1


def imbalance(bid_qty, ask_qty):
    bid_qty = np.asarray(bid_qty, dtype=np.float64)
    ask_qty = np.asarray(ask_qty, dtype=np.float64)
    total = bid_qty + ask_qty
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (bid_qty - ask_qty) / total, 0.0)


def multilevel_imbalance(bid_qtys: np.ndarray, ask_qtys: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """Depth-weighted imbalance across levels.

    ``bid_qtys``/``ask_qtys`` have shape ``(n_events, n_levels)``; level ``k``
    (0 = best) gets weight ``decay**k``.
    """
    w = decay ** np.arange(bid_qtys.shape[1])
    b = bid_qtys @ w
    a = ask_qtys @ w
    return imbalance(b, a)


def order_flow_imbalance(bid, bid_qty, ask, ask_qty) -> np.ndarray:
    """Per-event best-quote order flow imbalance (Cont, Kukanov & Stoikov 2014).

    e_n = q^b_n 1{P^b_n >= P^b_{n-1}} - q^b_{n-1} 1{P^b_n <= P^b_{n-1}}
        - q^a_n 1{P^a_n <= P^a_{n-1}} + q^a_{n-1} 1{P^a_n >= P^a_{n-1}}

    Positive values mean net buying pressure at the top of book. With L1 data
    this combines additions, cancellations and executions; separating them
    requires L3 (order-level) data, which these venues do not publish.
    """
    pb, qb, pa, qa = (np.asarray(x, dtype=np.float64) for x in (bid, bid_qty, ask, ask_qty))
    e = np.zeros(len(pb))
    if len(pb) < 2:
        return e
    e[1:] = (
        qb[1:] * (pb[1:] >= pb[:-1])
        - qb[:-1] * (pb[1:] <= pb[:-1])
        - qa[1:] * (pa[1:] <= pa[:-1])
        + qa[:-1] * (pa[1:] >= pa[:-1])
    )
    return e


def trade_window_sum(book_ts: np.ndarray, trade_ts: np.ndarray, values: np.ndarray, window_ms: int) -> np.ndarray:
    """Sum of trade ``values`` with ``book_ts - window_ms <= trade_ts < book_ts``."""
    cs = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    end = np.searchsorted(trade_ts, book_ts, side="left")
    start = np.searchsorted(trade_ts, book_ts - window_ms, side="left")
    return cs[end] - cs[start]


def build_features(
    book: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    windows_ms: tuple[int, ...] = (5_000, 30_000, 120_000),
) -> pd.DataFrame:
    """Compute the feature set from the cleaned book frame.

    ``book`` needs ``ts, bid, bid_qty, ask, ask_qty`` and optionally depth
    columns ``bid_qty_i``/``ask_qty_i``; ``trades`` (optional) needs
    ``ts, qty, side``. Default windows suit ~0.5 s Hyperliquid snapshots.
    """
    ts = book["ts"].to_numpy(np.int64)
    bid = book["bid"].to_numpy(np.float64)
    ask = book["ask"].to_numpy(np.float64)
    bq = book["bid_qty"].to_numpy(np.float64)
    aq = book["ask_qty"].to_numpy(np.float64)
    mid = 0.5 * (bid + ask)

    f = pd.DataFrame(index=book.index)
    f["imbalance"] = imbalance(bq, aq)
    f["spread_bps"] = (ask - bid) / mid * 1e4
    f["microprice_dev_bps"] = ((ask * bq + bid * aq) / (bq + aq) - mid) / mid * 1e4
    f["log_depth"] = np.log1p(bq + aq)
    levels = sum(1 for c in book.columns if c.startswith("bid_qty_"))
    if levels > 1:
        B = book[[f"bid_qty_{i}" for i in range(levels)]].to_numpy(np.float64)
        A = book[[f"ask_qty_{i}" for i in range(levels)]].to_numpy(np.float64)
        f[f"imbalance_l{levels}"] = multilevel_imbalance(B, A, decay=0.5)
        f[f"depth_imbalance_l{levels}"] = multilevel_imbalance(B, A, decay=1.0)

    ofi = order_flow_imbalance(bid, bq, ask, aq)
    log_mid = np.log(mid)
    r = np.diff(log_mid, prepend=log_mid[0])
    for w in windows_ms:
        depth = window_sum(ts, bq + aq, w) / np.maximum(window_count(ts, w), 1)
        f[f"ofi_{w}ms"] = window_sum(ts, ofi, w) / np.maximum(depth, 1e-12)
        f[f"vol_{w}ms_bps"] = np.sqrt(window_sum(ts, r * r, w)) * 1e4
        past = asof_index(ts, ts - w)
        f[f"ret_{w}ms_bps"] = np.where(past >= 0, (log_mid - log_mid[np.maximum(past, 0)]) * 1e4, 0.0)
        f[f"updates_{w}ms"] = window_count(ts, w)

    if trades is not None and len(trades):
        tts = trades["ts"].to_numpy(np.int64)
        tq = trades["qty"].to_numpy(np.float64)
        signed = tq * trades["side"].to_numpy(np.float64)
        for w in windows_ms:
            vol = trade_window_sum(ts, tts, tq, w)
            sv = trade_window_sum(ts, tts, signed, w)
            f[f"trade_imb_{w}ms"] = np.where(vol > 0, sv / np.maximum(vol, 1e-12), 0.0)
            f[f"trade_vol_{w}ms"] = vol
            f[f"trade_count_{w}ms"] = trade_window_sum(ts, tts, np.ones_like(tq), w)
    return f


# ------------------------------------------------------- signal-study features

PRICE_WINDOWS_MS = (2_000, 5_000, 15_000, 30_000, 60_000, 300_000)
BANDS_BPS = (1, 2, 5, 10)


def lagged_change(ts: np.ndarray, x: np.ndarray, lag_ms: int) -> np.ndarray:
    """``x[i] - x(at ts[i] - lag_ms)`` (0 where there is no earlier value)."""
    past = asof_index(ts, ts - lag_ms)
    return np.where(past >= 0, x - x[np.maximum(past, 0)], 0.0)


def price_vector_features(ts: np.ndarray, mid: np.ndarray, windows_ms=PRICE_WINDOWS_MS) -> dict[str, np.ndarray]:
    """The recent price-change vector, per look-back window ``w``:

    - ``ret_w`` (bps): signed move — direction and size;
    - ``speed_w`` (bps/s): |move| per second — how fast, regardless of direction;
    - ``eff_w`` in [-1, 1]: net move / total path — how one-directional the
      move was (1 = straight up, ~0 = choppy);
    - ``activity_w`` (changes/s): how often the mid changed;

    plus ``accel_short`` = velocity(5 s) - velocity(30 s) and
    ``accel_long`` = velocity(30 s) - velocity(300 s), in bps/s.
    """
    log_mid = np.log(mid)
    r = np.diff(log_mid, prepend=log_mid[0])
    out: dict[str, np.ndarray] = {}
    vel = {}
    for w in windows_ms:
        s = w / 1e3
        ret = lagged_change(ts, log_mid, w) * 1e4
        path = window_sum(ts, np.abs(r), w) * 1e4
        out[f"ret_{s:g}s"] = ret
        out[f"speed_{s:g}s"] = np.abs(ret) / s
        out[f"eff_{s:g}s"] = np.where(path > 0, ret / np.maximum(path, 1e-12), 0.0)
        out[f"activity_{s:g}s"] = window_sum(ts, (r != 0).astype(np.float64), w) / s
        vel[w] = ret / s
    if {5_000, 30_000, 300_000} <= set(windows_ms):
        out["accel_short"] = vel[5_000] - vel[30_000]
        out["accel_long"] = vel[30_000] - vel[300_000]
    return out


def band_imbalance(book: pd.DataFrame, bands_bps=BANDS_BPS) -> dict[str, np.ndarray]:
    """Notional imbalance of resting size within ``n`` bps of mid, per band.

    ``(bid notional - ask notional) / total`` over levels priced within the
    band. With only the levels present in ``book`` the band is truncated where
    the book is shallower than ``n`` bps; ``band_cov_n`` is the share of rows
    where both sides of the visible book reach the full band.
    """
    levels = sum(1 for c in book.columns if c.startswith("bid_qty_"))
    bid_px = book[[f"bid_px_{i}" for i in range(levels)]].to_numpy(np.float64)
    ask_px = book[[f"ask_px_{i}" for i in range(levels)]].to_numpy(np.float64)
    bid_q = book[[f"bid_qty_{i}" for i in range(levels)]].to_numpy(np.float64)
    ask_q = book[[f"ask_qty_{i}" for i in range(levels)]].to_numpy(np.float64)
    mid = 0.5 * (bid_px[:, 0] + ask_px[:, 0])
    bid_d = (mid[:, None] - bid_px) / mid[:, None] * 1e4
    ask_d = (ask_px - mid[:, None]) / mid[:, None] * 1e4
    reach = np.minimum(np.nanmax(bid_d, axis=1), np.nanmax(ask_d, axis=1))
    out = {}
    for n in bands_bps:
        b = np.nansum(np.where(bid_d <= n, bid_q * bid_px, 0.0), axis=1)
        a = np.nansum(np.where(ask_d <= n, ask_q * ask_px, 0.0), axis=1)
        out[f"band_imb_{n}bps"] = imbalance(b, a)
        out[f"band_cov_{n}bps"] = (reach >= n).astype(np.float64)
    return out


def build_study_features(book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
    """Base features plus the price vector, band depth and spread/imbalance dynamics."""
    f = build_features(book, trades)
    ts = book["ts"].to_numpy(np.int64)
    mid = 0.5 * (book["bid"].to_numpy(np.float64) + book["ask"].to_numpy(np.float64))
    for k, v in price_vector_features(ts, mid).items():
        f[k] = v
    for k, v in band_imbalance(book).items():
        f[k] = v
    spread = f["spread_bps"].to_numpy()
    f["spread_chg_5s"] = lagged_change(ts, spread, 5_000)
    f["spread_max_30s"] = pd.Series(spread, index=pd.to_datetime(ts, unit="ms")).rolling("30s").max().to_numpy()
    for col in ("imbalance_l5", "band_imb_5bps"):
        if col in f:
            f[f"{col}_chg_5s"] = lagged_change(ts, f[col].to_numpy(), 5_000)
    return f


DEPTH_BANDS_BPS = (10, 25, 50, 100, 200)  # 0.1% ... 2% of mid


def merge_depth_bands(exec_ts: np.ndarray, agg_books: dict[int, pd.DataFrame], bands_bps=DEPTH_BANDS_BPS) -> pd.DataFrame:
    """Depth imbalance within ``n`` bps of mid from aggregated books, aligned to ``exec_ts``.

    ``agg_books`` maps ``nSigFigs`` -> 20-level book frame (``ts``, ``bid_px_i``,
    ...). For each band and each execution timestamp the most recent snapshot
    (``ts <= exec_ts``, so causal) of the *finest* aggregation that reaches
    the band on both sides is used; if none reaches it, the coarsest is used
    and ``depth_cov_<n>bps`` is 0. ``depth_age_ms`` is the snapshot age.
    """
    out = pd.DataFrame(index=np.arange(len(exec_ts)))
    per_n = {}
    for n, book in agg_books.items():
        bands = band_imbalance(book, bands_bps)
        idx = asof_index(book["ts"].to_numpy(np.int64), exec_ts)
        ok = idx >= 0
        safe = np.maximum(idx, 0)
        per_n[n] = (bands, safe, ok, np.where(ok, exec_ts - book["ts"].to_numpy(np.int64)[safe], np.nan))
    order = sorted(per_n, reverse=True)  # finest (most significant figures) first
    for b in bands_bps:
        val = np.full(len(exec_ts), np.nan)
        cov = np.zeros(len(exec_ts))
        age = np.full(len(exec_ts), np.nan)
        for n in order:
            bands, safe, ok, a = per_n[n]
            take = ok & np.isnan(val) & (bands[f"band_cov_{b}bps"][safe] == 1.0)
            val[take], cov[take], age[take] = bands[f"band_imb_{b}bps"][safe][take], 1.0, a[take]
        coarsest = order[-1]
        bands, safe, ok, a = per_n[coarsest]
        rest = ok & np.isnan(val)
        val[rest], age[rest] = bands[f"band_imb_{b}bps"][safe][rest], a[rest]
        out[f"depth_imb_{b}bps"] = val
        out[f"depth_cov_{b}bps"] = cov
        out[f"depth_age_ms_{b}bps"] = age
    return out
