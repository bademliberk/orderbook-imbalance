"""Live dry run on Hyperliquid: predict up / down / calm from order book
imbalance, verify each call after the horizon (default 60 s), and simulate the
limit-order round trip each up/down call would have made. No orders are sent.

Feeds (see ``obi.data.hyperliquid``): ``l2Book`` fast snapshots drive features
(identical to the research data), ``bbo`` gives verification and order prices,
``trades`` decide whether simulated limit orders fill. All timing uses exchange
timestamps (ms).

Simulated execution per up/down call (independent virtual orders, so calls may
overlap): a post-only limit at our touch ``latency_ms`` after the call, behind
``queue_fraction`` of the displayed size; filled by trades through our price or
by volume at our price exceeding the queue ahead. Unfilled after
``entry_timeout_ms`` -> no trade. Once filled, hold ``horizon``, then a limit
exit at our touch; unfilled after ``exit_timeout_ms`` -> market exit.
"""

from __future__ import annotations

import csv
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .costs import HYPERLIQUID_BASE, FeeSchedule, net_return
from .features import multilevel_imbalance
from .predictor import NAMES, BucketPredictor, direction_class, mid_direction_class, score


@dataclass
class OnlineFeatures:
    """Imbalance (L1 and 5-level) and 5 s realised volatility, matching ``features.py``."""

    vol_window_ms: int = 5_000
    _r2: deque = field(default_factory=deque)  # (t_ms, r^2)
    _sum: float = 0.0
    _last_mid: float | None = None
    values: dict = field(default_factory=dict)

    def update(self, t_ms: int, book: dict) -> None:
        bq, aq = book["bid_qty"], book["ask_qty"]
        self.values["imbalance"] = (bq - aq) / (bq + aq) if bq + aq > 0 else 0.0
        levels = sum(1 for k in book if k.startswith("bid_qty_"))
        B = np.array([[book[f"bid_qty_{i}"] for i in range(levels)]])
        A = np.array([[book[f"ask_qty_{i}"] for i in range(levels)]])
        self.values[f"imbalance_l{levels}"] = float(multilevel_imbalance(B, A, 0.5)[0])
        self.values[f"depth_imbalance_l{levels}"] = float(multilevel_imbalance(B, A, 1.0)[0])
        mid = 0.5 * (book["bid"] + book["ask"])
        if self._last_mid is not None:
            r2 = math.log(mid / self._last_mid) ** 2
            self._r2.append((t_ms, r2))
            self._sum += r2
        self._last_mid = mid
        while self._r2 and self._r2[0][0] <= t_ms - self.vol_window_ms:
            self._sum -= self._r2.popleft()[1]
        self.values["vol_5000ms_bps"] = math.sqrt(max(self._sum, 0.0)) * 1e4


@dataclass
class LimitOrder:
    side: int  # +1 buy, -1 sell
    price: float
    queue: float  # displayed size ahead of us + our size
    t_post: int
    t_cancel: int
    filled_at: int | None = None
    _at_price: float = 0.0

    def on_trade(self, ts: int, px: float, qty: float, aggressor: int) -> None:
        if self.filled_at is not None or ts <= self.t_post or ts > self.t_cancel:
            return
        if aggressor == -self.side:  # only the opposite aggressor can hit us
            if self.side * (self.price - px) > 0:  # traded through our price
                self.filled_at = ts
            elif px == self.price:
                self._at_price += qty
                if self._at_price >= self.queue:
                    self.filled_at = ts


@dataclass
class Prediction:
    t: int
    call: int
    probs: tuple[float, float, float]
    signal: float
    vol_bps: float
    mid0: float
    entry_bid: float | None = None
    entry_ask: float | None = None
    # limit-order simulation state
    entry: LimitOrder | None = None
    exit: LimitOrder | None = None
    verified: bool = False
    sim_done: bool = False  # calm calls have nothing to simulate
    sim: dict = field(default_factory=dict)


@dataclass
class DryRun:
    model: BucketPredictor
    fees: FeeSchedule = HYPERLIQUID_BASE
    every_s: float = 10.0
    latency_ms: int = 200
    min_prob: float = 0.5
    queue_fraction: float = 1.0
    entry_timeout_ms: int = 5_000
    exit_timeout_ms: int = 5_000
    order_qty: float = 0.001
    max_quote_delay_ms: int = 2_000  # drop a prediction if its entry/verify quote is staler than this
    out: Path = Path("reports/dry_run.csv")
    feats: OnlineFeatures = field(default_factory=OnlineFeatures)
    open_preds: list = field(default_factory=list)
    done: list = field(default_factory=list)
    sims: list = field(default_factory=list)
    dropped: int = 0
    bbo: tuple | None = None  # (ts, bid, bq, ask, aq)
    _next_call: int = 0
    _warm_from: int | None = None

    @property
    def horizon_ms(self) -> int:
        return self.model.horizon_ms

    # ---------------------------------------------------------------- events

    def on_book(self, book: dict) -> None:
        t = book["ts"]
        if self._warm_from is None:
            self._warm_from = t
        self.feats.update(t, book)
        if t >= self._next_call and t - self._warm_from >= self.feats.vol_window_ms:
            self._next_call = t + int(self.every_s * 1e3)
            self._predict(t, book)

    def on_bbo(self, ts: int, bid: float, bq: float, ask: float, aq: float) -> None:
        self.bbo = (ts, bid, bq, ask, aq)
        lat = self.latency_ms
        still_open = []
        for p in self.open_preds:
            if p.entry_ask is None and ts >= p.t + lat:
                if ts - (p.t + lat) > self.max_quote_delay_ms:
                    p.entry_bid = p.entry_ask = math.nan
                else:
                    p.entry_bid, p.entry_ask = bid, ask
                    if p.call != 0:
                        price, ahead = (bid, bq) if p.call > 0 else (ask, aq)
                        p.entry = LimitOrder(p.call, price, ahead * self.queue_fraction + self.order_qty, ts, ts + self.entry_timeout_ms)
            self._advance_sim(p, ts, bid, bq, ask, aq)
            if not p.sim_done and p.entry is None and p.call != 0 and p.entry_ask is not None and math.isnan(p.entry_ask):
                p.sim_done = True  # gap at entry: no simulated trade
            if not p.verified and ts >= p.t + self.horizon_ms + lat:
                p.verified = True
                late = ts - (p.t + self.horizon_ms + lat) > self.max_quote_delay_ms
                if late or p.entry_ask is None or math.isnan(p.entry_ask):
                    self.dropped += 1
                    p.sim_done = True
                else:
                    self._verify(p, ts, bid, ask)
            if not (p.verified and p.sim_done):
                still_open.append(p)
            elif p.sim:
                self.sims.append(p.sim)
                self._report_sim(p)
        self.open_preds = still_open

    def on_trade(self, ts: int, px: float, qty: float, aggressor: int) -> None:
        for p in self.open_preds:
            for o in (p.entry, p.exit):
                if o is not None:
                    o.on_trade(ts, px, qty, aggressor)

    def reset_features(self) -> None:
        """Call after a feed gap: volatility must re-warm before new calls."""
        self.feats = OnlineFeatures(self.feats.vol_window_ms)
        self._warm_from = None

    # ---------------------------------------------------------------- logic

    def _predict(self, t: int, book: dict) -> None:
        sig = self.feats.values[self.model.signal_feature]
        vol = self.feats.values[self.model.vol_feature]
        probs = self.model.predict_proba(sig, vol)[0]
        call = int(self.model.predict(sig, vol, self.min_prob)[0])
        self.open_preds.append(
            Prediction(t, call, tuple(map(float, probs)), sig, vol, 0.5 * (book["bid"] + book["ask"]), sim_done=call == 0)
        )
        print(
            f"{time.strftime('%H:%M:%S', time.localtime(t / 1e3))} predict {NAMES[call]:>4}  "
            f"P(down/calm/up)={probs[0]:.2f}/{probs[1]:.2f}/{probs[2]:.2f}  "
            f"{self.model.signal_feature}={sig:+.2f} vol5s={vol:.2f}bp",
            flush=True,
        )

    def _advance_sim(self, p: Prediction, ts: int, bid, bq, ask, aq) -> None:
        if p.sim_done or p.entry is None:
            return
        side = p.call
        if p.exit is None:
            if p.entry.filled_at is None:
                if ts > p.entry.t_cancel:
                    p.sim = {"t": p.t, "side": side, "entry_filled": False}
                    p.sim_done = True
                return
            if ts >= p.entry.filled_at + self.horizon_ms + self.latency_ms:
                price, ahead = (ask, aq) if side > 0 else (bid, bq)
                p.exit = LimitOrder(-side, price, ahead * self.queue_fraction + self.order_qty, ts, ts + self.exit_timeout_ms)
            return
        if p.exit.filled_at is not None:
            exit_px, fee_exit, exit_type = p.exit.price, self.fees.maker, "limit"
        elif ts > p.exit.t_cancel + self.latency_ms:
            exit_px, fee_exit, exit_type = (bid if side > 0 else ask), self.fees.taker, "market_fallback"
        else:
            return
        net = float(net_return(side, p.entry.price, exit_px, self.fees.maker, fee_exit))
        p.sim = {
            "t": p.t,
            "side": side,
            "entry_filled": True,
            "entry_px": p.entry.price,
            "exit_px": exit_px,
            "exit_type": exit_type,
            "gross_bps": float(net_return(side, p.entry.price, exit_px, 0.0, 0.0)) * 1e4,
            "net_bps": net * 1e4,
        }
        p.sim_done = True

    def _verify(self, p: Prediction, ts: int, bid: float, ask: float) -> None:
        mid_move = 0.5 * (bid + ask) / p.mid0 - 1
        long_ret = bid / p.entry_ask - 1
        short_ret = 1 - ask / p.entry_bid
        if self.model.label == "mid":
            actual = int(mid_direction_class(mid_move, self.model.threshold))
        else:
            actual = int(direction_class(long_ret, short_ret, self.model.threshold))
        row = {
            "t_predict_ms": p.t,
            "t_verify_ms": ts,
            "call": NAMES[p.call],
            "actual": NAMES[actual],
            "correct": p.call == actual,
            "p_down": round(p.probs[0], 4),
            "p_calm": round(p.probs[1], 4),
            "p_up": round(p.probs[2], 4),
            "signal": round(p.signal, 4),
            "vol5s_bps": round(p.vol_bps, 3),
            "mid_move_bps": round(mid_move * 1e4, 3),
        }
        self.done.append((p, actual, mid_move))
        new = not self.out.exists()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        self._report(row)

    def _report(self, row: dict) -> None:
        preds = np.array([d[0].call for d in self.done])
        actual = np.array([d[1] for d in self.done])
        s = score(preds, actual)
        lean = np.array([d[0].probs[2] - d[0].probs[0] for d in self.done])
        mv = np.array([d[2] for d in self.done])
        moved = (mv != 0) & (lean != 0)
        dir_hit = float(((lean[moved] > 0) == (mv[moved] > 0)).mean()) if moved.any() else float("nan")
        print(
            f"   verify {row['call']:>4} -> actual {row['actual']:>4}  mid {row['mid_move_bps']:+.2f}bp | "
            f"n={s['n']} acc={s['accuracy']:.1%} (always-calm {s['always_calm_accuracy']:.1%}) "
            f"up {s['up_calls']}/{s['up_precision']:.0%} down {s['down_calls']}/{s['down_precision']:.0%} "
            f"raw-direction hit={dir_hit:.1%} dropped={self.dropped}",
            flush=True,
        )

    def _report_sim(self, p: Prediction) -> None:
        filled = [s for s in self.sims if s["entry_filled"]]
        tot = sum(s["net_bps"] for s in filled)
        last = p.sim
        detail = f"{last['exit_type']} net={last['net_bps']:+.2f}bp" if last["entry_filled"] else "entry not filled"
        print(
            f"   sim {NAMES[last['side']]:>4}: {detail} | limit entries filled {len(filled)}/{len(self.sims)}, "
            f"avg net {f'{tot / len(filled):+.2f}bp' if filled else 'n/a'}, total {tot:+.1f}bp",
            flush=True,
        )


async def run_dry(coin: str, dry: DryRun, duration_s: float | None = None) -> None:
    from .data.hyperliquid import parse_bbo, parse_book, parse_trades, stream

    t_stop = time.time() + duration_s if duration_s else None
    async for _, ch, data in stream([coin]):
        if ch == "l2Book":
            book = parse_book(data)
            if book:
                dry.on_book(book)
        elif ch == "bbo":
            q = parse_bbo(data)
            if q:
                dry.on_bbo(*q)
        elif ch == "trades":
            for ts, _, px, qty, side in parse_trades(data):
                dry.on_trade(ts, px, qty, side)
        elif ch == "_disconnect":
            # Open predictions survive short gaps; on_bbo drops any whose entry
            # or verification quote fell inside the gap.
            dry.reset_features()
        if t_stop and time.time() >= t_stop:
            return
