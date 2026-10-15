"""Live Hyperliquid market data: websocket stream, recorder and loaders.

Channels:

- ``l2Book`` with ``fast: true``: 5-level snapshots every ~0.54 s. This matches
  the cadence of the Tardis history used for research, so live features are
  computed from it. (Without ``fast`` the feed gives 20 levels but was observed
  to arrive only every ~5 s.)
- ``bbo``: best bid/offer on every block where it changes (~130 ms); used for
  fill and verification prices.
- ``trades``: every public trade; used for limit-order fill simulation.

Snapshots are complete, so there is no sequence/diff synchronisation to get
wrong: a reconnect only loses the data during the outage.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time
import urllib.request
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import pandas as pd

WS_URL = "wss://api.hyperliquid.xyz/ws"
INFO_URL = "https://api.hyperliquid.xyz/info"
PING_EVERY_S = 30  # the server drops connections idle for 60 s
LIVE_LEVELS = 5
SUBSCRIPTION_EXTRA = {"l2Book": {"fast": True}}


def info(payload: dict) -> dict:
    req = urllib.request.Request(
        INFO_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


async def stream(
    coins: list[str], channels=("l2Book", "bbo", "trades"), extra: dict | None = None
) -> AsyncIterator[tuple[int, str, dict | list]]:
    """Yield ``(recv_ns, channel, data)`` forever, reconnecting on errors.

    ``extra`` overrides the per-channel subscription parameters (default
    ``SUBSCRIPTION_EXTRA``), e.g. ``{"l2Book": {"fast": True, "nSigFigs": 3}}``.

    A ``("_disconnect", {})`` event is yielded after each drop so consumers
    can invalidate state that depends on continuous data.
    """
    import websockets

    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=60, max_size=2**24) as ws:
                for c in coins:
                    for ch in channels:
                        sub = {"type": ch, "coin": c, **(extra or SUBSCRIPTION_EXTRA).get(ch, {})}
                        await ws.send(json.dumps({"method": "subscribe", "subscription": sub}))

                async def keepalive():
                    while True:
                        await asyncio.sleep(PING_EVERY_S)
                        await ws.send(json.dumps({"method": "ping"}))

                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} connected: {list(coins)} {list(channels)}", flush=True)
                ka = asyncio.create_task(keepalive())
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        ch = msg.get("channel")
                        if ch in channels:
                            yield time.time_ns(), ch, msg["data"]
                finally:
                    ka.cancel()
        except Exception as exc:  # network errors (offline, sleep/wake, server drop): reconnect
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} stream error: {exc!r}; reconnecting", flush=True)
        yield time.time_ns(), "_disconnect", {}
        await asyncio.sleep(1)


def parse_book(data: dict, levels: int = LIVE_LEVELS) -> dict | None:
    """One ``l2Book`` message -> flat row matching ``tardis.load_book`` columns."""
    bids, asks = data["levels"]
    if not bids or not asks:
        return None
    row = {"ts": int(data["time"])}
    for i in range(levels):
        b = bids[i] if i < len(bids) else None
        a = asks[i] if i < len(asks) else None
        row[f"bid_px_{i}"] = float(b["px"]) if b else np.nan
        row[f"bid_qty_{i}"] = float(b["sz"]) if b else 0.0
        row[f"ask_px_{i}"] = float(a["px"]) if a else np.nan
        row[f"ask_qty_{i}"] = float(a["sz"]) if a else 0.0
    row["bid"], row["bid_qty"] = row["bid_px_0"], row["bid_qty_0"]
    row["ask"], row["ask_qty"] = row["ask_px_0"], row["ask_qty_0"]
    return row


def parse_bbo(data: dict) -> tuple[int, float, float, float, float] | None:
    """``bbo`` message -> (ts, bid, bid_qty, ask, ask_qty)."""
    b, a = data["bbo"]
    if not b or not a:
        return None
    return int(data["time"]), float(b["px"]), float(b["sz"]), float(a["px"]), float(a["sz"])


def parse_trades(data: list) -> list[tuple[int, str, float, float, int]]:
    """``trades`` message -> (ts, coin, price, qty, side); side +1 = buyer aggressed.

    Hyperliquid's ``side`` is the taker side: "B" (bid) means the taker bought.
    """
    return [(int(t["time"]), t["coin"], float(t["px"]), float(t["sz"]), 1 if t["side"] == "B" else -1) for t in data]


class HourlyWriter:
    """Hourly gzip JSONL files: ``<channel>/<coin>/<YYYYmmddHH>_<run>.jsonl.gz``.

    Each recorder run writes its own files (``run`` = start time), so a
    restart never appends to a file whose last block may have been cut off by
    a crash or power loss. Loaders read every file in time order.
    """

    FLUSH_EVERY_S = 10  # sync-flush so files are readable while being written

    def __init__(self, root: Path, channel: str, coin: str, run: str):
        self.dir = Path(root) / channel / coin
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.hour = None
        self.fh = None
        self._last_flush = 0.0

    def write(self, record: dict) -> None:
        hour = time.strftime("%Y%m%d%H", time.gmtime(record["recv_ns"] / 1e9))
        if hour != self.hour:
            if self.fh:
                self.fh.close()
            self.fh = gzip.open(self.dir / f"{hour}_{self.run}.jsonl.gz", "at")
            self.hour = hour
        self.fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        now = time.monotonic()
        if now - self._last_flush >= self.FLUSH_EVERY_S:
            self.fh.flush()
            self._last_flush = now

    def close(self) -> None:
        if self.fh:
            self.fh.close()


async def record(coins: list[str], out: Path, duration_s: float | None = None, agg_sig_figs: tuple[int, ...] = ()) -> None:
    """Record raw ``l2Book``, ``bbo`` and ``trades`` messages with local receive times.

    ``agg_sig_figs`` adds aggregated 20-level books (``nSigFigs``, ~5 s cadence)
    on separate connections, written as channel ``l2Book_sf<n>``. Price levels
    are bucketed to ``n`` significant figures, so they reach far deeper than the
    full-resolution book (BTC: nSigFigs=4 ~0.25%, 3 ~2.4% from mid), which is
    what "depth within n% of mid" features need. Aggregated messages do not say
    which aggregation they carry, hence one connection each.
    """
    writers: dict[tuple[str, str], HourlyWriter] = {}
    run = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    print(f"recorder run {run} started: coins={coins} agg_sig_figs={list(agg_sig_figs)} out={out}", flush=True)

    def w(ch, coin):
        if (ch, coin) not in writers:
            writers[(ch, coin)] = HourlyWriter(out, ch, coin, run)
        return writers[(ch, coin)]

    async def consume(src, book_channel: str):
        async for recv_ns, ch, data in src:
            if ch in ("l2Book", "bbo"):
                w(book_channel if ch == "l2Book" else ch, data["coin"]).write({"recv_ns": recv_ns, "data": data})
            elif ch == "trades" and data:
                w(ch, data[0]["coin"]).write({"recv_ns": recv_ns, "data": data})
            elif ch == "_disconnect":
                for coin in coins:
                    w(book_channel, coin).write({"recv_ns": recv_ns, "gap": True})

    tasks = [consume(stream(coins), "l2Book")]
    for n in agg_sig_figs:
        tasks.append(consume(stream(coins, ("l2Book",), {"l2Book": {"nSigFigs": n}}), f"l2Book_sf{n}"))
    try:
        if duration_s:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=duration_s)
        else:
            await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        pass
    finally:
        for x in writers.values():
            x.close()


def _records(paths):
    for p in sorted(paths):
        try:
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        break  # partial last line of a file still being written
        except (EOFError, gzip.BadGzipFile):
            continue  # file still being written: use what was readable


def load_recorded_book(paths, levels: int = LIVE_LEVELS) -> pd.DataFrame:
    """Recorded ``l2Book`` files -> the canonical book frame (plus ``recv_ts``)."""
    rows = []
    for rec in _records(paths):
        if rec.get("gap"):
            continue
        row = parse_book(rec["data"], levels)
        if row:
            row["recv_ts"] = rec["recv_ns"] // 1_000_000
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["ts", "bid", "bid_qty", "ask", "ask_qty", "recv_ts"])
    return pd.DataFrame(rows).sort_values("ts", kind="stable").reset_index(drop=True)


def load_recorded_trades(paths) -> pd.DataFrame:
    rows = [t for rec in _records(paths) for t in parse_trades(rec["data"])]
    df = pd.DataFrame(rows, columns=["ts", "coin", "price", "qty", "side"]).drop(columns="coin")
    return df.sort_values("ts", kind="stable").reset_index(drop=True)
