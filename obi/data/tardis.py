"""Hyperliquid history from Tardis.dev.

Tardis serves the first day of every month without an API key; other days need
``TARDIS_API_KEY``. Datasets used:

- ``book_snapshot_25``: book snapshots derived from Hyperliquid's ``l2Book``
  feed, about every 0.54 s. We load 5 levels to match the live
  ``l2Book`` ``fast`` feed, so research and live features are identical.
- ``trades``: every public trade with the aggressor side.

Timestamps are microseconds in the files and converted to milliseconds here.
"""

from __future__ import annotations

import os
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://datasets.tardis.dev/v1/hyperliquid"
DEPTH_LEVELS = 25


def first_of_months(start: date, end: date) -> list[date]:
    """First day of every month in ``[start, end]`` (the free Tardis days)."""
    out, y, m = [], start.year, start.month
    while date(y, m, 1) <= end:
        if date(y, m, 1) >= start:
            out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def file_path(root: Path, dataset: str, coin: str, day: date) -> Path:
    return Path(root) / dataset / coin / f"{day.isoformat()}.csv.gz"


def download(dataset: str, coin: str, day: date, root: Path) -> Path:
    path = file_path(root, dataset, coin, day)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{dataset}/{day:%Y/%m/%d}/{coin}.csv.gz"
    req = urllib.request.Request(url)
    key = os.environ.get("TARDIS_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    elif day.day != 1:
        raise ValueError(f"{day} is not a free day; set TARDIS_API_KEY")
    with urllib.request.urlopen(req, timeout=300) as r:
        payload = r.read()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(payload)
    tmp.rename(path)
    return path


def load_book(path: Path, levels: int = 5) -> pd.DataFrame:
    """Canonical book frame from ``book_snapshot_25``.

    Columns: ``ts`` (exchange time, ms), ``recv_ts`` (collector receive time,
    ms), ``bid, bid_qty, ask, ask_qty`` (best level) and
    ``bid_px_i, bid_qty_i, ask_px_i, ask_qty_i`` for ``i < levels``.
    """
    cols = ["timestamp", "local_timestamp"]
    for i in range(levels):
        cols += [f"asks[{i}].price", f"asks[{i}].amount", f"bids[{i}].price", f"bids[{i}].amount"]
    raw = pd.read_csv(path, usecols=cols, engine="pyarrow")
    out = {
        "ts": (raw["timestamp"].to_numpy(np.int64) // 1000),
        "recv_ts": (raw["local_timestamp"].to_numpy(np.int64) // 1000),
    }
    for i in range(levels):
        out[f"bid_px_{i}"] = raw[f"bids[{i}].price"].to_numpy(np.float64)
        out[f"bid_qty_{i}"] = raw[f"bids[{i}].amount"].to_numpy(np.float64)
        out[f"ask_px_{i}"] = raw[f"asks[{i}].price"].to_numpy(np.float64)
        out[f"ask_qty_{i}"] = raw[f"asks[{i}].amount"].to_numpy(np.float64)
    df = pd.DataFrame(out)
    df.insert(2, "bid", df["bid_px_0"])
    df.insert(3, "bid_qty", df["bid_qty_0"])
    df.insert(4, "ask", df["ask_px_0"])
    df.insert(5, "ask_qty", df["ask_qty_0"])
    # Levels missing in a thin book come through as NaN: treat as zero size.
    qty_cols = [c for c in df.columns if "_qty" in c]
    df[qty_cols] = df[qty_cols].fillna(0.0)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def load_trades(path: Path) -> pd.DataFrame:
    """Columns: ``ts`` (ms), ``price``, ``qty``, ``side`` (+1 buy aggressor, -1 sell)."""
    raw = pd.read_csv(path, usecols=["timestamp", "side", "price", "amount"], engine="pyarrow")
    df = pd.DataFrame(
        {
            "ts": raw["timestamp"].to_numpy(np.int64) // 1000,
            "price": raw["price"].to_numpy(np.float64),
            "qty": raw["amount"].to_numpy(np.float64),
            "side": np.where(raw["side"].to_numpy() == "buy", 1, -1).astype(np.int8),
        }
    )
    return df.sort_values("ts", kind="stable").reset_index(drop=True)

