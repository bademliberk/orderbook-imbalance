"""Data-quality checks for book snapshots (plan step 1).

Hyperliquid book messages are complete snapshots without sequence numbers, so
data loss is detected from time gaps between snapshots (normally ~0.54 s).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    rows: int
    duplicate_rows: int
    non_monotonic_ts: int
    crossed_or_locked: int
    non_positive_qty: int
    time_gaps: list[tuple[int, int]] = field(default_factory=list)
    invalid_rows: int = 0

    def summary(self) -> str:
        gap_ms = sum(b - a for a, b in self.time_gaps)
        return (
            f"rows={self.rows:,} dup={self.duplicate_rows:,} ts_backwards={self.non_monotonic_ts:,} "
            f"crossed/locked={self.crossed_or_locked:,} bad_qty={self.non_positive_qty:,} "
            f"gaps={len(self.time_gaps)} ({gap_ms / 1000:.1f}s) invalid={self.invalid_rows:,}"
        )


def clean_top_of_book(df: pd.DataFrame, max_gap_ms: int = 5_000) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a top-of-book frame.

    Returns the cleaned frame with an extra boolean column ``valid``. Rows are
    kept (so the book state stays continuous) but flagged invalid when the
    quote is crossed/locked, has non-positive size, or follows a feed silence
    longer than ``max_gap_ms`` (the book may have been stale before it).
    """
    n0 = len(df)
    # Input arrives in feed order; count timestamps that go backwards before sorting.
    backwards = int(np.sum(np.diff(df["ts"].to_numpy()) < 0))
    df = df.drop_duplicates(subset=["ts", "bid", "bid_qty", "ask", "ask_qty"], keep="last")
    dup = n0 - len(df)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    ts = df["ts"].to_numpy()

    crossed = (df["bid"] >= df["ask"]).to_numpy()
    bad_qty = ((df["bid_qty"] <= 0) | (df["ask_qty"] <= 0)).to_numpy()

    dts = np.diff(ts, prepend=ts[0])
    gap_idx = np.flatnonzero(dts > max_gap_ms)
    gaps = [(int(ts[i - 1]), int(ts[i])) for i in gap_idx]
    after_gap = np.zeros(len(df), dtype=bool)
    after_gap[gap_idx] = True

    valid = ~(crossed | bad_qty | after_gap)
    df["valid"] = valid
    report = ValidationReport(
        rows=n0,
        duplicate_rows=dup,
        non_monotonic_ts=backwards,
        crossed_or_locked=int(crossed.sum()),
        non_positive_qty=int(bad_qty.sum()),
        time_gaps=gaps,
        invalid_rows=int((~valid).sum()),
    )
    return df, report
