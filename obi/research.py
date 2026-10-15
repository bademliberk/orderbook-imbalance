"""End-to-end research pipeline on Hyperliquid data: load -> validate -> features
-> labels -> analysis -> walk-forward strategy selection -> out-of-sample test
and stress tests.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import ExecConfig, run_backtest
from .data.tardis import file_path, load_book, load_trades
from .features import asof_index, build_features
from .labels import executable_labels
from .metrics import summarize
from .strategy import LinearForecaster, momentum_signal, threshold_signal
from .validate import ValidationReport, clean_top_of_book

# Book snapshots arrive every ~0.54 s, so sub-second horizons are not resolvable.
HORIZONS = (1_000, 5_000, 30_000, 60_000, 300_000)
EXCURSION_HORIZONS = (30_000, 60_000)
MODEL_FEATURES = [
    "imbalance",
    "imbalance_l5",
    "depth_imbalance_l5",
    "microprice_dev_bps",
    "spread_bps",
    "log_depth",
    "ofi_5000ms",
    "ofi_30000ms",
    "ret_5000ms_bps",
    "ret_30000ms_bps",
    "ret_120000ms_bps",
    "vol_5000ms_bps",
    "vol_30000ms_bps",
    "trade_imb_5000ms",
    "trade_imb_30000ms",
    "trade_count_5000ms",
]


@dataclass
class DayData:
    coin: str
    day: date
    book: pd.DataFrame
    trades: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame | None
    sample_idx: np.ndarray
    validation: ValidationReport


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare_day(coin: str, day: date, raw_dir: Path, latency_ms: int, sample_ms: int = 1_000) -> DayData:
    book = load_book(file_path(raw_dir, "book_snapshot_25", coin, day))
    trades = load_trades(file_path(raw_dir, "trades", coin, day))
    book, rep = clean_top_of_book(book)
    feats = build_features(book, trades).astype(np.float32)
    labels = executable_labels(book, HORIZONS, latency_ms=latency_ms, with_excursions=False)
    exc = executable_labels(book, EXCURSION_HORIZONS, latency_ms=latency_ms, with_excursions=True)
    labels = pd.concat([labels, exc.filter(regex="_(mfe|mae)_")], axis=1)
    # Clock-time sample (one row per sample_ms) for analysis and model fitting.
    ts = book["ts"].to_numpy()
    grid = np.arange(ts[0], ts[-1], sample_ms)
    idx = np.unique(asof_index(ts, grid))
    idx = idx[book["valid"].to_numpy()[idx]]
    return DayData(coin, day, book, trades, feats, labels, idx, rep)


def sample_frame(d: DayData) -> pd.DataFrame:
    s = pd.concat([d.features.iloc[d.sample_idx], d.labels.iloc[d.sample_idx]], axis=1)
    s.insert(0, "day", d.day.isoformat())
    s.insert(0, "coin", d.coin)
    return s.reset_index(drop=True)


# ---------------------------------------------------------------- strategies


@dataclass(frozen=True)
class RuleSpec:
    kind: str  # "imbalance" | "imbalance_l5" | "momentum" | "linear"
    threshold: float
    hold_ms: int
    target: float | None = None
    stop: float | None = None

    def label(self) -> str:
        ex = "" if self.target is None else f" tgt={self.target * 1e4:.0f}bp stop={self.stop * 1e4:.0f}bp"
        return f"{self.kind} th={self.threshold:g} hold={self.hold_ms // 1000}s{ex}"


def make_signal(spec: RuleSpec, d: DayData, model: LinearForecaster | None, round_trip_fee: float) -> np.ndarray:
    f = d.features
    if spec.kind in ("imbalance", "imbalance_l5"):
        return threshold_signal(f[spec.kind].to_numpy(), spec.threshold)
    if spec.kind == "momentum":
        return momentum_signal(f["ret_30000ms_bps"].to_numpy(), spec.threshold)
    if spec.kind == "linear":
        assert model is not None
        return model.signal(f, round_trip_fee=round_trip_fee, margin=spec.threshold)
    raise ValueError(spec.kind)


def round_trip_fee(cfg: ExecConfig) -> float:
    return getattr(cfg.fees, cfg.entry_mode) + getattr(cfg.fees, cfg.exit_mode)


def backtest_spec(spec: RuleSpec, d: DayData, cfg: ExecConfig, model=None) -> tuple[pd.DataFrame, dict]:
    sig = make_signal(spec, d, model, round_trip_fee(cfg))
    cfg = replace(cfg, max_hold_ms=spec.hold_ms, target=spec.target, stop=spec.stop)
    tr, st = run_backtest(d.book, sig, cfg, trades=d.trades)
    if not tr.empty:
        tr.insert(0, "day", d.day.isoformat())
    return tr, st


def merge_stats(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def run_specs(specs, days: list[DayData], cfg: ExecConfig, model=None) -> pd.DataFrame:
    rows = []
    for spec in specs:
        trs, st = [], {}
        for d in days:
            tr, s = backtest_spec(spec, d, cfg, model)
            trs.append(tr)
            st = merge_stats(st, s)
        tr = pd.concat(trs, ignore_index=True) if any(len(t) for t in trs) else pd.DataFrame()
        m = summarize(tr, st, days=len(days))
        m["spec"] = spec.label()
        rows.append(m)
    return pd.DataFrame(rows)


def imbalance_grid() -> list[RuleSpec]:
    specs = []
    for kind, th, hold in itertools.product(("imbalance", "imbalance_l5"), (0.5, 0.8), (5_000, 30_000, 60_000, 300_000)):
        specs.append(RuleSpec(kind, th, hold))
    for kind, th in itertools.product(("imbalance", "imbalance_l5"), (0.5, 0.8)):
        specs.append(RuleSpec(kind, th, 300_000, target=0.0005, stop=-0.0020))
    return specs


def momentum_grid() -> list[RuleSpec]:
    return [RuleSpec("momentum", th, hold) for th, hold in itertools.product((2.0, 5.0), (30_000, 300_000))]


def linear_grid() -> list[tuple[int, list[RuleSpec]]]:
    return [(h, [RuleSpec("linear", m, h) for m in (0.0, 0.0001, 0.0003)]) for h in (30_000, 60_000, 300_000)]


def pick_best(results: pd.DataFrame, min_trades: int) -> pd.Series | None:
    ok = results[results["trades"] >= min_trades]
    if ok.empty:
        return None
    return ok.sort_values("net_pnl", ascending=False).iloc[0]
