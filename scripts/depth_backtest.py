"""Backtest order book depth imbalance within n% of mid (e.g. 1%) on live recordings.

Needs data from ``scripts/record_live.py --agg-sig-figs 4 3`` (aggregated
20-level books reach ~0.25% / ~2.4% from mid on BTC). History from Tardis has
only 5 levels and cannot measure these bands.

The recording is split chronologically: the first ``--train-frac`` chooses the
signal direction (follow or fade the deep book, by the sign of its train IC) and
the thresholds (quantiles of |signal| on train); the rest is the test.

Example:
    python scripts/depth_backtest.py --coins BTC ETH SOL --band-bps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obi import costs  # noqa: E402
from obi.backtest import ExecConfig, run_backtest  # noqa: E402
from obi.data.hyperliquid import load_recorded_book, load_recorded_trades  # noqa: E402
from obi.features import DEPTH_BANDS_BPS, build_features, merge_depth_bands  # noqa: E402
from obi.labels import executable_labels  # noqa: E402
from obi.metrics import summarize  # noqa: E402
from obi.signal_study import spearman  # noqa: E402
from obi.validate import clean_top_of_book  # noqa: E402

HORIZONS = (30_000, 60_000, 300_000)
MODES = {
    "market/market": dict(entry_mode="taker", exit_mode="taker"),
    "limit/limit": dict(entry_mode="maker", exit_mode="maker"),
    "limit/market": dict(entry_mode="maker", exit_mode="taker"),
}


def load_coin(root: Path, coin: str):
    files = lambda ch: sorted((root / ch / coin).glob("*.jsonl.gz"))  # noqa: E731
    book, rep = clean_top_of_book(load_recorded_book(files("l2Book"), levels=5))
    trades = load_recorded_trades(files("trades"))
    agg = {}
    for d in sorted(root.glob("l2Book_sf*")):
        n = int(d.name.removeprefix("l2Book_sf"))
        fs = files(d.name)
        if fs:
            agg[n] = load_recorded_book(fs, levels=20)
    return book, trades, agg, rep


def crosses_gap(ts: np.ndarray, window_ms: float, max_gap_ms: int = 5_000) -> np.ndarray:
    """True where ``[ts, ts + window_ms]`` spans a feed gap (e.g. laptop asleep).

    Labels and trades over such windows would use a price from after the gap,
    so they are excluded.
    """
    last_before_gap = ts[np.flatnonzero(np.diff(ts) > max_gap_ms)]
    if len(last_before_gap) == 0:
        return np.zeros(len(ts), dtype=bool)
    k = np.searchsorted(last_before_gap, ts, side="left")
    nxt = np.where(k < len(last_before_gap), last_before_gap[np.minimum(k, len(last_before_gap) - 1)], np.inf)
    return ts + window_ms > nxt


def md(df, floatfmt=".3g", index=False):
    return df.to_markdown(floatfmt=floatfmt, index=index)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/live"))
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH", "SOL", "HYPE", "DOGE", "PUMP", "FARTCOIN"])
    p.add_argument("--band-bps", type=int, default=100, help="band used for trading (100 = 1%%)")
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.99])
    p.add_argument("--latency-ms", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("reports/depth_backtest.md"))
    a = p.parse_args()
    col = f"depth_imb_{a.band_bps}bps"
    sec = [f"# Depth-band backtest — imbalance within {a.band_bps / 100:g}% of mid (live recordings)", ""]
    ic_rows, bt_rows = [], []

    for coin in a.coins:
        book, trades, agg, rep = load_coin(a.data, coin)
        if not agg or len(book) < 1000:
            print(f"{coin}: not enough data (book rows={len(book)}, agg feeds={list(agg)})")
            continue
        ts = book["ts"].to_numpy(np.int64)
        dts = np.diff(ts)
        hours = dts[dts <= 5_000].sum() / 3.6e6  # recorded time, excluding gaps
        feats = build_features(book, trades)
        depth = merge_depth_bands(ts, agg)
        feats = pd.concat([feats.reset_index(drop=True), depth], axis=1)
        labels = executable_labels(book, HORIZONS, latency_ms=a.latency_ms, with_excursions=False)
        for h in HORIZONS:
            labels.loc[crosses_gap(ts, h + a.latency_ms), [c for c in labels.columns if c.endswith(f"_{h}")]] = np.nan
        split_t = ts[0] + a.train_frac * (ts[-1] - ts[0])
        is_train = ts < split_t
        print(f"{coin}: {hours:.1f} h, {len(book):,} book rows, agg feeds {sorted(agg)}, {rep.summary()}")

        # 1 s sample for IC (avoid over-weighting bursts)
        grid = np.unique(np.searchsorted(ts, np.arange(ts[0], ts[-1], 1_000), side="right") - 1)
        for part, mask in (("train", is_train[grid]), ("test", ~is_train[grid])):
            g = grid[mask]
            row = {"coin": coin, "period": part, "samples_1s": int(mask.sum())}
            for b in DEPTH_BANDS_BPS:
                row[f"cov_{b}bps"] = float(feats[f"depth_cov_{b}bps"].iloc[g].mean())
            for h in (60_000, 300_000):
                y = labels[f"mid_ret_{h}"].iloc[g]
                row[f"IC_l5_{h // 1000}s"] = spearman(feats["imbalance_l5"].iloc[g], y)
                for b in DEPTH_BANDS_BPS:
                    row[f"IC_{b}bps_{h // 1000}s"] = spearman(feats[f"depth_imb_{b}bps"].iloc[g], y)
            ic_rows.append(row)

        # direction and thresholds from train only
        g_tr = grid[is_train[grid]]
        direction = np.sign(spearman(feats[col].iloc[g_tr], labels["mid_ret_60000"].iloc[g_tr])) or 1.0
        x = feats[col].to_numpy() * direction
        qty = 1_000.0 / float(np.median(book["bid"]))
        test_book = book[~is_train].reset_index(drop=True)
        test_trades = trades[trades["ts"] >= split_t].reset_index(drop=True)
        for q in a.quantiles:
            thr = float(np.quantile(np.abs(x[is_train]), q))
            sig_all = np.where(x > thr, 1, np.where(x < -thr, -1, 0))
            for hold in (60_000, 300_000):
                # no position may be open across a feed gap
                max_life = hold + 3 * a.latency_ms + 2 * ExecConfig().maker_timeout_ms
                sig = np.where(crosses_gap(ts, max_life), 0, sig_all)[~is_train]
                for mode, kw in MODES.items():
                    for fee_name, fees in (("tier 0", costs.HYPERLIQUID_BASE), ("zero fees", costs.ZERO_FEES)):
                        if fee_name == "zero fees" and mode != "market/market":
                            continue
                        cfg = ExecConfig(fees=fees, latency_ms=a.latency_ms, order_qty=qty, max_hold_ms=hold, **kw)
                        tr, st = run_backtest(test_book, sig, cfg, trades=test_trades)
                        m = summarize(tr, st)
                        bt_rows.append(
                            {
                                "coin": coin,
                                "direction": "follow" if direction > 0 else "fade",
                                "top": f"{1 - q:.0%}",
                                "hold_s": hold // 1000,
                                "mode": mode,
                                "fees": fee_name,
                                "trades": m["trades"],
                                "net_bps": m.get("net_bps_per_trade", np.nan),
                                "gross_bps": m.get("gross_bps_per_trade", np.nan),
                                "tstat": m.get("tstat", np.nan),
                                "fill_rate": m.get("fill_rate", np.nan),
                                "exit_fallback": m.get("exit_fallback_rate", np.nan),
                            }
                        )

    if not ic_rows:
        sys.exit("no coin had enough data")
    ic = pd.DataFrame(ic_rows)
    sec.append("## Coverage and rank IC vs forward mid move (1 s samples)\n\n`cov_*` = share of time the aggregated books reach the band on both sides. `IC_l5` = 5-level imbalance for comparison.\n\n" + md(ic))
    bt = pd.DataFrame(bt_rows)
    bt.to_csv(a.out.with_suffix(".csv"), index=False)
    sec.append(
        f"## Backtest on the test segment — trade when |`{col}`| is in the top X% of train values\n\n"
        "Direction (follow = bid-heavy deep book -> long; fade = the opposite) was chosen on train. "
        "Tier-0 fees: limit 0.015%, market 0.045%. `zero fees` rows show the raw edge.\n\n" + md(bt)
    )
    a.out.write_text("\n\n".join(sec) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
