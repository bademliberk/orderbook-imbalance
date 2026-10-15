"""Feature-discovery study: price-change vector x order book depth x spread.

Builds 1 s study frames for each coin (base features + price vector + band
depth + dynamics, with forward labels), then runs the analyses in
``obi.signal_study`` and writes reports/signal_study/results.md.

Example:
    python scripts/signal_study.py --coins BTC ETH SOL HYPE DOGE AVAX ENA APT PUMP FARTCOIN PENGU kPEPE
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obi.data.tardis import file_path, load_book, load_trades  # noqa: E402
from obi.features import asof_index, build_study_features  # noqa: E402
from obi.labels import executable_labels  # noqa: E402
from obi.research import log  # noqa: E402
from obi.signal_study import (  # noqa: E402
    ALL,
    family_importance,
    ic_table,
    interaction_table,
    model_comparison,
    partial_ic,
)
from obi.validate import clean_top_of_book  # noqa: E402

HORIZONS = (5_000, 30_000, 60_000, 300_000)
TRAIN = ["2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01"]
TEST = ["2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01"]
GROUPS = {
    "majors": ["BTC", "ETH", "SOL", "HYPE"],
    "mid": ["DOGE", "AVAX", "ENA", "APT"],
    "small": ["PUMP", "FARTCOIN", "PENGU", "kPEPE"],
}


def study_frame(coin: str, day: str, raw: Path, latency_ms: int) -> pd.DataFrame:
    d = date.fromisoformat(day)
    book, _ = clean_top_of_book(load_book(file_path(raw, "book_snapshot_25", coin, d)))
    trades = load_trades(file_path(raw, "trades", coin, d))
    f = build_study_features(book, trades)
    lab = executable_labels(book, HORIZONS, latency_ms=latency_ms, with_excursions=False)
    ts = book["ts"].to_numpy()
    idx = np.unique(asof_index(ts, np.arange(ts[0] + 300_000, ts[-1], 1_000)))  # skip warm-up of 5 min windows
    idx = idx[book["valid"].to_numpy()[idx]]
    out = pd.concat([f.iloc[idx], lab.iloc[idx]], axis=1).reset_index(drop=True)
    out.insert(0, "day", day)
    out.insert(0, "coin", coin)
    return out


def md(df, floatfmt=".3g", index=True) -> str:
    return df.to_markdown(floatfmt=floatfmt, index=index)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coins", nargs="+", default=sum(GROUPS.values(), []))
    p.add_argument("--raw", type=Path, default=Path("data/tardis"))
    p.add_argument("--out", type=Path, default=Path("reports/signal_study"))
    p.add_argument("--latency-ms", type=int, default=200)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ics = {h: {} for h in HORIZONS}
    icirs, partials, models, importances, inter, coverage = {}, {}, [], {}, {}, {}
    for coin in a.coins:
        log(f"=== {coin}: building frames")
        cache = a.out / f"{coin}_study.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            df = pd.concat([study_frame(coin, d, a.raw, a.latency_ms) for d in TRAIN + TEST], ignore_index=True)
            df.to_parquet(cache)
        train, test = df[df["day"].isin(TRAIN)], df[df["day"].isin(TEST)]
        coverage[coin] = {f"{n}bps": train[f"band_cov_{n}bps"].mean() for n in (1, 2, 5, 10)}
        log(f"{coin}: IC tables")
        for h in HORIZONS:
            t = ic_table(train, ALL, f"mid_ret_{h}")
            ics[h][coin] = t["IC"]
            if h == 60_000:
                icirs[coin] = t["ICIR"]
        partials[coin] = partial_ic(train, ALL, "mid_ret_60000")
        inter[coin] = interaction_table(train, "mid_ret_60000")
        log(f"{coin}: models")
        for h in (60_000, 300_000):
            res, gbm = model_comparison(train, test, h)
            res.insert(0, "horizon_s", h // 1000)
            res.insert(0, "coin", coin)
            models.append(res)
            if h == 60_000:
                importances[coin] = family_importance(gbm, test, h)

    def group_means(table: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({g: table[[c for c in cs if c in table]].mean(axis=1) for g, cs in GROUPS.items() if any(c in table for c in cs)})
        out["all"] = table.mean(axis=1)
        return out

    sec = ["# Signal study: price-change vector × order book depth × spread (Hyperliquid)", ""]
    sec.append(
        f"Coins: {', '.join(a.coins)}. 1 s samples; train {TRAIN[0]}…{TRAIN[-1]} (8 monthly days), "
        f"untouched test {TEST[0]}…{TEST[-1]} (4). Forward returns are mid-to-mid unless noted; "
        f"costs to beat: 3 bps (limit/limit, *if both fill*) and 9 bps (market/market)."
    )
    sec.append("## Book coverage of the n-bps bands (share of rows where 5 visible levels reach the band)\n\n" + md(pd.DataFrame(coverage).T))

    ic60 = pd.DataFrame(ics[60_000])
    g60 = group_means(ic60)
    g60["ICIR_all"] = pd.DataFrame(icirs).mean(axis=1)
    g60 = g60.reindex(g60["all"].abs().sort_values(ascending=False).index)
    g60.to_csv(a.out / "ic_60s.csv")
    sec.append("## 1. Single features: rank IC vs 60 s forward move (train), mean by coin group\n\n" + md(g60))
    decay = pd.DataFrame({f"{h // 1000}s": pd.DataFrame(ics[h]).mean(axis=1) for h in HORIZONS}).loc[g60.index[:15]]
    sec.append("### IC decay with horizon (mean over coins, top 15 features)\n\n" + md(decay))
    pi = group_means(pd.DataFrame(partials))
    pi = pi.reindex(pi["all"].abs().sort_values(ascending=False).index)
    sec.append("## 2. What each feature adds beyond `imbalance_l5` (partial rank IC, 60 s)\n\n" + md(pi.head(20)))

    for coin in [c for c in ("BTC", "SOL", "PUMP") if c in inter]:
        t = inter[coin]
        piv = t.pivot_table(index=["book", "speed"], columns="recent", values="mean_move_bps", observed=True)
        piv = piv.reindex(columns=[c for c in ("fell", "flat", "rose") if c in piv.columns])
        sec.append(f"## 3. Interaction — {coin}: mean 60 s move (bps) by book state × last-30 s move × 30 s speed\n\n" + md(piv))

    mdf = pd.concat(models, ignore_index=True)
    mdf.to_csv(a.out / "models.csv", index=False)
    for h in (60, 300):
        sub = mdf[mdf["horizon_s"] == h]
        summ = sub.groupby("model", sort=False)[["test_IC", "top10%_move_bps", "top5%_move_bps", "top1%_move_bps", "top1%_hit", "top1%_taker_gross_bps"]].mean()
        sec.append(f"## 4. Out-of-sample models, {h} s horizon — mean over coins\n\n`topX%_move_bps` = mean signed mid move on the X% most confident test predictions; `top1%_taker_gross_bps` = market-order round trip on those, before fees.\n\n" + md(summ))
        best = sub[sub["model"] == "all features (GBM)"].set_index("coin")[["test_IC", "top5%_move_bps", "top1%_move_bps", "top1%_hit", "top1%_taker_gross_bps"]]
        sec.append(f"### All-features GBM per coin, {h} s\n\n" + md(best))
    imp = pd.DataFrame(importances).T
    sec.append("## 5. GBM feature-family importance (drop in test IC when the family is shuffled, 60 s)\n\n" + md(imp))

    (a.out / "results.md").write_text("\n\n".join(sec) + "\n")
    log(f"wrote {a.out / 'results.md'}")


if __name__ == "__main__":
    main()
