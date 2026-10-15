"""Fit the up/down/calm predictor on Hyperliquid history and score it out of sample.

Uses the 1 s-sampled frames written by ``run_research.py``
(``reports/<COIN>_sample.parquet``). By default ``up``/``down`` mean the mid
moved by more than the Hyperliquid tier-0 limit/limit round trip (3 bps) over
60 s.

Example:
    python scripts/fit_predictor.py --coin BTC \
        --train 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01 2026-04-01 2026-05-01 2026-06-01 \
        --test 2026-07-01 2026-08-01 2026-09-01 2026-10-01 --save models/BTC_60s.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obi import costs  # noqa: E402
from obi.predictor import NAMES, BucketPredictor, score  # noqa: E402

DEFAULT_COST_BPS = costs.round_trip_cost(costs.HYPERLIQUID_BASE, "maker", "maker") * 1e4


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coin", default="BTC")
    p.add_argument("--samples", type=Path, default=Path("reports"))
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--horizon-ms", type=int, default=60_000)
    p.add_argument("--label", choices=["mid", "taker"], default="mid", help="mid = limit orders, taker = market orders")
    p.add_argument("--signal", default="imbalance_l5", help="imbalance feature to bucket on")
    p.add_argument("--cost-bps", type=float, nargs="+", default=[DEFAULT_COST_BPS], help="round-trip cost(s) defining up/down")
    p.add_argument("--min-prob", type=float, nargs="+", default=[0.35, 0.4, 0.45, 0.5])
    p.add_argument("--save", type=Path, default=None, help="save the model for the first --cost-bps value")
    a = p.parse_args()

    df = pd.read_parquet(a.samples / f"{a.coin}_sample.parquet")
    h = a.horizon_ms
    train = df[df["day"].isin(a.train)]
    test = df[df["day"].isin(a.test)].dropna(subset=[f"mid_ret_{h}", f"long_ret_{h}"])
    print(f"{a.coin}: train rows={len(train):,} test rows={len(test):,} horizon={h / 1000:g}s label={a.label} signal={a.signal}\n")

    rows = []
    for k, cost in enumerate(a.cost_bps):
        thr = cost / 1e4
        model = BucketPredictor(threshold=thr, horizon_ms=h, label=a.label, signal_feature=a.signal).fit(train)
        if k == 0 and a.save:
            a.save.parent.mkdir(parents=True, exist_ok=True)
            model.save(a.save)
            print(f"saved {a.save}")
        actual = model.labels(test)
        probs = model.predict_proba(test[a.signal], test[model.vol_feature])
        print(f"--- cost {cost:g} bps: train prior " + ", ".join(f"{NAMES[c]}={q:.1%}" for c, q in zip((-1, 0, 1), model.prior)))
        print(f"    max P(up) in any cell = {probs[:, 2].max():.1%}, max P(down) = {probs[:, 0].max():.1%}")
        # Acting on a call: gross ~ mid move for limit orders, executable move for market orders.
        if a.label == "mid":
            long_r, short_r = test[f"mid_ret_{h}"], -test[f"mid_ret_{h}"]
        else:
            long_r, short_r = test[f"long_ret_{h}"], test[f"short_ret_{h}"]
        for mp in a.min_prob:
            pred = model.predict(test[a.signal], test[model.vol_feature], min_prob=mp)
            s = score(pred, actual, long_r, short_r, fee_round_trip=thr)
            rows.append(
                {
                    "cost_bps": cost,
                    "min_prob": mp,
                    "accuracy": s["accuracy"],
                    "always_calm_acc": s["always_calm_accuracy"],
                    "up_calls": s["up_calls"],
                    "up_precision": s["up_precision"],
                    "down_calls": s["down_calls"],
                    "down_precision": s["down_precision"],
                    "net_bps_per_call*": s["net_bps_per_call"],
                }
            )
        lean = probs[:, 2] - probs[:, 0]
        mid = test[f"mid_ret_{h}"].to_numpy()
        strong = np.abs(lean) >= np.quantile(np.abs(lean), 0.9)
        moved = strong & (mid != 0)
        print(f"    top-10% strongest leans: raw {h / 1000:g} s direction hit rate = {((lean[moved] > 0) == (mid[moved] > 0)).mean():.1%}\n")
    print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g"))
    if a.label == "mid":
        print("\n* assumes both limit orders fill at the touch; the backtest and the dry-run fill simulation measure the real fill risk.")


if __name__ == "__main__":
    main()
