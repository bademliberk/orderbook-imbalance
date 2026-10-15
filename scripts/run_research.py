"""Run the research pipeline on Hyperliquid data (Tardis) and write a report.

Base case: post-only limit orders on entry and exit at Hyperliquid tier-0 fees
(0.015% maker), with market-order fallback when a limit exit does not fill.

Example:
    python scripts/run_research.py --coins BTC ETH \
        --train 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01 2026-04-01 \
        --val 2026-05-01 2026-06-01 --test 2026-07-01 2026-08-01 2026-09-01 2026-10-01
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obi import costs  # noqa: E402
from obi.analysis import bucket_table, excursion_table, information_coefficient, move_size_table  # noqa: E402
from obi.backtest import ExecConfig  # noqa: E402
from obi.research import (  # noqa: E402
    HORIZONS,
    MODEL_FEATURES,
    RuleSpec,
    imbalance_grid,
    linear_grid,
    log,
    momentum_grid,
    pick_best,
    prepare_day,
    round_trip_fee,
    run_specs,
    sample_frame,
)
from obi.strategy import LinearForecaster  # noqa: E402

KEY_COLS = [
    "spec",
    "trades",
    "trades_per_day",
    "net_bps_per_trade",
    "gross_bps_per_trade",
    "net_pnl",
    "win_rate",
    "fee_bps_per_trade",
    "fill_rate",
    "exit_maker_share",
    "exit_fallback_rate",
    "max_drawdown",
]


def md(df: pd.DataFrame, floatfmt=".4g") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def key(df: pd.DataFrame, cols=KEY_COLS) -> pd.DataFrame:
    """Select report columns; configs with no trades lack most metrics."""
    return df.reindex(columns=cols)


def fee_table() -> pd.DataFrame:
    base = costs.HYPERLIQUID_BASE
    rows = []
    for name, fe, fx in [
        ("Tier 0 limit / limit (base case)", base.maker, base.maker),
        ("Tier 0 limit entry / market exit", base.maker, base.taker),
        ("Tier 0 market / market", base.taker, base.taker),
        ("Tier 0 + Gold staking, limit / limit", *[costs.hyperliquid_fees(0, "gold").maker] * 2),
        ("Tier 3 (>$100M 14d), limit / limit", *[costs.hyperliquid_fees(3).maker] * 2),
        ("Maker rebate 0.001%, limit / limit", -0.00001, -0.00001),
    ]:
        rows.append({"scenario": name, "entry_fee_bps": fe * 1e4, "exit_fee_bps": fx * 1e4, "breakeven_bps": costs.breakeven_move(fe, fx) * 1e4})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH"])
    p.add_argument("--train", nargs="+", type=date.fromisoformat, required=True)
    p.add_argument("--val", nargs="+", type=date.fromisoformat, required=True)
    p.add_argument("--test", nargs="+", type=date.fromisoformat, required=True)
    p.add_argument("--raw", type=Path, default=Path("data/tardis"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--latency-ms", type=int, default=200)
    p.add_argument("--order-qty-usd", type=float, default=1_000.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fees = costs.HYPERLIQUID_BASE
    be = costs.breakeven_move(fees.maker, fees.maker)
    sections = ["## Cost model (Hyperliquid perps)\n\n" + md(fee_table())]

    for coin in args.coins:
        log(f"=== {coin} ===")
        all_days = args.train + args.val + args.test
        days, val_rows, samples = {}, [], []
        for d in all_days:
            log(f"prepare {coin} {d}")
            dd = prepare_day(coin, d, args.raw, args.latency_ms)
            samples.append(sample_frame(dd))
            dd.labels = None
            val_rows.append({"day": d.isoformat(), "summary": dd.validation.summary()})
            days[d] = dd
        sample = pd.concat(samples, ignore_index=True)
        sample.to_parquet(args.out / f"{coin}_sample.parquet")
        qty = args.order_qty_usd / float(np.median(days[all_days[0]].book["bid"]))
        cfg = ExecConfig(fees=fees, latency_ms=args.latency_ms, order_qty=qty)
        spread_bps = float(np.median(days[all_days[-1]].features["spread_bps"]))

        sec = [
            f"# {coin}",
            f"Order size {qty:.6g} {coin} ({args.order_qty_usd:,.0f} USD); latency {args.latency_ms} ms per order; "
            f"median spread {spread_bps:.3f} bps (last test day). Base execution: limit entry + limit exit "
            f"(timeouts {cfg.maker_timeout_ms / 1000:g}s / {cfg.exit_timeout_ms / 1000:g}s, market fallback on exit).",
            "### Data validation\n\n" + md(pd.DataFrame(val_rows)),
        ]

        train_s = sample[sample["day"].isin([d.isoformat() for d in args.train])]
        lab_cols = [c for c in sample.columns if c.startswith(("mid_ret", "long_", "short_", "next_mid"))]
        mv = move_size_table(train_s[lab_cols], HORIZONS, (0.0002, 0.0003, 0.0005, 0.0009, 0.002))
        sec.append(
            "### How often is the move big enough? (train days, |mid change|)\n\n"
            f"Limit/limit break-even is {be * 1e4:.2f} bps; market/market is {costs.breakeven_move(fees.taker, fees.taker) * 1e4:.2f} bps.\n\n" + md(mv)
        )
        for feat in ("imbalance", "imbalance_l5"):
            bt = bucket_table(train_s[feat], train_s[lab_cols], (5_000, 30_000, 60_000, 300_000), be)
            bt.to_csv(args.out / f"{coin}_bucket_{feat}.csv", index=False)
            sec.append(
                f"### `{feat}` buckets (train days)\n\n"
                f"`mid_bps` = mean mid change; `long_exec_bps`/`short_exec_bps` = market-order round trip (gross); "
                f"`p_absmid_gt_be` = P(|mid move| > {be * 1e4:.2f} bps limit/limit break-even).\n\n" + md(bt)
            )
        ex = excursion_table(train_s["imbalance_l5"], train_s[lab_cols], (30_000, 60_000))
        sec.append("### Favourable / adverse excursions by `imbalance_l5`, bps (train days)\n\n" + md(ex))
        ic = pd.DataFrame(
            {f"IC mid {h // 1000}s": information_coefficient(train_s[MODEL_FEATURES], train_s, f"mid_ret_{h}") for h in (5_000, 30_000, 60_000, 300_000)}
        ).reset_index(names="feature")
        sec.append("### Feature rank-IC vs forward mid returns (train days)\n\n" + md(ic))

        train, val, test = ([days[d] for d in ds] for ds in (args.train, args.val, args.test))
        min_tr = 5 * len(args.train)

        log("imbalance grid (limit/limit) on train")
        res = run_specs(imbalance_grid(), train, cfg)
        res.to_csv(args.out / f"{coin}_imbalance_train.csv", index=False)
        sec.append("### Imbalance rules — train, limit/limit\n\n" + md(key(res).sort_values("net_pnl", ascending=False)))
        zero = run_specs(imbalance_grid(), train, replace(cfg, fees=costs.ZERO_FEES))
        sec.append("### Imbalance rules — train, limit/limit, ZERO fees (diagnostic)\n\n" + md(key(zero).sort_values("net_pnl", ascending=False).head(8)))
        specs = {s.label(): s for s in imbalance_grid() + momentum_grid()}

        log("momentum baseline")
        mom = run_specs(momentum_grid(), train, cfg)
        sec.append("### Momentum-only baseline — train, limit/limit\n\n" + md(key(mom).sort_values("net_pnl", ascending=False)))

        # Linear model of the mid move (limit orders trade near the mid), fit on
        # train, horizon/margin chosen on validation.
        log("linear forecaster")
        lin_rows, models, pred_rows = [], {}, []
        for h, lspecs in linear_grid():
            y = train_s[f"mid_ret_{h}"]
            model = LinearForecaster(MODEL_FEATURES).fit(train_s, y, -y)
            models[h] = model
            r = run_specs(lspecs, val, cfg, model)
            r["horizon"] = h
            lin_rows.append(r)
            pl, ps = model.predict(val[0].features)
            best = np.maximum(pl, ps) * 1e4
            pred_rows.append(
                {
                    "horizon_s": h // 1000,
                    "pred_p50_bps": np.median(best),
                    "pred_p99_bps": np.quantile(best, 0.99),
                    "pred_max_bps": best.max(),
                    "hurdle_bps": round_trip_fee(cfg) * 1e4,
                    "train_R2": float(np.corrcoef(model.predict(train_s)[0], y.fillna(0))[0, 1] ** 2),
                }
            )
        lin = pd.concat(lin_rows, ignore_index=True)
        sec.append("### Linear mid-move model — validation (fit on train)\n\n" + md(key(lin)))
        sec.append("Predicted best-side move on the first validation day vs the fee hurdle:\n\n" + md(pd.DataFrame(pred_rows)))

        chosen = []
        if (b := pick_best(res, min_tr)) is not None:
            chosen.append(("imbalance", specs[b["spec"]], None))
        if (b := pick_best(mom, min_tr)) is not None:
            chosen.append(("momentum", specs[b["spec"]], None))
        if (b := pick_best(lin, 5 * len(args.val))) is not None:
            h = int(b["horizon"])
            chosen.append(("linear", RuleSpec("linear", float(b["spec"].split("th=")[1].split()[0]), h), models[h]))
        oos = []
        for name, spec, model in chosen:
            oos.append(run_specs([spec], val, cfg, model).assign(period="validation", family=name))
            oos.append(run_specs([spec], test, cfg, model).assign(period="test", family=name))
        oos.append(pd.DataFrame([{"family": "no-trade", "period": "test", "spec": "flat", "trades": 0, "net_pnl": 0.0}]))
        pd.concat(oos, ignore_index=True).to_csv(args.out / f"{coin}_oos.csv", index=False)
        sec.append("### Selected configurations — validation and untouched test\n\n" + md(key(pd.concat(oos, ignore_index=True), ["family", "period"] + KEY_COLS)))

        best = next((s for n, s, _ in chosen if n == "imbalance"), None)
        if best is not None:
            log("stress tests")
            rows = []
            for label, c in [
                ("limit/limit, tier 0 (base)", cfg),
                ("limit entry / market exit", replace(cfg, exit_mode="taker")),
                ("market / market", replace(cfg, entry_mode="taker", exit_mode="taker")),
                ("limit/limit, zero fees (gross)", replace(cfg, fees=costs.ZERO_FEES)),
                ("limit/limit, Gold staking", replace(cfg, fees=costs.hyperliquid_fees(0, "gold"))),
                ("limit/limit, tier 3", replace(cfg, fees=costs.hyperliquid_fees(3))),
                ("limit/limit, maker rebate 0.001%", replace(cfg, fees=costs.hyperliquid_fees(0, maker_rebate=0.00001))),
                ("queue: half of displayed size ahead", replace(cfg, queue_fraction=0.5)),
                ("limit timeouts 15 s", replace(cfg, maker_timeout_ms=15_000, exit_timeout_ms=15_000)),
                ("latency 500 ms", replace(cfg, latency_ms=500)),
                ("latency 1000 ms", replace(cfg, latency_ms=1_000)),
            ]:
                rows.append(run_specs([best], test, c).assign(scenario=label))
            sens = pd.concat(rows, ignore_index=True)
            sens.to_csv(args.out / f"{coin}_stress.csv", index=False)
            cols = ["scenario", "trades", "net_bps_per_trade", "gross_bps_per_trade", "fee_bps_per_trade", "net_pnl", "fill_rate", "exit_maker_share", "exit_fallback_rate"]
            sec.append(f"### Stress tests on test days — `{best.label()}`\n\n" + md(key(sens, cols)))

        sections.append("\n\n".join(sec))
        days.clear()
        gc.collect()

    (args.out / "results.md").write_text("# Order Book Imbalance on Hyperliquid — Research Results\n\n" + "\n\n".join(sections) + "\n")
    log(f"wrote {args.out / 'results.md'}")


if __name__ == "__main__":
    main()
