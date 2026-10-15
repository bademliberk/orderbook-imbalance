"""Cross-coin comparison of signal strength and strategy results.

Reads the per-coin files written by ``run_research.py`` (``<COIN>_sample.parquet``,
``<COIN>_oos.csv``, ``<COIN>_stress.csv``) from one or more report directories.

Example:
    python scripts/compare_coins.py reports reports/alts --test 2026-07-01 2026-08-01 2026-09-01 2026-10-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

H = 60_000
BE = 0.0003  # limit/limit round trip at tier 0


def coin_row(sample: pd.DataFrame, stress: pd.DataFrame | None, test_days: list[str]) -> dict:
    s = sample.dropna(subset=[f"mid_ret_{H}"])
    test = s[s["day"].isin(test_days)]
    imb, mid = s["imbalance_l5"], s[f"mid_ret_{H}"]
    row = {
        "median_spread_bps": float(np.median(s["spread_bps"])),
        "median_|60s|_bps": float(mid.abs().median() * 1e4),
        "P(|60s|>3bps)": float((mid.abs() > BE).mean()),
        "IC_60s": float(imb.rank().corr(mid.rank())),
        # Mean 60 s mid move after the most extreme imbalance (sign-adjusted).
        "extreme_move_bps": float(np.concatenate([mid[imb > 0.8], -mid[imb < -0.8]]).mean() * 1e4),
    }
    t_imb, t_mid = test["imbalance_l5"].to_numpy(), test[f"mid_ret_{H}"].to_numpy()
    strong = (np.abs(t_imb) > 0.8) & (t_mid != 0)
    row["test_hit_rate_|imb|>0.8"] = float(((t_imb[strong] > 0) == (t_mid[strong] > 0)).mean()) if strong.any() else np.nan
    if stress is not None:
        get = lambda scen, col: float(stress.loc[stress["scenario"] == scen, col].iloc[0])  # noqa: E731
        row.update(
            {
                "limit_net_bps": get("limit/limit, tier 0 (base)", "net_bps_per_trade"),
                "limit_gross_bps": get("limit/limit, tier 0 (base)", "gross_bps_per_trade"),
                "limit_fill_rate": get("limit/limit, tier 0 (base)", "fill_rate"),
                "exit_fallback": get("limit/limit, tier 0 (base)", "exit_fallback_rate"),
                "market_gross_bps": get("market / market", "gross_bps_per_trade"),
                "market_net_bps": get("market / market", "net_bps_per_trade"),
                "test_trades": get("limit/limit, tier 0 (base)", "trades"),
            }
        )
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--test", nargs="+", required=True)
    a = p.parse_args()
    rows = {}
    for d in a.dirs:
        for f in sorted(d.glob("*_sample.parquet")):
            coin = f.name.removesuffix("_sample.parquet")
            stress_path = d / f"{coin}_stress.csv"
            stress = pd.read_csv(stress_path) if stress_path.exists() else None
            rows[coin] = coin_row(pd.read_parquet(f), stress, a.test)
    df = pd.DataFrame.from_dict(rows, orient="index").sort_values("median_spread_bps")
    print(df.to_markdown(floatfmt=".3g"))


if __name__ == "__main__":
    main()
