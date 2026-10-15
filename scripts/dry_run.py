"""Live dry run on Hyperliquid: predict up / down / calm every few seconds,
verify each call after the model's horizon (60 s), and simulate the limit-order
round trip of every up/down call. No orders are sent.

Fit a model first with scripts/fit_predictor.py, then:
    python scripts/dry_run.py --coin BTC --model models/BTC_60s.json --minutes 60
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obi import costs  # noqa: E402
from obi.dry_run import DryRun, run_dry  # noqa: E402
from obi.predictor import BucketPredictor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coin", default="BTC")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--every-s", type=float, default=10.0, help="seconds between predictions")
    p.add_argument("--latency-ms", type=int, default=200, help="delay before a simulated order rests on the book")
    p.add_argument("--min-prob", type=float, default=0.35, help="probability needed to call up/down (fitted models rarely exceed ~0.4)")
    p.add_argument("--fee-tier", type=int, default=0, choices=sorted(costs.HYPERLIQUID_TIERS))
    p.add_argument("--staking", default="none", choices=list(costs.HYPE_STAKING_DISCOUNT))
    p.add_argument("--queue-fraction", type=float, default=1.0, help="share of displayed size ahead of our limit order")
    p.add_argument("--entry-timeout-s", type=float, default=5.0)
    p.add_argument("--exit-timeout-s", type=float, default=5.0)
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--out", type=Path, default=Path("reports/dry_run.csv"))
    a = p.parse_args()
    fees = costs.hyperliquid_fees(a.fee_tier, a.staking)
    dry = DryRun(
        BucketPredictor.load(a.model),
        fees=fees,
        every_s=a.every_s,
        latency_ms=a.latency_ms,
        min_prob=a.min_prob,
        queue_fraction=a.queue_fraction,
        entry_timeout_ms=int(a.entry_timeout_s * 1e3),
        exit_timeout_ms=int(a.exit_timeout_s * 1e3),
        out=a.out,
    )
    m = dry.model
    print(
        f"{a.coin}: horizon {m.horizon_ms / 1e3:g}s, up/down = {m.label} move > {m.threshold * 1e4:g} bps, "
        f"signal {m.signal_feature}; fees maker {fees.maker * 1e4:.2f} / taker {fees.taker * 1e4:.2f} bps"
    )
    asyncio.run(run_dry(a.coin, dry, a.minutes * 60 if a.minutes else None))


if __name__ == "__main__":
    main()
