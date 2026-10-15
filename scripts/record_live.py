"""Record live Hyperliquid l2Book (5 levels, ~0.5 s), bbo and trades, plus
optional aggregated deep books for "depth within n% of mid" features.

Example:
    python scripts/record_live.py --coins BTC ETH --hours 24 --agg-sig-figs 4 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obi.data.hyperliquid import record  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coins", nargs="+", default=["BTC"])
    p.add_argument("--out", type=Path, default=Path("data/live"))
    p.add_argument("--hours", type=float, default=None, help="stop after this many hours (default: forever)")
    p.add_argument("--agg-sig-figs", type=int, nargs="*", default=[], help="also record aggregated books, e.g. 4 3")
    a = p.parse_args()
    asyncio.run(record(a.coins, a.out, a.hours * 3600 if a.hours else None, tuple(a.agg_sig_figs)))


if __name__ == "__main__":
    main()
