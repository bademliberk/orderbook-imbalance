"""Download Hyperliquid book snapshots and trades from Tardis.dev.

Without TARDIS_API_KEY only the first day of each month is available, so by
default this fetches every first-of-month in the range.

Example:
    python scripts/download_data.py --coins BTC ETH --start 2025-11-01 --end 2026-10-01
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from obi.data.tardis import download, first_of_months  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH"])
    p.add_argument("--datasets", nargs="+", default=["book_snapshot_25", "trades"])
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--every-day", action="store_true", help="every day in range (needs TARDIS_API_KEY)")
    p.add_argument("--out", type=Path, default=Path("data/tardis"))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.every_day:
        if not os.environ.get("TARDIS_API_KEY"):
            p.error("--every-day needs TARDIS_API_KEY")
        days = [a.start + timedelta(d) for d in range((a.end - a.start).days + 1)]
    else:
        days = first_of_months(a.start, a.end)

    jobs = [(ds, c, d) for c in a.coins for ds in a.datasets for d in days]

    def run(job):
        print(f"ok  {download(*job, a.out)}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        for f in [pool.submit(run, j) for j in jobs]:
            f.result()


if __name__ == "__main__":
    main()
