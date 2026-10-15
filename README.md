# Order Book Imbalance on Hyperliquid — research toolkit

Implementation of [order_book_imbalance_hft_project.md](order_book_imbalance_hft_project.md)
for **Hyperliquid perpetuals with limit-order execution**. Results are in
[reports/results.md](reports/results.md) and summarised in section 12 of the plan.

## Setup

Standalone project; dependencies are pinned in [requirements.txt](requirements.txt).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## Workflow

```bash
# 1. History from Tardis.dev (first day of each month is free; others need TARDIS_API_KEY)
python scripts/download_data.py --coins BTC ETH --start 2025-11-01 --end 2026-10-01

# 2. Validation, features, labels, bucket analysis, walk-forward selection,
#    untouched test, fee-tier / queue / latency / execution-mode stress tests -> reports/results.md
python scripts/run_research.py --coins BTC ETH \
    --train 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01 2026-04-01 \
    --val 2026-05-01 2026-06-01 --test 2026-07-01 2026-08-01 2026-09-01 2026-10-01

# 3. Fit the up/down/calm predictor (60 s, up/down = mid move > 3 bps limit round trip)
python scripts/fit_predictor.py --coin BTC \
    --train 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01 2026-04-01 2026-05-01 2026-06-01 \
    --test 2026-07-01 2026-08-01 2026-09-01 2026-10-01 --save models/BTC_60s.json

# 4. Live dry run: predict every 10 s, verify after 60 s, simulate the limit-order
#    round trip of each up/down call (no orders are sent) -> reports/dry_run.csv
python scripts/dry_run.py --coin BTC --model models/BTC_60s.json --minutes 60

# 5. Compare coins (any set run through step 2, e.g. alts into reports/alts)
python scripts/download_data.py --coins SOL HYPE PUMP ENA DOGE kPEPE AVAX FARTCOIN PENGU APT --start 2025-11-01 --end 2026-10-01
python scripts/run_research.py --coins SOL HYPE PUMP ENA DOGE kPEPE AVAX FARTCOIN PENGU APT --out reports/alts --train ... --val ... --test ...
python scripts/compare_coins.py reports reports/alts --test 2026-07-01 2026-08-01 2026-09-01 2026-10-01

# 6. Feature study: price-change vector x book depth bands x spread -> reports/signal_study/results.md
python scripts/signal_study.py --coins BTC ETH SOL HYPE DOGE AVAX ENA APT PUMP FARTCOIN PENGU kPEPE

# 7. Record live Hyperliquid data (l2Book fast, bbo, trades); --agg-sig-figs adds
#    aggregated 20-level books (BTC: 4 -> ~0.25%, 3 -> ~2.4% from mid) for wide depth bands
python scripts/record_live.py --coins BTC ETH SOL HYPE DOGE PUMP FARTCOIN --agg-sig-figs 4 3 --hours 72

#    ...or run the recorder as a service: starts at login, restarts after crashes,
#    reconnects after Wi-Fi drops / sleep; stop + start continues collecting
scripts/recorder_service.sh install   # also: start | stop | restart | status | logs | uninstall

# 8. Backtest depth imbalance within 1% of mid on those recordings (history can't: only 5 levels)
python scripts/depth_backtest.py --band-bps 100   # -> reports/depth_backtest.md
```

Fee options for the dry run: `--fee-tier 0..6`, `--staking none|wood|...|diamond`.
Backtests: `obi.costs.hyperliquid_fees(tier, staking, referral, maker_rebate)`.

## Running the recorder on a VPS

Any Ubuntu 24.04 server with 1 vCPU / 1 GB RAM / 25 GB disk is enough (7 coins
use ~250 MB of disk per day). Create it with your SSH key, then from this laptop:

```bash
deploy/vps.sh deploy   root@<ip>   # copy code, install deps, start systemd service (~3 min)
deploy/vps.sh status   root@<ip>   # running? restarts, data size, free disk, recent log
deploy/vps.sh logs     root@<ip>
deploy/vps.sh stop|start|restart root@<ip>
deploy/vps.sh sync     root@<ip>   # pull recordings into data/vps/ (incremental)
deploy/vps.sh backtest root@<ip>   # sync + 1%-depth backtest -> reports/depth_backtest_vps.md
deploy/vps.sh remove   root@<ip>   # remove the service (data stays on the VPS)
```

The service starts on boot, restarts on crash, and reconnects after network
drops. Re-run `deploy` after code changes; recordings are never touched.
Options: `COINS="BTC ETH" AGG="4 3" SSH_OPTS="-i ~/.ssh/key" deploy/vps.sh deploy ...`.
Keep laptop (`data/live`) and VPS (`data/vps`) recordings separate. They cover
the same market, so merging them would double-count trades in the fill model.

## Layout

| Module | Purpose |
|---|---|
| [obi/costs.py](obi/costs.py) | Hyperliquid fee tiers, staking/referral discounts, rebates; long/short net returns |
| [obi/data/tardis.py](obi/data/tardis.py) | Tardis downloads and loaders (book snapshots, trades) |
| [obi/data/hyperliquid.py](obi/data/hyperliquid.py) | Live websocket stream (`l2Book` fast, `bbo`, `trades`), recorder, loaders |
| [obi/validate.py](obi/validate.py) | Ordering, crossed books, feed gaps, validity flags |
| [obi/features.py](obi/features.py) | Imbalance (L1, 5-level, n-bps bands), price-change vector (direction, speed, efficiency, acceleration), OFI, trade flow, spread dynamics (all causal) |
| [obi/labels.py](obi/labels.py) | Latency-aware forward returns and excursions |
| [obi/analysis.py](obi/analysis.py) | Move-size feasibility, bucket tables, excursions, rank IC |
| [obi/backtest.py](obi/backtest.py) | Post-only limit entry/exit with queue model and market fallback, market orders, stops, hourly funding, per-leg fees |
| [obi/strategy.py](obi/strategy.py) | Imbalance threshold rule, momentum baseline, linear model |
| [obi/metrics.py](obi/metrics.py) | Net PnL, cost split, fill / maker-share / fallback rates, drawdown, Sharpe |
| [obi/research.py](obi/research.py) | Pipeline and walk-forward grids |
| [obi/predictor.py](obi/predictor.py) | Up/down/calm classifier and scoring |
| [obi/dry_run.py](obi/dry_run.py) | Live dry run with limit-order fill simulation |
| [scripts/compare_coins.py](scripts/compare_coins.py) | Cross-coin signal and strategy comparison |
| [obi/signal_study.py](obi/signal_study.py) | Feature-discovery analyses: IC/ICIR, partial IC, interactions, OOS models, family importance |

## Notes and limitations

- **Feed cadence:** the live `l2Book` feed without `fast` was observed at ~5 s
  per update; with `fast: true` it is ~0.54 s (5 levels), matching the Tardis
  history. `bbo` updates every block (~130 ms).
- **Fill model:** public trades are the only evidence of fills. Orders join
  behind the displayed size (`queue_fraction`, default 1.0) and ignore
  cancellations ahead of them, which is conservative for fill rate. Your real
  queue position is only known once you place real orders.
- **Depth beyond ~0.1–10 bps** (e.g. 1% of mid) is not in the history; it comes
  only from live aggregated `nSigFigs` books recorded with `--agg-sig-figs`.
- **Free history** covers one day per month; set `TARDIS_API_KEY` and use
  `--every-day` for continuous periods.
- **Fees** come from Hyperliquid docs (Oct 2026); governance can change them.
