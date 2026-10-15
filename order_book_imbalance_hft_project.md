# Order Book Imbalance Strategy — Hyperliquid Perpetuals

> **Revision note (2026-10-14).** Reviewed and implemented. The venue is
> **Hyperliquid** and execution uses **limit (post-only) orders**. Corrections to
> the original plan are in section 0; the original is kept in
> [order_book_imbalance_hft_project.orig.md](order_book_imbalance_hft_project.orig.md).
> Study results are in section 12 and [reports/results.md](reports/results.md).
> An earlier Binance study is archived in [reports/archive_binance_2024-03/](reports/archive_binance_2024-03/results.md).

## 0. Review: Corrections to the Original Plan

| # | Section | Problem | Correction |
|---|---|---|---|
| 1 | §3 short side | Gross short return was given as `current_bid / future_ask - 1`. For a linear perpetual, a short earns `(entry - exit) × qty`, so return on entry notional is `1 - exit / entry`. The original formula measures against exit notional and overstates gains (100 → 90 gives 11.1% instead of 10%). | Use `1 - exit / entry`; net `1 - f_entry - (exit / entry)(1 + f_exit)`. |
| 2 | §3 fees | The plan assumed a flat 0.04% per side (a Binance-style taker rate). Hyperliquid charges **0.015% maker / 0.045% taker** at tier 0, and execution here is by limit order. | Use maker fees for filled limit orders and taker fees for market fallbacks and stops (table in §3). |
| 3 | §3 limit orders | A limit order's fee is low only *if it fills*. Fill probability and adverse selection (resting orders fill mainly when the price moves through them) dominate the economics, and a plan that only changes the fee rate misses them. | Model queue position and fills from public trades; report fill rate, fallback rate and the gross return of *filled* trades (§7 step 3). |
| 4 | §2, §7 horizons | 10 ms to 1 s horizons are below what Hyperliquid data and latency resolve. Book snapshots arrive every ~0.54 s (history and the live `l2Book` fast feed), blocks every ~0.1–0.2 s, and there is no colocation. | Study 1 s to 5 min. |
| 5 | §5 step 3 | "Expected net return after fees, **spread**, slippage" double-counts the spread when the forecast is already bid-to-ask. | Subtract the spread only from a mid-price forecast. |
| 6 | §7 steps 1 and 3 | Orders were filled at the decision-time quote. | Orders rest or fill against the book at decision time + latency. |
| 7 | §4.2 OFI | Additions, cancellations and executions cannot be separated with L2 data. | Use best-quote OFI (Cont–Kukanov–Stoikov). |
| 8 | §8 Sharpe | Annualising a per-trade Sharpe ratio is misleading for irregular trading. | Report daily-PnL Sharpe plus the per-trade t-statistic. |
| 9 | Framing | Imbalance mostly predicts the next tick. Always check the move size against costs before analysing direction. | Move-size feasibility check first (§7 step 2). |
| 10 | §5 limit entries | A limit buy rests on the bid. When imbalance is strongly positive, that bid queue is the *crowded* side: the order fills only after the large queue ahead of it trades away, which happens mostly when the expected up-move fails. Directional limit entries are adversely selected *by construction*. | Measure it (§12). Treat limit execution as a market-making problem, not a fee discount on a directional strategy. |

## 1. Project Goal

Build and evaluate a short-horizon strategy for Hyperliquid BTC and major altcoin perpetuals using order book imbalance and related microstructure signals, executed with limit orders. Trade only when the expected move covers fees, fill risk, adverse selection and other execution costs. The objective is positive **net PnL**, not directional accuracy.

Staging: (1) offline research on history; (2) a live **dry run** that predicts up / down / calm and verifies each call after 1 minute; (3) only if (1) and (2) pass, an on-chain trading app.

## 2. Initial Scope

- **Venue:** Hyperliquid perpetuals (USDC-margined, linear, hourly funding).
- **Markets:** BTC and ETH first; SOL and other liquid perpetuals after validation.
- **Signal:** Best-level and 5-level order book imbalance, with order flow imbalance, trade flow and recent returns as comparisons.
- **Horizons:** 1 s, 5 s, 30 s, 60 s, 5 min. The dry run verifies at 60 s.
- **Execution:** Post-only (ALO) limit orders at the touch for entry and exit. A limit exit that does not fill within a timeout is closed with a market order; stops always use market orders.
- **Process:** Historical replay, then the live dry run with simulated fills, then (if justified) a small live deployment.

## 3. Trading-Cost Model

Hyperliquid perps fees (docs, checked Oct 2026; governance can change them), charged on notional per fill:

| Tier (14-day weighted volume) | Taker | Maker |
|---|---|---|
| 0 (< $5M) | 0.045% | 0.015% |
| 1 (> $5M) | 0.040% | 0.012% |
| 2 (> $25M) | 0.035% | 0.008% |
| 3 (> $100M) | 0.030% | 0.004% |
| 4–6 (> $500M) | 0.028–0.024% | 0.000% |

HYPE staking cuts these by 5–40% (for example, Gold, >10k HYPE staked, gives 20% off). A referral gives 4% off the first $25M of volume. Maker rebates of −0.001% to −0.003% require at least a 0.5% share of total maker volume. There is no gas per trade.

Round-trip break-even at tier 0:

| Execution (entry / exit) | Fees | Break-even move between fill prices |
|---|---|---|
| Limit / limit | 0.015% + 0.015% | 3.00 bps |
| Limit / market (fallback or stop) | 0.015% + 0.045% | 6.00 bps |
| Market / market | 0.045% + 0.045% | 9.00 bps |

Net returns (fractions of entry notional), with fill prices `entry` and `exit`:

- Long: `net = (exit / entry)(1 - f_exit) - 1 - f_entry`
- Short: `net = 1 - f_entry - (exit / entry)(1 + f_exit)`

A limit order fills at its own price, so a limit round trip buys at the bid and sells at the ask and *earns* the spread. Hyperliquid BTC/ETH spreads are usually 1 tick (BTC: $1, about 0.12 bps). That gain is real only if both orders fill, which is the central risk (§0 rows 3 and 10).

Funding is paid hourly; it matters only for holds spanning the hour.

## 4. Signal Definitions

### 4.1 Order book imbalance

`imbalance = (Q_bid - Q_ask) / (Q_bid + Q_ask)` at the best level. `imbalance_l5` applies the same formula to the first five levels, weighting level *k* by 0.5^k. Near +1 the bid dominates (upward pressure); near −1 the ask dominates. This is probabilistic: displayed size can be cancelled or replenished.

### 4.2 Comparison signals

Order flow imbalance at the best quotes, aggressive trade imbalance and count, microprice deviation, spread, depth, recent returns and short-term volatility. All features are causal: they use only data available at the decision time.

## 5. Candidate Trading Logic (hypotheses)

**Long:** when the signal indicates upward pressure, post a buy limit at the best bid. If it fills, hold for the horizon, then post a sell limit at the best ask. If the sell does not fill within the timeout, sell at market. Use a market-order stop if the loss exceeds the limit. **Short:** mirror image.

Trade only if the expected move, net of the fees implied by the *expected mix* of limit and market fills, exceeds a safety margin. Do not assume any threshold works until it is validated out of sample.

## 6. Data

- **History:** Tardis.dev Hyperliquid `book_snapshot_25` (~0.54 s snapshots; 5 levels used to match the live feed) and `trades` (aggressor side). The first day of each month is free; other days need an API key.
- **Live:** Hyperliquid websocket (`wss://api.hyperliquid.xyz/ws`): `l2Book` with `fast: true` (5 levels, ~0.54 s; without `fast` it was observed at ~5 s), `bbo` (each block, ~130 ms) and `trades`. Snapshots are complete, so there is no sequence synchronisation; gaps are detected by time.
- Store exchange and local timestamps. Flag gaps longer than 5 s.

## 7. Backtesting Plan

1. **Validation:** ordering, crossed books, gaps.
2. **Predictive analysis:** move-size feasibility versus break-even, imbalance buckets, excursions and feature rank-IC per horizon.
3. **Simulation:** post-only limit orders join the queue behind the displayed size at the touch (configurable fraction). They fill on trade-through, or when printed volume at the price exceeds the queue ahead. Entries are cancelled after a timeout; exits fall back to market orders. Apply latency to every order, stops at market, hourly funding, and per-leg maker/taker fees.
4. **Robustness:** chronological train / validation / untouched test; fee tiers, staking, rebates, queue position, timeouts and latency stress; momentum and no-trade baselines.

## 8. Key Metrics

Net PnL; net, gross and fee bps per trade; trade count; win rate; drawdown; daily Sharpe and t-statistic; **limit entry fill rate, exit maker share and market fallback rate**; gross return of filled trades (adverse selection); results by bucket, horizon, asset and period. For the dry run: precision of up/down calls, accuracy against an always-calm baseline, raw direction hit rate, and simulated limit round-trip PnL.

## 9. Experiments

1. Download monthly sample days (done: Nov 2025 – Oct 2026, BTC and ETH).
2. Run the research pipeline (`scripts/run_research.py`).
3. Fit the up/down/calm predictor (`scripts/fit_predictor.py`).
4. Live dry run with simulated limit fills (`scripts/dry_run.py`) for at least several days across conditions.
5. Proceed to an on-chain app only if net results are positive out of sample *and* the dry run's simulated fills agree with the backtest.

## 10. Risks

Adverse selection of resting orders; low fill rates and market-order fallbacks; competition from market makers who are faster and have better fee tiers or rebates; queue position uncertainty (public data cannot show where you are in the queue); fee schedule changes by governance; funding; overfitting; API and connectivity outages (the dry run handles gaps, a live app must also handle open orders).

## 11. Success Criteria

Positive net returns on untouched out-of-sample data after realistic fees **and fill modelling**. These must remain positive under conservative queue and latency assumptions, with enough trades to matter and stable results across months. The dry run must agree with the simulation. Otherwise, reject or redesign.

## 12. Results (2026-10-14)

**Setup.** BTC and ETH, first day of each month from Nov 2025 to Oct 2026 (about 156k snapshots and 120–300k trades per day). Train Nov–Apr, validate May–Jun, untouched test Jul–Oct. Orders are 1,000 USD with 200 ms latency, tier-0 fees, and post-only limit entry and exit (5 s timeouts, market fallback on exit). Full tables are in [reports/results.md](reports/results.md).

**Data quality.** No crossed books. A few feed gaps (12–320 s per day) were flagged and excluded from decisions.

**1. Moves can cover limit-order fees at 60 s or longer** (BTC, train): P(|60 s mid move| > 3 bps) = 46%, with a median of 2.6 bps; at 5 min, 71% and 6.3 bps. This is much more feasible than taker trading on Binance.

**2. The signal is stronger than on Binance.** Rank IC of `imbalance_l5` against the forward mid move: BTC 0.43 (5 s), 0.21 (30 s), 0.15 (60 s), 0.07 (5 min); ETH is slightly lower. The 5-level imbalance is marginally better than the best level. The strongest decile calls the 60 s direction correctly **63–65%** of the time out of sample. The mean 60 s mid move in the most extreme bucket is still only about ±1 bp.

**3. Limit orders do not rescue it. All configurations lose on train, validation and test:**

| BTC, test (Jul–Oct 2026), best train config | Net bps/trade | Gross bps/trade | Limit entry fill rate | Exits via market fallback |
|---|---|---|---|---|
| Limit / limit, tier 0 (base) | −7.1 | −2.1 | 4.6% | 67% |
| Limit entry / market exit | −7.3 | −1.3 | 4.5% | 100% (by design) |
| Market / market | −8.5 | +0.5 | 100% | — |
| Limit / limit, zero fees | −2.1 | −2.1 | 4.6% | 67% |
| Limit / limit, tier 3 fees | −4.6 | −2.1 | 4.6% | 67% |
| Limit / limit, 15 s timeouts | −7.0 | −2.5 | 14% | 50% |

ETH is similar (test: −7.8 bps/trade, 3.8% fill rate, 79% fallback).

**Why.** The limit-entry columns show adverse selection directly. Market entries capture the signal (+0.5 bps gross), but limit entries that do fill are **−2 bps gross**, because they fill when the predicted move fails. Most limit exits then miss and fall back to market orders, so the realised fee is about 5 bps rather than 3. **Even with zero fees, limit/limit loses.**

**4. Predictor (dry-run model).** At a 3 bps limit round-trip threshold and 60 s, the model's maximum P(up) or P(down) is 35–42%, so it rarely makes a call. Up/down precision is 34–38%. Acting on calls loses about −1.6 to −2 bps per call *even if every limit order filled*. Accuracy never beats the always-calm baseline.

**5. Live dry run (smoke test).** The predict → verify → simulate loop works on the live Hyperliquid feed. As in the backtest, 0 of 6 simulated limit entries filled within 5 s.

**Conclusion against §11.** Not met. On Hyperliquid, imbalance is a genuine short-term direction signal, but its edge is about 1 bp at 60 s. Limit execution adds adverse selection that more than cancels the fee saving. Market execution costs 9 bps against about 0.5 bps of edge.

**What could still work:**
- **Market making, not direction.** Quote both sides and use imbalance to *skew* quotes or pull the side about to be run over. Profit comes from the spread plus rebates; it needs high maker share and fast cancels, and higher tiers help a lot (tier 4+: 0% maker).
- **Imbalance as an execution filter** for a slower strategy that must trade anyway (for example, delay a buy when `imbalance_l5` < −0.8). Saving even 0.5–1 bp per fill is pure gain there.
- **Contrarian limit placement.** Post on the side the signal says will be *hit* (sell limits when imbalance is very positive) and test whether the fill-conditional return is positive. Not tested yet.
- **Less liquid Hyperliquid perps**, where ticks are larger relative to price. Fill and adverse-selection risks are also higher.

### 12.1 Small alts (added 2026-10-14)

The same pipeline was run on ten more Hyperliquid perps, from volume rank 3 to 56 (SOL, HYPE, PUMP, ENA, DOGE, kPEPE, AVAX, FARTCOIN, PENGU, APT), over the same 12 monthly days. Full table: [reports/coin_comparison.md](reports/coin_comparison.md); per-coin detail: [reports/alts/results.md](reports/alts/results.md).

| | BTC / ETH | Mid alts (SOL, DOGE, AVAX) | Small alts (ENA, APT, FARTCOIN, PENGU, PUMP, kPEPE) |
|---|---|---|---|
| Median spread | 0.1–0.4 bps | 0.5–1.1 bps | 2–5 bps |
| P(\|60 s move\| > 3 bps) | 43–51% | 58–62% | 67–81% |
| Rank IC of `imbalance_l5` vs 60 s move | 0.14–0.16 | 0.05–0.11 | 0.03–0.09 |
| Test hit rate when \|imbalance\| > 0.8 | 63–64% | 56–61% | 54–65% |
| Limit/limit gross per trade (test) | −2.0 to −2.4 bps | −3.0 to −3.7 bps | −4.6 to −14 bps |
| Limit/limit net per trade (test) | −7.1 to −7.8 bps | −8.6 to −9.4 bps | −9.9 to −20 bps |
| Market/market net per trade (test) | −8.5 to −9.1 bps | −8.7 to −9.3 bps | −6.8 to −11 bps |

**Findings:**
- **Bigger moves, weaker signal.** Small alts move past the 3 bps hurdle more often, but imbalance predicts their moves less well (IC falls about 5×). Thin books are easily reshaped by a single order, so displayed size says less about future flow.
- **Adverse selection gets worse.** Filled limit entries lose 5–14 bps gross on small alts, against 2 bps on BTC. Exits fall back to market orders 69–100% of the time. (kPEPE had only 16 test trades, so its −20 bps is noisy.)
- **Market orders come closest but still lose.** On ENA, PENGU, PUMP and kPEPE, market-order round trips are *gross positive* (+0.9 to +2.2 bps after crossing a 2–5 bps spread). That is a real signal, but well below the 9 bps of fees.
- **Of 240 coin × rule combinations, none had positive net returns on training data.**

**Conclusion.** Small alts do not rescue the directional imbalance strategy. Their wider spreads make *passive market making* the more natural angle there (§12, "What could still work"), but that also faces the adverse selection measured here.

### 12.2 Feature study: price-change vector × book depth × spread (added 2026-10-14)

**Question.** Does combining the recent price-change vector (direction, speed, acceleration, directional consistency), the order book weight difference within the top *n* bps of mid, and spread dynamics give a clearer signal than imbalance alone, and how are they related? Data: all 12 coins, 1 s samples, train Nov–Jun, untouched test Jul–Oct. Full output: [reports/signal_study/results.md](reports/signal_study/results.md) (`scripts/signal_study.py`).

**Features.** Price vector over 2 s–5 min: signed return, speed (|return|/s), efficiency (net move ÷ path length, −1…1), activity (mid changes/s), short and long acceleration. Book: imbalance at L1, 5-level weighted and flat, notional imbalance within 1/2/5/10 bps of mid, and their 5 s changes. Spread: level, 5 s change, 30 s max.

**Limitation.** The free history has only 5 book levels, which reach ~0.6 bps from mid on BTC, 2–5 bps on ETH, SOL and mid alts, and 5–10 bps on small alts. Bands wider than that (0.05–1%) cannot be studied historically. `record_live.py --agg-sig-figs 4 3` now records aggregated books that reach ~5 and ~55 bps on BTC at 0.54 s, so wider bands can be tested on data collected from now on.

**Findings:**
1. **The book carries the signal; the price vector mostly does not.** Rank IC vs the 60 s move (mean over coins): book imbalance within 5 bps 0.073, L1 imbalance 0.073, 5-level 0.072 (majors ~0.11). At 5 s it is ~0.20. The best band is 2–5 bps; L1 is almost as good. The 5 s *change* in book imbalance is weaker (0.046) but the most stable day to day (ICIR ~3.5–4). The best price features are very short momentum (2 s return, 5 s efficiency: ~0.03). **Speed alone has no directional information** (IC ≈ −0.01). Spread features have none.
2. **Relation: the price vector is mostly a consequence of the same order flow.** What price features add beyond imbalance is ≤0.02 partial IC. The one consistent extra is mild **reversal after big 5-minute moves** (`ret_300s`, `eff_300s` ≈ −0.02).
3. **Interaction: imbalance is stronger when a fast recent move agrees with it.** Mean 60 s move with a bid-heavy book: after a fast 30 s rise +1.1 bps (BTC), +1.3 (SOL), +3.5 (PUMP); after a fall +0.3, +0.6, +1.7. The mirror holds for ask-heavy books. Agreement roughly doubles the expected move, but only to 1–3.5 bps.
4. **Combining does not help out of sample.** Mean test IC at 60 s: book-only linear 0.082, imbalance alone 0.077, all features linear 0.080, all features with interactions 0.079, gradient boosting 0.076, price vector alone 0.03. Gradient boosting relies almost entirely on book features (shuffling price features changes IC by <0.01), so there is no hidden nonlinear combination. The 1% most confident predictions move only **+2.3 bps** (book-only) at 60 s, and market round trips on them earn +0.2 to +0.7 bps before fees.
5. **The 5-minute reversal lead fails a robustness check.** Fading the top 1% of 5-minute moves earned +6.8 bps per independent episode on the test months (255 episodes), but **lost −2.7 bps on train** (702 episodes). No coin reached |t| > 2. It is regime noise, not a usable signal.

**Conclusion.** The clearest signal is the order book itself: imbalance at L1 or within 2–5 bps, best read when the last 30 s moved the same way. No combination of price direction, speed, depth bands and spread pushes the expected 60 s or 5 min move above trading costs out of sample. The open questions are wider depth bands (needs the live aggregated recording) and the market-making framing from §12.

### 12.3 Depth within 1% of mid (in progress, started 2026-10-15)

The Tardis history has only 5 book levels, so depth within 0.1–2% of mid cannot be backtested on it. Hyperliquid's aggregated `l2Book` feeds (`nSigFigs`, 20 levels, ~5 s cadence) reach 0.25–19% from mid depending on the coin and aggregation, and cover the 1% band ~100% of the time on all recorded coins.

- **Recording:** `record_live.py --coins BTC ETH SOL HYPE DOGE PUMP FARTCOIN --agg-sig-figs 4 3 --hours 72` into `data/live/`.
- **Feature:** `features.merge_depth_bands`, notional imbalance within 10/25/50/100/200 bps. Each band uses the finest aggregation that reaches it, merged causally onto the execution book.
- **Backtest:** `scripts/depth_backtest.py --band-bps 100`. Chronological split; direction (follow or fade) and top-X% thresholds come from the first 60%; the remaining 40% is run through the event-driven backtester in market/market, limit/limit and limit/market modes.
- **Status:** pipeline validated on the first 25 minutes. Results so far are noise (train/test IC flips sign; 0–3 trades per setting). A first read needs ≥ 24 h; a decision needs several days to weeks across different market conditions.
