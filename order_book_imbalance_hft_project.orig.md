# Order Book Imbalance HFT Strategy

## 1. Project Goal

Build and evaluate a short-horizon trading strategy for BTC and major altcoin perpetual futures using order book imbalance and related microstructure signals.

The strategy should only trade when the expected move in executable prices is large enough to cover trading fees, spread, slippage, and other execution costs. The primary objective is positive **net PnL**, not simply accurate direction prediction.

## 2. Initial Scope

- **Markets:** BTC and ETH perpetual futures first; consider SOL and other highly liquid major altcoins after validation.
- **Signal:** Best-level order book imbalance, with multi-level imbalance and order flow imbalance as comparison signals.
- **Holding horizon:** Test several horizons, initially 10 ms, 50 ms, 100 ms, 500 ms, and 1 second where data quality supports them.
- **Execution:** Begin with historical replay and paper trading. Do not deploy live capital until results include realistic fees, spread, latency, partial fills, and slippage.
- **Initial target:** Investigate a 0.1% favorable executable price move as a candidate threshold, not an assumed achievable or optimal target.

## 3. Trading-Cost Model

Assume a fee of **0.04% per side** (0.0004).

For a buy followed by a sell:

- Entry: current ask.
- Exit: future bid.
- Gross price return: `future_bid / current_ask - 1`.
- Net return, assuming fees apply proportionally to entry and exit notional:

  `net_return = (future_bid / current_ask) * (1 - 0.0004) - 1 - 0.0004`

Break-even requires:

`future_bid / current_ask > (1 + 0.0004) / (1 - 0.0004)`

This corresponds to approximately **0.08003% favorable executable price movement**, before slippage, funding, and other costs.

For a short followed by a buyback:

- Entry: current bid.
- Exit: future ask.
- Gross price return: `current_bid / future_ask - 1`.
- Net return can be modeled with the same two fee factors; break-even requires:

  `current_bid / future_ask > (1 + 0.0004) / (1 - 0.0004)`

A 0.1% favorable executable-price move would leave only about **0.02% net return** under this fee assumption, before slippage, funding, and other costs. Confirm the exchange's actual maker/taker fee schedule and fee calculation before relying on these figures.

**Important:** Mid-price movement is not the same as executable profit. A strategy must compare entry and exit prices on the correct sides of the spread.

## 4. Signal Definitions

### 4.1 Best-Level Order Book Imbalance

Let:

- `Q_bid`: quantity resting at the best bid.
- `Q_ask`: quantity resting at the best ask.

Calculate:

`imbalance = (Q_bid - Q_ask) / (Q_bid + Q_ask)`

Interpretation:

- Near `+1`: best-bid quantity dominates best-ask quantity; this may indicate upward pressure.
- Near `-1`: best-ask quantity dominates best-bid quantity; this may indicate downward pressure.
- Near `0`: displayed quantities are more balanced.

This is a probabilistic signal, not a guarantee. Resting orders can be canceled, replenished, or spoofed, and a large displayed queue may not represent durable demand or supply.

### 4.2 Comparison Signals

Evaluate whether these improve prediction beyond best-level imbalance:

- Multi-level imbalance across the first several price levels.
- Order flow imbalance, using additions, cancellations, and executions at the best quotes.
- Aggressive buy/sell trade volume and trade-arrival rate.
- Spread, top-of-book depth, and depth changes.
- Short-term volatility and recent mid-price returns.

Keep feature calculations causal: only use information available at the decision timestamp.

## 5. Candidate Trading Logic

Treat these as hypotheses to test, not final production rules.

### Long candidate

1. Calculate imbalance and supporting features from the latest valid order book.
2. Consider a long entry when the signal indicates upward pressure and the predicted executable exit bid is sufficiently above the current ask.
3. Estimate expected net return after entry/exit fees, spread, slippage, and any applicable funding.
4. Enter only if expected net return exceeds a safety margin.
5. Exit according to a tested profit target, stop-loss, time limit, or signal reversal.

### Short candidate

1. Calculate the same features.
2. Consider a short entry when the signal indicates downward pressure and the predicted executable buyback ask is sufficiently below the current bid.
3. Require expected net return after all modeled costs to exceed a safety margin.
4. Exit according to a tested profit target, stop-loss, time limit, or signal reversal.

Do not assume that an imbalance threshold or a 0.1% target will work until it is validated out of sample.

## 6. Data Requirements

Capture synchronized, timestamped market data:

- Order book snapshots and incremental updates, ideally with sequence numbers.
- Best bid/ask prices and quantities.
- Several levels of book depth.
- Public trades, including price, quantity, timestamp, and aggressor side if available.
- Exchange timestamps and local receive timestamps.
- Instrument metadata, tick size, quantity step, contract multiplier, and fee schedule.
- Funding rates and funding timestamps for perpetual futures.
- Order acknowledgments, fills, rejects, cancellations, and local send/receive timestamps during execution testing.

Detect sequence gaps and stale books. Rebuild the book correctly before generating signals. Store both exchange event time and local arrival time so the effect of feed and decision latency can be measured.

## 7. Backtesting Plan

### Step 1: Data validation

- Verify timestamp ordering, sequence continuity, and book reconstruction.
- Remove or flag periods with missing updates or stale quotes.
- Confirm that the entry quote was actually available at the decision time.

### Step 2: Predictive analysis

For each imbalance bucket and forecast horizon, measure:

- Probability of the next mid-price move being up or down.
- Future mid-price change.
- Future **executable** long return: future bid relative to current ask.
- Future **executable** short return: current bid relative to future ask.
- Distribution of favorable and adverse excursions.
- Time to target and time to stop.
- Results by market, volatility regime, spread, and depth.

Suggested initial horizons: 10 ms, 50 ms, 100 ms, 500 ms, and 1 second, depending on data granularity.

### Step 3: Strategy simulation

- Simulate decisions using only data available at each timestamp.
- Include entry and exit fees, spread, slippage, latency, and funding where relevant.
- Model partial fills and rejected orders where possible.
- Avoid assuming a fill merely because a candle or mid-price touched a level.
- For passive orders, model queue position and fill probability; do not assume every resting order fills.

### Step 4: Robustness checks

- Split data chronologically into training, validation, and untouched test periods.
- Tune thresholds on training/validation data only.
- Test across multiple market regimes and assets.
- Include latency stress tests and more conservative slippage assumptions.
- Compare against simple baselines, including no-trade and momentum-only rules.

## 8. Key Metrics

Report at minimum:

- Net PnL after all modeled costs.
- Net return per trade and per unit of exposure.
- Trade count and turnover.
- Win rate and average win/loss.
- Maximum drawdown.
- Profit factor and Sharpe ratio, with assumptions stated.
- Target-hit rate before stop-hit rate.
- Average holding time.
- Fee, spread, and slippage contribution to PnL.
- Fill rate, partial-fill rate, rejection rate, and latency distribution.
- Results by imbalance bucket, forecast horizon, asset, and market regime.

Directional accuracy alone is insufficient. A signal may predict direction well but still lose money if the move is too small or execution costs consume the edge.

## 9. Initial Experiments

1. Collect order book and trade data for BTC and ETH perpetual futures.
2. Establish baseline results using best-level imbalance only.
3. Measure executable-price outcomes across the candidate horizons.
4. Check how often favorable movement reaches 0.1% before adverse movement, while also testing smaller and larger targets.
5. Add multi-level imbalance and order flow imbalance.
6. Compare taker execution with realistic passive-order simulations.
7. Paper trade the best out-of-sample configuration and compare live execution with simulated fills.

## 10. Risks and Limitations

- **Competition:** Other participants may react to the same public signals.
- **Latency:** The signal may decay before an order reaches the exchange.
- **Adverse selection:** A fill may occur precisely because the market is moving against the strategy.
- **Book instability:** Displayed liquidity can be canceled or replenished quickly.
- **Execution uncertainty:** Backtests often overstate fills and understate slippage.
- **Fees:** Actual maker/taker fees may differ from the assumed 0.04% per side.
- **Funding and contract rules:** Perpetual futures may incur funding and have instrument-specific constraints.
- **Overfitting:** Thresholds that work in one period or asset may fail elsewhere.

## 11. Success Criteria

The project is worth advancing only if it demonstrates:

- Positive net returns on untouched out-of-sample data after realistic costs.
- Results that remain positive under conservative latency and slippage assumptions.
- Enough trade opportunities to be operationally meaningful.
- Acceptable drawdown and stable performance across multiple periods.
- Paper-trading results reasonably consistent with the simulation.

No profitability should be assumed in advance. If the measured edge is smaller than execution uncertainty or total costs, reject or redesign the strategy.
