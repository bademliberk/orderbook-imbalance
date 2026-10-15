"""Performance metrics (plan section 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(pnl: np.ndarray) -> float:
    equity = np.concatenate([[0.0], np.cumsum(pnl)])
    return float(np.max(np.maximum.accumulate(equity) - equity))


def summarize(trades: pd.DataFrame, stats: dict | None = None, days: float | None = None) -> dict:
    """Summary metrics. Returns are fractions of entry notional per trade.

    Sharpe is computed on daily PnL (sum of per-trade returns at constant
    notional) and annualised with sqrt(365); per-trade Sharpe is not annualised
    because trade frequency is irregular. Both need several days to mean much.
    """
    out: dict = {"trades": len(trades)}
    if stats:
        sub = max(stats.get("submitted", 0), 1)
        out["fill_rate"] = stats.get("filled", 0) / sub
        out["reject_rate"] = stats.get("rejected", 0) / sub
        out["partial_rate"] = stats.get("partial", 0) / max(stats.get("filled", 0), 1)
        out["maker_unfilled_rate"] = stats.get("unfilled_maker", 0) / sub
        out["exit_fallback_rate"] = stats.get("exit_fallback", 0) / max(stats.get("filled", 0), 1)
    if trades.empty:
        out.update(net_pnl=0.0, net_bps_per_trade=np.nan)
        return out
    net = trades["net"].to_numpy()
    wins, losses = net[net > 0], net[net <= 0]
    out.update(
        net_pnl=float(net.sum()),
        net_bps_per_trade=float(net.mean() * 1e4),
        gross_bps_per_trade=float(trades["gross"].mean() * 1e4),
        tstat=float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))) if len(net) > 1 and net.std() > 0 else np.nan,
        win_rate=float((net > 0).mean()),
        avg_win_bps=float(wins.mean() * 1e4) if len(wins) else np.nan,
        avg_loss_bps=float(losses.mean() * 1e4) if len(losses) else np.nan,
        profit_factor=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf,
        max_drawdown=max_drawdown(net),
        fee_bps_per_trade=float(trades["fees"].mean() * 1e4),
        spread_bps_per_trade=float(trades["spread_cost"].mean() * 1e4),
        slippage_bps_per_trade=float(trades["slippage"].mean() * 1e4),
        funding_bps_per_trade=float(trades["funding"].mean() * 1e4),
        avg_hold_ms=float(trades["hold_ms"].mean()),
        target_rate=float((trades["reason"] == "target").mean()),
        stop_rate=float((trades["reason"] == "stop").mean()),
        long_share=float((trades["side"] > 0).mean()),
    )
    if "exit_type" in trades:
        out["exit_maker_share"] = float((trades["exit_type"] == "maker").mean())
    day = trades["t_entry"] // 86_400_000
    daily = trades.groupby(day)["net"].sum()
    out["days_traded"] = len(daily)
    if len(daily) > 1 and daily.std() > 0:
        out["sharpe_daily_ann"] = float(daily.mean() / daily.std(ddof=1) * np.sqrt(365))
    if days:
        out["trades_per_day"] = len(trades) / days
    return out
