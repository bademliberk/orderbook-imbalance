"""Predictive analysis tables: imbalance buckets x horizons (plan step 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BUCKETS = np.array([-1.0001, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0001])


def bucket_table(
    signal: pd.Series,
    labels: pd.DataFrame,
    horizons_ms: tuple[int, ...],
    breakeven: float,
    buckets: np.ndarray = DEFAULT_BUCKETS,
) -> pd.DataFrame:
    """Per-bucket, per-horizon outcome statistics.

    Returns are reported in basis points. ``p_long_be`` / ``p_short_be`` are the
    probabilities that a long / short taker round trip beats ``breakeven``
    (a gross fraction, e.g. ``costs.breakeven_move(...)``).
    """
    b = pd.cut(signal, buckets, right=True)
    rows = []
    for h in horizons_ms:
        df = pd.DataFrame(
            {
                "bucket": b,
                "next_up": labels["next_mid_move"] > 0,
                "next_down": labels["next_mid_move"] < 0,
                "mid": labels[f"mid_ret_{h}"] * 1e4,
                "long": labels[f"long_ret_{h}"] * 1e4,
                "short": labels[f"short_ret_{h}"] * 1e4,
                "long_be": labels[f"long_ret_{h}"] > breakeven,
                "short_be": labels[f"short_ret_{h}"] > breakeven,
                "abs_mid_gt_be": labels[f"mid_ret_{h}"].abs() > breakeven,
            }
        ).dropna(subset=["mid"])
        g = df.groupby("bucket", observed=True)
        t = pd.DataFrame(
            {
                "n": g.size(),
                "p_next_up": g["next_up"].mean(),
                "p_next_down": g["next_down"].mean(),
                "mid_bps": g["mid"].mean(),
                "long_exec_bps": g["long"].mean(),
                "short_exec_bps": g["short"].mean(),
                "p_long_be": g["long_be"].mean(),
                "p_short_be": g["short_be"].mean(),
                "p_absmid_gt_be": g["abs_mid_gt_be"].mean(),
            }
        )
        t.insert(0, "horizon_ms", h)
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


def move_size_table(labels: pd.DataFrame, horizons_ms: tuple[int, ...], thresholds: tuple[float, ...]) -> pd.DataFrame:
    """How often the |mid move| at each horizon exceeds each threshold.

    This is the first sanity check: if moves larger than the round-trip cost
    are rare at a horizon, no direction signal can make a taker strategy pay
    there.
    """
    rows = []
    for h in horizons_ms:
        m = labels[f"mid_ret_{h}"].dropna().abs()
        row = {"horizon_ms": h, "n": len(m), "median_abs_bps": m.median() * 1e4, "p99_abs_bps": m.quantile(0.99) * 1e4}
        for th in thresholds:
            row[f"p_abs>{th * 1e4:.1f}bps"] = (m > th).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def excursion_table(signal: pd.Series, labels: pd.DataFrame, horizons_ms: tuple[int, ...], buckets=DEFAULT_BUCKETS) -> pd.DataFrame:
    """Mean favourable/adverse excursion (bps) per bucket and horizon."""
    b = pd.cut(signal, buckets)
    rows = []
    for h in horizons_ms:
        cols = [f"long_mfe_{h}", f"long_mae_{h}", f"short_mfe_{h}", f"short_mae_{h}"]
        if not all(c in labels for c in cols):
            continue
        t = (labels[cols] * 1e4).groupby(b, observed=True).mean()
        t.columns = ["long_mfe_bps", "long_mae_bps", "short_mfe_bps", "short_mae_bps"]
        t.insert(0, "horizon_ms", h)
        rows.append(t.reset_index(names="bucket"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def information_coefficient(features: pd.DataFrame, labels: pd.DataFrame, target: str) -> pd.Series:
    """Spearman rank correlation of each feature with a forward label."""
    y = labels[target]
    mask = y.notna()
    ranked_y = y[mask].rank()
    return features[mask].rank().corrwith(ranked_y).sort_values(key=np.abs, ascending=False)
