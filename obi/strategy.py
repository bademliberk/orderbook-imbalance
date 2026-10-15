"""Signal rules (plan section 5). These are hypotheses to be tested."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def threshold_signal(feature: np.ndarray, threshold: float, max_spread_bps: float | None = None, spread_bps=None) -> np.ndarray:
    """+1 when ``feature > threshold``, -1 when ``feature < -threshold``."""
    feature = np.asarray(feature)
    s = np.where(feature > threshold, 1, np.where(feature < -threshold, -1, 0)).astype(np.int8)
    if max_spread_bps is not None and spread_bps is not None:
        s[np.asarray(spread_bps) > max_spread_bps] = 0
    return s


def momentum_signal(ret_bps: np.ndarray, threshold_bps: float) -> np.ndarray:
    """Momentum-only baseline: trade in the direction of the recent return."""
    return threshold_signal(ret_bps, threshold_bps)


@dataclass
class LinearForecaster:
    """OLS forecasts of gross executable long and short returns.

    Trades when the predicted gross return exceeds round-trip fees plus a
    safety margin. Spread is *not* subtracted separately: the targets are
    already measured ask-to-bid / bid-to-ask, so they include it.
    """

    features: list[str]
    coef_long: np.ndarray | None = None
    coef_short: np.ndarray | None = None
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    clip: float = 5.0
    history: dict = field(default_factory=dict)

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.features].to_numpy(np.float64) - self.mu) / self.sd
        Z = np.clip(np.nan_to_num(Z), -self.clip, self.clip)
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, X: pd.DataFrame, y_long: pd.Series, y_short: pd.Series) -> "LinearForecaster":
        mask = y_long.notna() & y_short.notna()
        Xm = X.loc[mask, self.features].to_numpy(np.float64)
        self.mu = np.nanmean(Xm, axis=0)
        self.sd = np.nanstd(Xm, axis=0) + 1e-12
        A = self._design(X.loc[mask])
        self.coef_long = np.linalg.lstsq(A, y_long[mask].to_numpy(), rcond=None)[0]
        self.coef_short = np.linalg.lstsq(A, y_short[mask].to_numpy(), rcond=None)[0]
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        A = self._design(X)
        return A @ self.coef_long, A @ self.coef_short

    def signal(self, X: pd.DataFrame, round_trip_fee: float, margin: float) -> np.ndarray:
        pl, ps = self.predict(X)
        hurdle = round_trip_fee + margin
        go_long = (pl > hurdle) & (pl >= ps)
        go_short = (ps > hurdle) & (ps > pl)
        return np.where(go_long, 1, np.where(go_short, -1, 0)).astype(np.int8)
