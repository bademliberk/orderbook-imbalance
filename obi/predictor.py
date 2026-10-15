"""Up / down / calm direction predictor and its scoring.

Classes are defined by whether the move over the horizon would pay for a round
trip, not by raw price direction. Two label modes:

- ``label="mid"`` (default, for limit orders on both legs): ``up`` if the mid
  rises by more than ``threshold``, ``down`` if it falls by more, else ``calm``.
  With 1-tick Hyperliquid spreads this approximates a maker round trip *if both
  orders fill*; fill risk is measured separately (backtest / dry-run fill sim).
- ``label="taker"``: ``up`` if ``exit_bid / entry_ask - 1 > threshold``,
  ``down`` if ``1 - exit_ask / entry_bid > threshold`` (market orders).

``threshold`` should be the round-trip fee of the chosen execution (Hyperliquid
base tier: 0.03% maker+maker, 0.09% taker+taker) plus any safety margin.

The model is a smoothed lookup table of class frequencies by imbalance bucket
and volatility regime. It is simple on purpose: it is easy to audit, and the
probabilities it reports are just historical frequencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = (-1, 0, 1)
NAMES = {-1: "down", 0: "calm", 1: "up"}
IMB_EDGES = [-1.0001, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0001]


def mid_direction_class(mid_ret, threshold: float) -> np.ndarray:
    mid_ret = np.asarray(mid_ret, dtype=np.float64)
    return np.where(mid_ret > threshold, 1, np.where(mid_ret < -threshold, -1, 0)).astype(np.int8)


def direction_class(long_ret, short_ret, threshold: float) -> np.ndarray:
    long_ret = np.asarray(long_ret, dtype=np.float64)
    short_ret = np.asarray(short_ret, dtype=np.float64)
    return np.where(long_ret > threshold, 1, np.where(short_ret > threshold, -1, 0)).astype(np.int8)


@dataclass
class BucketPredictor:
    threshold: float
    horizon_ms: int = 60_000
    label: str = "mid"  # "mid" (limit orders) or "taker"
    signal_feature: str = "imbalance"
    vol_feature: str = "vol_5000ms_bps"
    vol_edges: list[float] = field(default_factory=list)  # interior tercile edges
    probs: dict[str, list[float]] = field(default_factory=dict)  # "i,j" -> P(down, calm, up)
    prior: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    smoothing: float = 20.0  # pseudo-counts drawn from the prior

    def _cells(self, imbalance, vol) -> list[str]:
        i = np.clip(np.searchsorted(IMB_EDGES, np.asarray(imbalance), side="left") - 1, 0, len(IMB_EDGES) - 2)
        j = np.searchsorted(self.vol_edges, np.asarray(vol), side="right")
        return [f"{a},{b}" for a, b in zip(i, j)]

    def labels(self, df: pd.DataFrame) -> np.ndarray:
        h = self.horizon_ms
        if self.label == "mid":
            return mid_direction_class(df[f"mid_ret_{h}"], self.threshold)
        return direction_class(df[f"long_ret_{h}"], df[f"short_ret_{h}"], self.threshold)

    def fit(self, df: pd.DataFrame) -> "BucketPredictor":
        h = self.horizon_ms
        df = df.dropna(subset=[f"mid_ret_{h}", f"long_ret_{h}", f"short_ret_{h}"])
        y = self.labels(df)
        self.vol_edges = [float(x) for x in np.quantile(df[self.vol_feature], [1 / 3, 2 / 3])]
        counts = np.array([(y == c).sum() for c in CLASSES], dtype=np.float64)
        self.prior = (counts / counts.sum()).tolist()
        cells = pd.Series(self._cells(df[self.signal_feature], df[self.vol_feature]))
        tab = pd.crosstab(cells, y).reindex(columns=list(CLASSES), fill_value=0)
        prior = np.array(self.prior)
        self.probs = {
            cell: ((row.to_numpy() + self.smoothing * prior) / (row.sum() + self.smoothing)).tolist()
            for cell, row in tab.iterrows()
        }
        return self

    def predict_proba(self, imbalance, vol) -> np.ndarray:
        cells = self._cells(np.atleast_1d(imbalance), np.atleast_1d(vol))
        return np.array([self.probs.get(c, self.prior) for c in cells])

    def predict(self, imbalance, vol, min_prob: float = 0.5) -> np.ndarray:
        """Call up/down only if that class has probability >= ``min_prob``."""
        p = self.predict_proba(imbalance, vol)
        call = np.zeros(len(p), dtype=np.int8)
        call[(p[:, 2] >= min_prob) & (p[:, 2] > p[:, 0])] = 1
        call[(p[:, 0] >= min_prob) & (p[:, 0] > p[:, 2])] = -1
        return call

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: Path) -> "BucketPredictor":
        return cls(**json.loads(Path(path).read_text()))


def score(pred: np.ndarray, actual: np.ndarray, long_ret=None, short_ret=None, fee_round_trip: float = 0.0) -> dict:
    """Accuracy, per-class precision, a confusion matrix and the PnL of acting.

    ``long_ret``/``short_ret`` are gross executable returns; acting means going
    long on ``up`` calls and short on ``down`` calls for the horizon.
    """
    pred = np.asarray(pred)
    actual = np.asarray(actual)
    out = {
        "n": int(len(pred)),
        "accuracy": float((pred == actual).mean()) if len(pred) else np.nan,
        "always_calm_accuracy": float((actual == 0).mean()) if len(pred) else np.nan,
        "actual_share": {NAMES[c]: float((actual == c).mean()) for c in CLASSES},
    }
    for c in (1, -1):
        called = pred == c
        out[f"{NAMES[c]}_calls"] = int(called.sum())
        out[f"{NAMES[c]}_precision"] = float((actual[called] == c).mean()) if called.any() else np.nan
        out[f"{NAMES[c]}_recall"] = float((pred[actual == c] == c).mean()) if (actual == c).any() else np.nan
    out["confusion"] = (
        pd.crosstab(pd.Series(actual, name="actual").map(NAMES), pd.Series(pred, name="predicted").map(NAMES))
        .reindex(index=["down", "calm", "up"], columns=["down", "calm", "up"], fill_value=0)
    )
    if long_ret is not None:
        long_ret, short_ret = np.asarray(long_ret), np.asarray(short_ret)
        gross = np.where(pred == 1, long_ret, np.where(pred == -1, short_ret, 0.0))
        acted = pred != 0
        net = gross[acted] - fee_round_trip
        out["acted"] = int(acted.sum())
        out["net_bps_per_call"] = float(net.mean() * 1e4) if acted.any() else np.nan
        out["net_bps_total"] = float(net.sum() * 1e4)
    return out
