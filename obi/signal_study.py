"""Which microstructure features carry a clear signal, alone or combined?

Analyses (all on 1 s samples, train / untouched test split by month):

1. ``ic_table``: Spearman rank IC of each feature vs forward mid return, and its
   day-to-day stability (ICIR = mean daily IC / std of daily IC).
2. ``partial_ic``: IC of each feature against the forward return *after*
   removing what 5-level imbalance explains — i.e. what it adds.
3. ``interaction_table``: forward move by imbalance x recent-move direction
   (x speed): does imbalance mean more when price agrees or disagrees with it?
4. ``model_comparison``: out-of-sample models by feature family (linear,
   linear with interactions, gradient boosting) scored by IC and by whether
   their most confident predictions move more than trading costs.
5. ``family_importance``: permutation importance of feature families in the
   gradient-boosting model on the test months.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BOOK = [
    "imbalance",
    "imbalance_l5",
    "depth_imbalance_l5",
    "microprice_dev_bps",
    "band_imb_1bps",
    "band_imb_2bps",
    "band_imb_5bps",
    "band_imb_10bps",
    "imbalance_l5_chg_5s",
    "band_imb_5bps_chg_5s",
    "log_depth",
]
FLOW = ["ofi_5000ms", "ofi_30000ms", "trade_imb_5000ms", "trade_imb_30000ms", "trade_count_5000ms"]
PRICE = (
    [f"ret_{w}s" for w in (2, 5, 15, 30, 60, 300)]
    + [f"speed_{w}s" for w in (5, 30, 300)]
    + [f"eff_{w}s" for w in (5, 30, 300)]
    + [f"activity_{w}s" for w in (5, 30)]
    + ["accel_short", "accel_long"]
)
SPREAD_VOL = ["spread_bps", "spread_chg_5s", "spread_max_30s", "vol_5000ms_bps", "vol_30000ms_bps"]
FAMILIES = {"book": BOOK, "flow": FLOW, "price": PRICE, "spread_vol": SPREAD_VOL}
ALL = BOOK + FLOW + PRICE + SPREAD_VOL


def spearman(x, y) -> float:
    x = pd.Series(np.asarray(x, dtype=np.float64))
    y = pd.Series(np.asarray(y, dtype=np.float64))
    m = x.notna() & y.notna()
    return float(x[m].rank().corr(y[m].rank())) if m.sum() > 10 else np.nan


def ic_table(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    rows = []
    for f in features:
        daily = df.groupby("day").apply(lambda g: spearman(g[f], g[target]), include_groups=False)
        rows.append({"feature": f, "IC": spearman(df[f], df[target]), "daily_mean": daily.mean(), "ICIR": daily.mean() / daily.std() if daily.std() > 0 else np.nan})
    return pd.DataFrame(rows).set_index("feature")


def partial_ic(df: pd.DataFrame, features: list[str], target: str, control: str = "imbalance_l5") -> pd.Series:
    """Rank IC of each feature with the target residual after the control's rank."""
    m = df[target].notna()
    y = df.loc[m, target].rank().to_numpy()
    x0 = df.loc[m, control].rank().to_numpy()
    x0c = x0 - x0.mean()
    resid = y - y.mean() - (x0c @ (y - y.mean())) / (x0c @ x0c) * x0c
    return pd.Series({f: spearman(df.loc[m, f], resid) for f in features if f != control})


def interaction_table(df: pd.DataFrame, target: str, price_col: str = "ret_30s", speed_col: str = "speed_30s") -> pd.DataFrame:
    """Mean forward move (bps) by imbalance state x recent move direction (x speed)."""
    d = df.dropna(subset=[target]).copy()
    d["book"] = pd.cut(d["imbalance_l5"], [-1.01, -0.5, 0.5, 1.01], labels=["ask-heavy", "balanced", "bid-heavy"])
    thr = d[price_col].abs().median()
    d["recent"] = np.select([d[price_col] < -thr, d[price_col] > thr], ["fell", "rose"], "flat")
    d["speed"] = pd.qcut(d[speed_col].rank(method="first"), 3, labels=["slow", "mid", "fast"])
    d["move_bps"] = d[target] * 1e4
    g = d.groupby(["book", "recent", "speed"], observed=True)["move_bps"]
    t = pd.DataFrame({"n": g.size(), "mean_move_bps": g.mean(), "P(|move|>3bps)": d.assign(big=d[target].abs() > 3e-4).groupby(["book", "recent", "speed"], observed=True)["big"].mean()})
    return t.reset_index()


@dataclass
class ModelResult:
    name: str
    ic: float
    top: dict  # quantile -> (mean signed mid move bps, mean signed taker exec bps, hit rate, n)


def _fit_predict(name: str, X_tr, y_tr, X_te):
    if name == "gbm":
        model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0, random_state=0)
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(X_tr, y_tr)
    return model, model.predict(X_te)


def _design(df: pd.DataFrame, cols: list[str], interactions: bool) -> np.ndarray:
    X = df[cols].to_numpy(np.float64)
    X = np.nan_to_num(X)
    if interactions:
        imb = np.nan_to_num(df["imbalance_l5"].to_numpy(np.float64))
        extra = [imb[:, None] * np.nan_to_num(df[[c for c in cols if c != "imbalance_l5"]].to_numpy(np.float64))]
        X = np.hstack([X] + extra)
    return X


def top_bucket_stats(pred, mid_ret, long_ret, short_ret, quantiles=(0.10, 0.05, 0.01)) -> dict:
    pred = np.asarray(pred)
    out = {}
    for q in quantiles:
        cut = np.quantile(np.abs(pred), 1 - q)
        sel = np.abs(pred) >= cut
        s = np.sign(pred[sel])
        mid_signed = s * mid_ret[sel]
        exec_signed = np.where(s > 0, long_ret[sel], short_ret[sel])
        out[q] = (float(np.mean(mid_signed) * 1e4), float(np.mean(exec_signed) * 1e4), float(np.mean(mid_signed > 0)), int(sel.sum()))
    return out


MODEL_SPECS = {
    "imbalance_l5 only (linear)": (["imbalance_l5"], "linear", False),
    "price vector only (linear)": (PRICE, "linear", False),
    "price vector only (GBM)": (PRICE, "gbm", False),
    "book only (linear)": (BOOK, "linear", False),
    "all features (linear)": (ALL, "linear", False),
    "all + imbalance interactions (linear)": (ALL, "linear", True),
    "all features (GBM)": (ALL, "gbm", False),
}


def model_comparison(train: pd.DataFrame, test: pd.DataFrame, h: int) -> tuple[pd.DataFrame, object]:
    tgt = f"mid_ret_{h}"
    tr = train.dropna(subset=[tgt])
    te = test.dropna(subset=[tgt, f"long_ret_{h}", f"short_ret_{h}"])
    lo, hi = tr[tgt].quantile([0.005, 0.995])
    y = tr[tgt].clip(lo, hi).to_numpy()
    rows, gbm_all = [], None
    for name, (cols, kind, inter) in MODEL_SPECS.items():
        model, pred = _fit_predict(kind, _design(tr, cols, inter), y, _design(te, cols, inter))
        if name == "all features (GBM)":
            gbm_all = model
        st = top_bucket_stats(pred, te[tgt].to_numpy(), te[f"long_ret_{h}"].to_numpy(), te[f"short_ret_{h}"].to_numpy())
        row = {"model": name, "test_IC": spearman(pred, te[tgt])}
        for q, (mid_b, exec_b, hit, n) in st.items():
            row[f"top{q:.0%}_move_bps"] = mid_b
            row[f"top{q:.0%}_hit"] = hit
        row["top1%_taker_gross_bps"] = st[0.01][1]
        rows.append(row)
    return pd.DataFrame(rows), gbm_all


def family_importance(model, test: pd.DataFrame, h: int, seed: int = 0) -> pd.Series:
    """Drop in test IC when each feature family is shuffled (jointly)."""
    tgt = f"mid_ret_{h}"
    te = test.dropna(subset=[tgt]).reset_index(drop=True)
    base = spearman(model.predict(_design(te, ALL, False)), te[tgt])
    rng = np.random.default_rng(seed)
    out = {}
    for fam, cols in FAMILIES.items():
        shuffled = te.copy()
        perm = rng.permutation(len(te))
        shuffled[cols] = te[cols].to_numpy()[perm]
        out[fam] = base - spearman(model.predict(_design(shuffled, ALL, False)), te[tgt])
    return pd.Series(out)
