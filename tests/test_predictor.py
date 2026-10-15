import numpy as np
import pandas as pd
import pytest

from obi.costs import FeeSchedule
from obi.data.hyperliquid import parse_bbo, parse_book, parse_trades
from obi.dry_run import DryRun, LimitOrder, OnlineFeatures
from obi.predictor import BucketPredictor, direction_class, mid_direction_class, score


def test_direction_classes():
    np.testing.assert_array_equal(direction_class([0.002, 0.0, -0.001], [-0.003, 0.002, 0.0], 0.001), [1, -1, 0])
    np.testing.assert_array_equal(mid_direction_class([0.0004, -0.0004, 0.0001], 0.0003), [1, -1, 0])


def frame(n=30_000, seed=0, h=60_000):
    rng = np.random.default_rng(seed)
    imb = rng.uniform(-1, 1, n)
    move = 0.002 * imb + rng.normal(0, 0.001, n)
    return pd.DataFrame(
        {
            "imbalance": imb,
            "imbalance_l5": imb,
            "vol_5000ms_bps": rng.gamma(2, 1, n),
            f"mid_ret_{h}": move,
            f"long_ret_{h}": move,
            f"short_ret_{h}": -move,
        }
    )


def test_bucket_predictor_learns_and_roundtrips(tmp_path):
    m = BucketPredictor(threshold=0.001, signal_feature="imbalance_l5").fit(frame())
    p = m.predict_proba([0.95, -0.95], [1.0, 1.0])
    assert p[0, 2] > 0.6 and p[1, 0] > 0.6
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert list(m.predict([0.95, 0.0, -0.95], [1, 1, 1], min_prob=0.5)) == [1, 0, -1]
    m.save(tmp_path / "m.json")
    m2 = BucketPredictor.load(tmp_path / "m.json")
    assert m2.signal_feature == "imbalance_l5" and m2.label == "mid"
    np.testing.assert_allclose(m2.predict_proba([0.3], [2.0]), m.predict_proba([0.3], [2.0]))


def test_score_pnl():
    s = score(np.array([1, -1, 0]), np.array([1, 0, 0]), np.array([0.003, 0, 0]), np.array([0, 0.0005, 0]), 0.001)
    assert s["up_precision"] == 1.0 and s["down_precision"] == 0.0
    assert s["net_bps_per_call"] == pytest.approx(((0.003 - 0.001) + (0.0005 - 0.001)) / 2 * 1e4)


def hl_book(t, bid=100.0, ask=101.0, bq=9.0, aq=1.0):
    lv = lambda px, sz, k: [{"px": str(px - k), "sz": str(sz), "n": 1} for k in range(5)]  # noqa: E731
    bids = lv(bid, bq, 0)[:1] + [{"px": str(bid - k), "sz": "1", "n": 1} for k in range(1, 5)]
    asks = lv(ask, aq, 0)[:1] + [{"px": str(ask + k), "sz": "1", "n": 1} for k in range(1, 5)]
    return parse_book({"coin": "BTC", "time": t, "levels": [bids, asks]})


def test_hyperliquid_parsers():
    b = hl_book(1000)
    assert (b["ts"], b["bid"], b["bid_qty"], b["ask"], b["ask_qty"]) == (1000, 100.0, 9.0, 101.0, 1.0)
    assert b["ask_px_4"] == 105.0
    q = parse_bbo({"coin": "BTC", "time": 5, "bbo": [{"px": "1", "sz": "2", "n": 1}, {"px": "3", "sz": "4", "n": 1}]})
    assert q == (5, 1.0, 2.0, 3.0, 4.0)
    t = parse_trades([{"coin": "BTC", "side": "B", "px": "10", "sz": "0.5", "time": 7, "hash": "", "tid": 1, "users": []}])
    assert t == [(7, "BTC", 10.0, 0.5, 1)]


def test_online_features_vol_window():
    f = OnlineFeatures(vol_window_ms=1_000)
    f.update(0, hl_book(0, 100, 101))
    f.update(500, hl_book(500, 101, 102))
    assert f.values["vol_5000ms_bps"] > 0
    f.update(2_000, hl_book(2_000, 101, 102))  # first return falls out of the window
    assert f.values["vol_5000ms_bps"] == pytest.approx(0.0)
    assert f.values["imbalance"] == pytest.approx(0.8)
    assert "imbalance_l5" in f.values


def test_limit_order_queue_and_trade_through():
    o = LimitOrder(side=1, price=100.0, queue=3.0, t_post=0, t_cancel=100)
    o.on_trade(10, 100.0, 2.0, 1)  # buy aggressor can't fill a resting buy
    o.on_trade(20, 100.0, 2.0, -1)
    assert o.filled_at is None
    o.on_trade(30, 100.0, 1.5, -1)
    assert o.filled_at == 30
    o2 = LimitOrder(side=-1, price=101.0, queue=50.0, t_post=0, t_cancel=100)
    o2.on_trade(40, 101.5, 0.1, 1)  # traded through our ask
    assert o2.filled_at == 40


def make_dry(tmp_path, **kw):
    m = BucketPredictor(threshold=0.001, horizon_ms=1_000, signal_feature="imbalance").fit(frame(h=1_000))
    d = DryRun(m, fees=FeeSchedule(0.00015, 0.00045), every_s=100, latency_ms=10, min_prob=0.5,
               entry_timeout_ms=200, exit_timeout_ms=200, order_qty=1.0, out=tmp_path / "d.csv", **kw)
    d.feats.vol_window_ms = 0
    return d


def test_dry_run_verifies_and_simulates_limit_round_trip(tmp_path):
    d = make_dry(tmp_path)
    d.on_book(hl_book(0, 100, 101, bq=9, aq=1))  # imbalance 0.8 -> up call
    assert d.open_preds[0].call == 1
    d.on_bbo(20, 100.0, 2.0, 101.0, 1.0)  # entry: post buy at 100 behind 2
    d.on_trade(50, 100.0, 3.5, -1)  # queue 2 + own 1 depleted -> filled
    d.on_bbo(1_020, 101.0, 1.0, 102.0, 1.0)  # verify: mid 100.5 -> 101.5 = +99.5 bps
    assert d.done[0][1] == 1
    d.on_bbo(1_070, 101.0, 1.0, 102.0, 1.0)  # hold over (fill 50 + 1000 + 10): post sell at 102
    d.on_trade(1_100, 102.5, 0.1, 1)  # traded through -> exit filled
    d.on_bbo(1_120, 101.0, 1.0, 102.0, 1.0)
    assert len(d.sims) == 1 and d.sims[0]["exit_type"] == "limit"
    assert d.sims[0]["net_bps"] == pytest.approx((102 / 100 * (1 - 0.00015) - 1 - 0.00015) * 1e4)
    assert d.open_preds == []


def test_dry_run_unfilled_entry_and_gap_drop(tmp_path):
    d = make_dry(tmp_path)
    d.on_book(hl_book(0, 100, 101, bq=9, aq=1))
    d.on_bbo(20, 100.0, 2.0, 101.0, 1.0)
    d.on_bbo(300, 100.0, 2.0, 101.0, 1.0)  # entry timeout passed, nothing traded
    assert d.sims == [] and d.open_preds[0].sim["entry_filled"] is False
    d.on_bbo(5_000, 100.0, 2.0, 101.0, 1.0)  # verification quote 4 s late -> dropped
    assert d.done == [] and d.dropped == 1 and d.open_preds == []
