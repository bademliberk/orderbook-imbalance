import numpy as np
import pandas as pd
import pytest

from obi import features as F
from obi.analysis import bucket_table
from obi.labels import executable_labels
from obi.synthetic import make_book
from obi.validate import clean_top_of_book


def tiny_book():
    return pd.DataFrame(
        {
            "ts": [0, 10, 20, 30, 40],
            "update_id": [1, 2, 3, 4, 5],
            "bid": [100.0, 100.0, 101.0, 101.0, 99.0],
            "bid_qty": [3.0, 1.0, 2.0, 2.0, 1.0],
            "ask": [100.5, 100.5, 101.5, 101.5, 99.5],
            "ask_qty": [1.0, 1.0, 2.0, 6.0, 1.0],
        }
    )


def test_imbalance_values():
    np.testing.assert_allclose(F.imbalance([3, 1, 0], [1, 3, 0]), [0.5, -0.5, 0.0])


def test_multilevel_imbalance_weights_best_level_most():
    b = np.array([[10.0, 0.0]])
    a = np.array([[0.0, 10.0]])
    assert F.multilevel_imbalance(b, a, decay=0.5)[0] == pytest.approx(1 / 3)


def test_ofi_matches_definition():
    bk = tiny_book()
    e = F.order_flow_imbalance(bk.bid, bk.bid_qty, bk.ask, bk.ask_qty)
    # event 1: bid same price qty 3->1 => +1 - 3 ; ask unchanged => -1 + 1
    assert e[1] == pytest.approx(-2.0)
    # event 2: bid up (+2), ask up: -0 + 1*1
    assert e[2] == pytest.approx(2.0 + 1.0)


def test_window_sum_is_trailing_and_causal():
    ts = np.array([0, 5, 10, 20])
    v = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(F.window_sum(ts, v, 10), [1, 3, 5, 4])


def test_trade_features_exclude_same_ms_trades():
    book_ts = np.array([10, 20])
    trade_ts = np.array([5, 10, 15])
    vals = np.array([1.0, 10.0, 100.0])
    np.testing.assert_allclose(F.trade_window_sum(book_ts, trade_ts, vals, 100), [1.0, 111.0])


def test_executable_labels_use_latency_and_correct_sides():
    bk = tiny_book()
    lab = executable_labels(bk, horizons_ms=(20,), latency_ms=10, with_excursions=True)
    # decision at row 0: entry at ts=10 (ask 100.5, bid 100), exit at ts=30 (bid 101, ask 101.5)
    assert lab.loc[0, "long_ret_20"] == pytest.approx(101.0 / 100.5 - 1)
    assert lab.loc[0, "short_ret_20"] == pytest.approx(1 - 101.5 / 100.0)
    # exit beyond the data is NaN
    assert np.isnan(lab.loc[3, "long_ret_20"])
    # long MFE between entry(ts=10) and exit(ts=30): max bid 101
    assert lab.loc[0, "long_mfe_20"] == pytest.approx(101.0 / 100.5 - 1)


def test_validation_flags_crossed_and_gaps():
    bk = tiny_book()
    bk.loc[2, "bid"] = 102.0  # crossed
    bk.loc[4, "ts"] = 10_000
    clean, rep = clean_top_of_book(bk, max_gap_ms=1_000)
    assert rep.crossed_or_locked == 1
    assert len(rep.time_gaps) == 1
    assert list(clean["valid"]) == [True, True, False, True, False]


def test_bucket_table_recovers_synthetic_edge():
    book, _ = make_book(n=100_000, edge=0.8, seed=1)
    lab = executable_labels(book, horizons_ms=(100,), with_excursions=False)
    imb = pd.Series(F.imbalance(book.bid_qty, book.ask_qty))
    t = bucket_table(imb, lab, (100,), breakeven=0.0008)
    p = t["p_next_up"].to_numpy()
    assert p[0] < 0.35 and p[-1] > 0.65
    assert np.corrcoef(np.arange(len(p)), p)[0, 1] > 0.95


def test_price_vector_direction_speed_efficiency():
    ts = np.arange(0, 11_000, 1_000)
    mid = 100 * np.exp(np.array([0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8]) * 1e-4)  # mostly up, one dip
    pv = F.price_vector_features(ts, mid, windows_ms=(5_000, 10_000))
    # last 10 s: net +8 bps over a path of 10 bps
    assert pv["ret_10s"][-1] == pytest.approx(8.0, abs=1e-6)
    assert pv["speed_10s"][-1] == pytest.approx(0.8, abs=1e-6)
    assert pv["eff_10s"][-1] == pytest.approx(0.8, abs=1e-6)
    assert pv["activity_10s"][-1] == pytest.approx(1.0)


def test_band_imbalance_counts_only_levels_inside_band():
    book = pd.DataFrame(
        {
            "bid_px_0": [99.99], "bid_qty_0": [1.0], "bid_px_1": [99.90], "bid_qty_1": [10.0],
            "ask_px_0": [100.01], "ask_qty_0": [1.0], "ask_px_1": [100.10], "ask_qty_1": [1.0],
        }
    )
    out = F.band_imbalance(book, bands_bps=(2, 15))
    assert out["band_imb_2bps"][0] == pytest.approx(0.0, abs=1e-3)  # only the best levels, equal size
    assert out["band_imb_15bps"][0] > 0.6  # the deep 10-lot bid is inside 15 bps
    assert out["band_cov_2bps"][0] == 1.0 and out["band_cov_15bps"][0] == 0.0


def _agg(ts, bid_px, bid_q, ask_px, ask_q):
    d = {"ts": ts}
    for i, (bp, bq, ap, aq) in enumerate(zip(bid_px, bid_q, ask_px, ask_q)):
        d[f"bid_px_{i}"], d[f"bid_qty_{i}"], d[f"ask_px_{i}"], d[f"ask_qty_{i}"] = [bp], [bq], [ap], [aq]
    return pd.DataFrame(d)


def test_merge_depth_bands_prefers_finest_covering_book_and_is_causal():
    # fine book reaches only ~0.2%; coarse book reaches 2%
    fine = _agg([1_000], [99.99, 99.8], [1.0, 1.0], [100.01, 100.2], [1.0, 1.0])
    coarse = _agg([1_000], [99.9, 99.0, 98.0], [1.0, 9.0, 1.0], [100.1, 101.0, 102.0], [1.0, 1.0, 1.0])
    out = F.merge_depth_bands(np.array([500, 1_500]), {4: fine, 3: coarse}, bands_bps=(10, 100))
    assert np.isnan(out["depth_imb_100bps"][0])  # no snapshot yet at t=500
    assert out["depth_cov_10bps"][1] == 1.0  # fine book covers 0.1%
    assert out["depth_imb_10bps"][1] == pytest.approx(0.0, abs=1e-3)
    # 1% band only covered by the coarse book: bid 1+9 vs ask 1+1 lots inside 1%
    assert out["depth_cov_100bps"][1] == 1.0
    assert out["depth_imb_100bps"][1] > 0.6
    assert out["depth_age_ms_100bps"][1] == 500


def test_depth_backtest_excludes_windows_across_gaps():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("depth_backtest", Path(__file__).parents[1] / "scripts" / "depth_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ts = np.array([0, 1_000, 2_000, 60_000, 61_000])  # gap after t=2000 (laptop asleep)
    np.testing.assert_array_equal(mod.crosses_gap(ts, 1_500), [False, True, True, False, False])
    # a window from the last pre-gap row ends inside the gap: price unknown, excluded
    np.testing.assert_array_equal(mod.crosses_gap(ts, 500), [False, False, True, False, False])
