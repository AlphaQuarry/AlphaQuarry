from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_mining.engine import ExpressionEngine
from alpha_mining.panel_store import PanelStore


def _store(df: pd.DataFrame) -> PanelStore:
    return PanelStore.from_long_frame(df, group_fields=["industry"])


def test_ts_min_max_median_av_diff_covariance_and_count_nans() -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    rows = []
    xa = [1.0, 3.0, np.nan, 2.0, 4.0]
    xb = [2.0, 2.0, 2.0, 2.0, 2.0]
    ya = [1.0, 2.0, 3.0, 4.0, 5.0]
    yb = [5.0, 4.0, 3.0, 2.0, 1.0]
    for i, dt in enumerate(dates):
        rows.extend(
            [
                {"date": dt, "code": "A", "x": xa[i], "y": ya[i], "industry": "g1"},
                {"date": dt, "code": "B", "x": xb[i], "y": yb[i], "industry": "g1"},
            ]
        )
    engine = ExpressionEngine(_store(pd.DataFrame(rows)))

    out_min = engine.eval("ts_min(x, 3)")
    out_max = engine.eval("ts_max(x, 3)")
    out_med = engine.eval("ts_median(x, 3)")
    out_avg_diff = engine.eval("ts_av_diff(x, 3)")
    out_cov = engine.eval("ts_covariance(x, y, 3)")
    out_nan_cnt = engine.eval("ts_count_nans(x, 3)")
    out_mean = engine.eval("ts_mean(x, 3)")

    expected_min_a = [1.0, 1.0, 1.0, 2.0, 2.0]
    expected_max_a = [1.0, 3.0, 3.0, 3.0, 4.0]
    expected_med_a = [1.0, 2.0, 2.0, 2.5, 3.0]
    for i, dt in enumerate(dates):
        assert out_min.loc[dt, "A"] == expected_min_a[i]
        assert out_max.loc[dt, "A"] == expected_max_a[i]
        assert out_med.loc[dt, "A"] == expected_med_a[i]

    pd.testing.assert_frame_equal(out_avg_diff, out_mean * 0.0 + (engine.eval("x") - out_mean))

    ref_cov = engine.eval("x").rolling(3, min_periods=2).cov(engine.eval("y"))
    pd.testing.assert_frame_equal(out_cov, ref_cov)
    assert np.isnan(out_cov.loc[pd.Timestamp("2024-01-01"), "A"])

    assert out_nan_cnt.loc[pd.Timestamp("2024-01-01"), "A"] == 0.0
    assert out_nan_cnt.loc[pd.Timestamp("2024-01-03"), "A"] == 1.0
    assert out_nan_cnt.loc[pd.Timestamp("2024-01-04"), "A"] == 1.0


def test_group_median_and_group_scale() -> None:
    dt = pd.Timestamp("2024-02-01")
    rows = [
        {"date": dt, "code": "A", "x": 1.0, "industry": "g1"},
        {"date": dt, "code": "B", "x": 3.0, "industry": "g1"},
        {"date": dt, "code": "C", "x": 2.0, "industry": "g2"},
        {"date": dt, "code": "D", "x": 6.0, "industry": "g2"},
    ]
    engine = ExpressionEngine(_store(pd.DataFrame(rows)))
    out_med = engine.eval("group_median(x, industry)")
    out_scale = engine.eval("group_scale(x, industry)")

    assert out_med.loc[dt, "A"] == 2.0
    assert out_med.loc[dt, "B"] == 2.0
    assert out_med.loc[dt, "C"] == 4.0
    assert out_med.loc[dt, "D"] == 4.0

    assert out_scale.loc[dt, "A"] == 0.25
    assert out_scale.loc[dt, "B"] == 0.75
    assert out_scale.loc[dt, "C"] == 0.25
    assert out_scale.loc[dt, "D"] == 0.75


def test_quantile_truncate_and_tails_are_stable() -> None:
    dt = pd.Timestamp("2024-03-01")
    rows = [
        {"date": dt, "code": "A", "x": -2.0, "industry": "g1"},
        {"date": dt, "code": "B", "x": -1.0, "industry": "g1"},
        {"date": dt, "code": "C", "x": 1.0, "industry": "g2"},
        {"date": dt, "code": "D", "x": 2.0, "industry": "g2"},
    ]
    engine = ExpressionEngine(_store(pd.DataFrame(rows)))

    q_uni = engine.eval("quantile(x, 'uniform', 1.0)")
    q_gauss = engine.eval("cs_quantile(x, 'gaussian', 1.0)")
    trunc = engine.eval("truncate(x, 1.5)")
    ltail = engine.eval("left_tail(x, 0.0)")
    rtail = engine.eval("right_tail(x, 0.0)")

    assert set(np.round(q_uni.loc[dt].values, 4)) == {0.25, 0.5, 0.75, 1.0}
    assert np.isinf(q_gauss.to_numpy(dtype=float)).sum() == 0
    assert np.isnan(q_gauss.to_numpy(dtype=float)).sum() == 0

    assert trunc.loc[dt, "A"] == -1.5
    assert trunc.loc[dt, "D"] == 1.5
    assert np.isnan(ltail.loc[dt, "D"])
    assert np.isnan(rtail.loc[dt, "A"])
    assert ltail.loc[dt, "A"] == -2.0
    assert rtail.loc[dt, "D"] == 2.0


def test_hump_is_numerically_stable_and_stateful() -> None:
    dates = pd.date_range("2024-04-01", periods=6)
    values = [0.0, 0.1, 0.4, 0.35, np.nan, 0.9]
    rows = [{"date": dt, "code": "A", "x": v, "industry": "g1"} for dt, v in zip(dates, values)]
    engine = ExpressionEngine(_store(pd.DataFrame(rows)))
    out = engine.eval("hump(x, 0.2)")
    expected = [0.0, 0.0, 0.2, 0.2, 0.2, 0.4]
    for dt, exp in zip(dates, expected):
        got = float(out.loc[dt, "A"])
        assert abs(got - exp) <= 1.0e-9


def test_trade_when_hold_keeps_hold_until_exit_without_overwriting_trade_when() -> None:
    dates = pd.date_range("2024-05-01", periods=6)
    entry = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    exitv = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    alpha = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    rows = []
    for dt, e, x, a in zip(dates, entry, exitv, alpha):
        rows.append(
            {
                "date": dt,
                "code": "A",
                "entry_sig": e,
                "exit_sig": x,
                "alpha": a,
                "industry": "g1",
            }
        )
    engine = ExpressionEngine(_store(pd.DataFrame(rows)))

    out_hold = engine.eval("trade_when_hold(greater(entry_sig, 0), alpha, greater(exit_sig, 0))")
    out_simple = engine.eval("trade_when(greater(entry_sig, 0), alpha, 0)")

    expected = [10.0, 10.0, np.nan, 40.0, 40.0, np.nan]
    for dt, exp in zip(dates, expected):
        got = out_hold.loc[dt, "A"]
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            assert got == exp

    # Existing trade_when semantics must remain stateless if_else style.
    assert out_simple.loc[pd.Timestamp("2024-05-02"), "A"] == 0.0
