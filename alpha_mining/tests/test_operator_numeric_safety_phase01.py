from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_mining.engine import ExpressionEngine
from alpha_mining.panel_store import PanelStore


def _store(rows: list[dict[str, object]]) -> PanelStore:
    return PanelStore.from_long_frame(pd.DataFrame(rows), group_fields=["industry"])


def test_divide_and_inverse_guard_zero_and_near_zero_denominator() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=2):
        rows.extend(
            [
                {"date": date, "code": "A", "x": 1.0, "y": 0.0, "industry": "g1"},
                {"date": date, "code": "B", "x": -2.0, "y": 1.0e-14, "industry": "g1"},
            ]
        )
    engine = ExpressionEngine(_store(rows))

    out_div = engine.eval("divide(x, y)")
    out_inv = engine.eval("inverse(y)")

    assert np.isinf(out_div.to_numpy(dtype=float)).sum() == 0
    assert np.isinf(out_inv.to_numpy(dtype=float)).sum() == 0
    assert np.isnan(out_div.loc[pd.Timestamp("2024-01-01"), "A"])
    assert np.isnan(out_div.loc[pd.Timestamp("2024-01-01"), "B"])
    assert np.isnan(out_inv.loc[pd.Timestamp("2024-01-01"), "A"])
    assert np.isnan(out_inv.loc[pd.Timestamp("2024-01-01"), "B"])


def test_log_and_sqrt_return_nan_on_invalid_domain() -> None:
    rows = []
    for date in pd.date_range("2024-01-01", periods=1):
        rows.extend(
            [
                {"date": date, "code": "A", "x": -1.0, "industry": "g1"},
                {"date": date, "code": "B", "x": 0.0, "industry": "g1"},
                {"date": date, "code": "C", "x": 4.0, "industry": "g1"},
            ]
        )
    engine = ExpressionEngine(_store(rows))

    out_log = engine.eval("log(x)")
    out_sqrt = engine.eval("sqrt(x)")

    assert np.isnan(out_log.loc[pd.Timestamp("2024-01-01"), "A"])
    assert np.isnan(out_log.loc[pd.Timestamp("2024-01-01"), "B"])
    assert out_log.loc[pd.Timestamp("2024-01-01"), "C"] == np.log(4.0)

    assert np.isnan(out_sqrt.loc[pd.Timestamp("2024-01-01"), "A"])
    assert out_sqrt.loc[pd.Timestamp("2024-01-01"), "B"] == 0.0
    assert out_sqrt.loc[pd.Timestamp("2024-01-01"), "C"] == 2.0

    assert np.isinf(out_log.to_numpy(dtype=float)).sum() == 0
    assert np.isinf(out_sqrt.to_numpy(dtype=float)).sum() == 0


def test_ts_arg_max_min_follow_relative_lag_semantics() -> None:
    # oldest -> latest values are [4, 9, 5, 8, 2, 6],
    # equivalent WQ "today first" view is [6, 2, 8, 5, 9, 4]:
    # argmax lag = 4, argmin lag = 1
    values = [4.0, 9.0, 5.0, 8.0, 2.0, 6.0]
    rows = [
        {"date": date, "code": "A", "x": value, "industry": "g1"}
        for date, value in zip(pd.date_range("2024-01-01", periods=6), values)
    ]
    engine = ExpressionEngine(_store(rows))

    out_max = engine.eval("ts_arg_max(x, 6)")
    out_min = engine.eval("ts_arg_min(x, 6)")

    last_date = pd.Timestamp("2024-01-06")
    assert out_max.loc[last_date, "A"] == 4.0
    assert out_min.loc[last_date, "A"] == 1.0


def test_ts_arg_max_min_ignore_nan_and_keep_nan_for_all_nan_window() -> None:
    rows = [
        {
            "date": pd.Timestamp("2024-01-01"),
            "code": "A",
            "x": np.nan,
            "industry": "g1",
        },
        {
            "date": pd.Timestamp("2024-01-02"),
            "code": "A",
            "x": np.nan,
            "industry": "g1",
        },
        {"date": pd.Timestamp("2024-01-03"), "code": "A", "x": 1.0, "industry": "g1"},
        {
            "date": pd.Timestamp("2024-01-04"),
            "code": "A",
            "x": np.nan,
            "industry": "g1",
        },
        {"date": pd.Timestamp("2024-01-05"), "code": "A", "x": 2.0, "industry": "g1"},
    ]
    engine = ExpressionEngine(_store(rows))

    out_max = engine.eval("ts_arg_max(x, 2)")
    out_min = engine.eval("ts_arg_min(x, 2)")

    assert np.isnan(out_max.loc[pd.Timestamp("2024-01-02"), "A"])
    assert np.isnan(out_min.loc[pd.Timestamp("2024-01-02"), "A"])
    assert out_max.loc[pd.Timestamp("2024-01-05"), "A"] == 0.0
    assert out_min.loc[pd.Timestamp("2024-01-05"), "A"] == 0.0


def test_group_zscore_and_ts_regression_zero_variance_are_nan_not_inf() -> None:
    rows = []
    for date, yv in zip(pd.date_range("2024-01-01", periods=5), [1.0, 2.0, 3.0, 4.0, 5.0]):
        rows.extend(
            [
                {
                    "date": date,
                    "code": "A",
                    "x": 1.0,
                    "y": yv,
                    "z": 10.0,
                    "industry": "g1",
                },
                {
                    "date": date,
                    "code": "B",
                    "x": 1.0,
                    "y": yv + 1.0,
                    "z": 10.0,
                    "industry": "g1",
                },
            ]
        )
    engine = ExpressionEngine(_store(rows))

    out_group = engine.eval("group_zscore(z, industry)")
    out_beta = engine.eval("ts_regression(y, x, 3)")
    out_corr = engine.eval("ts_corr(y, x, 3)")

    assert np.isinf(out_group.to_numpy(dtype=float)).sum() == 0
    assert np.isinf(out_beta.to_numpy(dtype=float)).sum() == 0
    assert np.isinf(out_corr.to_numpy(dtype=float)).sum() == 0

    assert np.isnan(out_group.to_numpy(dtype=float)).all()
    assert np.isnan(out_beta.to_numpy(dtype=float)).all()


def test_days_from_last_change_keeps_current_compatibility_zero_based() -> None:
    # consecutive unchanged values [2,2,2] produce [0,1,2] on this branch by design.
    values = [1.0, 16.0, 5.0, 7.0, 2.0, 2.0, 2.0]
    rows = [
        {"date": date, "code": "A", "x": value, "industry": "g1"}
        for date, value in zip(pd.date_range("2024-01-01", periods=len(values)), values)
    ]
    engine = ExpressionEngine(_store(rows))
    out = engine.eval("days_from_last_change(x)")
    assert out.loc[pd.Timestamp("2024-01-07"), "A"] == 2.0


def test_binary_op_power_and_modulo_clean_inf() -> None:
    """** and % should produce nan, not inf, on edge cases."""
    from alpha_mining.engine import _binary_op

    # 0 ** -1 -> inf -> nan
    result = _binary_op("**", 0.0, -1)
    assert result is np.nan or (isinstance(result, float) and np.isnan(result))

    # DataFrame ** negative with zeros
    df = pd.DataFrame({"A": [0.0, 2.0, -3.0]})
    out = _binary_op("**", df, -1)
    assert np.isinf(out.to_numpy(dtype=float, na_value=np.nan)).sum() == 0

    # modulo by zero — scalar
    result_mod = _binary_op("%", 5.0, 0)
    assert result_mod is np.nan or (isinstance(result_mod, float) and np.isnan(result_mod))

    # modulo by zero — DataFrame
    right = pd.DataFrame({"A": [0.0, 2.0, 0.0]})
    out_mod = _binary_op("%", df, right)
    assert np.isinf(out_mod.to_numpy(dtype=float, na_value=np.nan)).sum() == 0
    assert np.isnan(out_mod.loc[0, "A"])
    assert out_mod.loc[1, "A"] == 0.0  # 2.0 % 2.0 = 0.0


def test_binary_op_ndarray_division_clean() -> None:
    """ndarray zero denominator should produce nan, not inf."""
    from alpha_mining.engine import _binary_op

    left = np.array([1.0, 2.0, 3.0])
    right = np.array([0.0, 1.0, 0.0])
    out = _binary_op("/", left, right)
    assert isinstance(out, np.ndarray)
    assert np.isinf(out).sum() == 0
    assert np.isnan(out[0])
    assert np.isnan(out[2])
    assert out[1] == 2.0

    # ndarray modulo by zero
    out_mod = _binary_op("%", left, right)
    assert np.isinf(out_mod).sum() == 0
    assert np.isnan(out_mod[0])


def test_binary_op_series_division_ndarray_clean() -> None:
    """Series / ndarray(0) should produce nan, not inf."""
    from alpha_mining.engine import _binary_op

    s = pd.Series([10.0, 20.0, 30.0])
    right = np.array([0.0, 5.0, 0.0])
    out = _binary_op("/", s, right)
    assert isinstance(out, pd.Series)
    assert np.isinf(out.to_numpy(dtype=float)).sum() == 0
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 4.0
