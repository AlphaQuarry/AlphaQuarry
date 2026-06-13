from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from alpha_mining.live.config import LiveConfig
from alpha_mining.live.lookback import estimate_expression_lookback
from alpha_mining.live.signal import build_live_superalpha_signal


def _snapshot() -> dict:
    return {
        "superalpha_id": "superalpha_demo",
        "universe": "u1",
        "combo_expression": "1",
        "component_join": "concat",
        "component_normalization": "none",
        "final_normalization": "none",
        "direction_adjustment": True,
        "components": [
            {
                "factor": "alpha_a",
                "expression": "ts_mean(close, 2)",
                "weight": 1.0,
                "direction_sign": 1,
            }
        ],
    }


def test_estimate_expression_lookback_warns_on_unknown_ts_operator() -> None:
    out = estimate_expression_lookback("ts_mystery(close, 99) + ts_mean(open, 5)", buffer=2)

    assert out["max_lookback"] == 7
    assert out["warnings"]


def test_estimate_expression_lookback_handles_nested_ts_operators() -> None:
    first = estimate_expression_lookback("ts_mean(rank(close), 20)", buffer=0)
    second = estimate_expression_lookback("ts_corr(ts_rank(x, 5), y, 20)", buffer=0)
    third = estimate_expression_lookback("ts_zscore(ts_delta(close, 5), 60)", buffer=0)

    assert first["max_lookback"] == 20
    assert second["max_lookback"] == 20
    assert third["max_lookback"] == 60


def test_live_signal_outputs_only_signal_date_and_not_full_history(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.duckdb"
    dates = pd.date_range("2026-05-18", periods=6, freq="B")
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "code": ["000001.SZ"] * len(dates) + ["000002.SZ"] * len(dates),
            "close": [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25],
            "universe": [1] * (len(dates) * 2),
        }
    )
    conn = duckdb.connect(str(db_path))
    try:
        conn.register("panel_df", df)
        conn.execute("CREATE TABLE panel AS SELECT * FROM panel_df")
    finally:
        conn.close()
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    cfg.superalpha.live_window_trade_days = 2
    cfg.superalpha.lookback_buffer_days = 0

    result = build_live_superalpha_signal(config=cfg, snapshot=_snapshot(), signal_date="2026-05-25")

    assert result["status"] == "ok"
    out = pd.read_parquet(result["signal_path"])
    assert out["date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2026-05-25"]
    assert sorted(out["code"].tolist()) == ["000001.SZ", "000002.SZ"]
    assert not (tmp_path / "u1" / "superalphas" / "superalpha_demo" / "superalpha_values.parquet").exists()
    assert result["window_start_date"] >= "2026-05-20"
