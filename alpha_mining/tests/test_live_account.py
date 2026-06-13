from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from alpha_mining.live.account import load_account_inputs, normalize_stock_code
from alpha_mining.live.config import LiveConfig


def test_positions_schema_standardizes_codes_and_derives_price(tmp_path: Path) -> None:
    path = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "code": ["000001", "600000.SH"],
            "shares": [1000, 200],
            "available_shares": [900, 0],
            "market_value": [10500.0, 2400.0],
            "position_date": ["2026-05-25", "2026-05-25"],
        }
    ).to_csv(path, index=False)
    cfg = LiveConfig(universe="u1", store_root=tmp_path)

    result = load_account_inputs(config=cfg, execute_date="2026-05-26", position_path=path)

    assert result["status"] == "ok"
    assert result["account"]["position_market_value"] == 12900.0
    assert result["warnings"] == ["last_price_derived_from_market_value"]
    assert result["positions"]["code"].tolist() == ["000001.SZ", "600000.SH"]
    assert result["positions"]["last_price"].round(2).tolist() == [10.5, 12.0]
    assert normalize_stock_code("300001") == "300001.SZ"
    assert normalize_stock_code("688001") == "688001.SH"
    assert normalize_stock_code("000001.XSHE") == "000001.SZ"
    assert normalize_stock_code("600000.XSHG") == "600000.SH"
    assert normalize_stock_code("000001.SZSE") == "000001.SZ"
    assert normalize_stock_code("600000.SSE") == "600000.SH"


def test_missing_position_path_is_skipped_not_directory_read(tmp_path: Path) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)

    result = load_account_inputs(config=cfg, execute_date="2026-05-26")

    assert result["status"] == "skipped"
    assert result["reason"] == "position_path_missing"


def test_stale_positions_block_by_default(tmp_path: Path) -> None:
    path = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "shares": [100],
            "available_shares": [100],
            "last_price": [10.0],
            "position_date": ["2026-05-22"],
        }
    ).to_csv(path, index=False)
    cfg = LiveConfig(universe="u1", store_root=tmp_path)

    result = load_account_inputs(config=cfg, execute_date="2026-05-26", position_path=path)

    assert result["status"] == "blocked"
    assert "stale_positions" in result["blocking_reasons"]


def test_position_staleness_uses_trading_days_not_calendar_days(tmp_path: Path) -> None:
    db_path = tmp_path / "market.duckdb"
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-22", "2026-05-25"]),
            "code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 10.5],
            "universe": [1, 1],
        }
    )
    conn = duckdb.connect(str(db_path))
    try:
        conn.register("panel_df", panel)
        conn.execute("CREATE TABLE panel AS SELECT * FROM panel_df")
    finally:
        conn.close()
    path = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "shares": [100],
            "available_shares": [100],
            "last_price": [10.0],
            "position_date": ["2026-05-22"],
        }
    ).to_csv(path, index=False)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    cfg.account.max_position_staleness_trade_days = 1

    result = load_account_inputs(config=cfg, execute_date="2026-05-25", position_path=path)

    assert result["status"] == "ok"


def test_missing_account_value_only_allows_weight_delta(tmp_path: Path) -> None:
    path = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "shares": [0],
            "available_shares": [0],
            "last_price": [10.0],
            "position_date": ["2026-05-26"],
        }
    ).to_csv(path, index=False)
    cfg = LiveConfig(universe="u1", store_root=tmp_path)

    result = load_account_inputs(config=cfg, execute_date="2026-05-26", position_path=path)

    assert result["status"] == "ok"
    assert result["account"]["account_total_value"] is None
    assert result["money_reliable"] is False
