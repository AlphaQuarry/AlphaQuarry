from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from alpha_mining.live.config import LiveConfig
from alpha_mining.live.readiness import check_live_readiness
from alpha_mining.live.registry import activate_superalpha


def _setup_ready_store(root: Path) -> Path:
    db_path = root / "market.duckdb"
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26", "2026-05-27"] * 2),
            "code": [
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "000002.SZ",
                "000002.SZ",
            ],
            "close": [10.0, 10.5, 10.8, 20.0, 20.5, 20.8],
            "can_buy": [1, 1, 1, 1, 1, 1],
            "can_sell": [1, 1, 1, 1, 1, 1],
            "is_st": [0, 0, 0, 0, 0, 0],
            "is_suspended": [0, 0, 0, 0, 0, 0],
            "up_limit": [11.0, 11.5, 11.8, 21.0, 21.5, 21.8],
            "down_limit": [9.0, 9.5, 9.8, 19.0, 19.5, 19.8],
            "is_limit_up_close": [0, 0, 0, 0, 0, 0],
            "is_limit_down_close": [0, 0, 0, 0, 0, 0],
            "circ_mv": [1000.0, 1001.0, 1002.0, 2000.0, 2001.0, 2002.0],
            "universe": [1, 1, 1, 1, 1, 1],
        }
    )
    conn = duckdb.connect(str(db_path))
    try:
        conn.register("panel_df", panel)
        conn.execute("CREATE TABLE panel AS SELECT * FROM panel_df")
    finally:
        conn.close()
    run_dir = root / "u1" / "superalphas" / "sa1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "superalpha_id": "sa1",
                "universe": "u1",
                "combo_expression": "1",
                "component_count": 1,
                "components": [
                    {
                        "factor": "alpha_a",
                        "expression": "ts_mean(close, 2)",
                        "weight": 1.0,
                    }
                ],
                "component_normalization": "none",
                "final_normalization": "none",
                "direction_adjustment": True,
                "period": 1,
                "layers": 10,
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    activate_superalpha(base_dir=root, universe="u1", superalpha_id="sa1")
    return db_path


def test_check_live_readiness_returns_warn_when_positions_are_missing(tmp_path: Path, monkeypatch) -> None:
    db_path = _setup_ready_store(tmp_path)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    # Prevent loading the real field catalog artifact from cwd.
    monkeypatch.setattr(
        "alpha_mining.live.data_status.load_live_field_catalog",
        lambda *, config: {"status": "missing", "source": "", "fields": {}},
    )

    result = check_live_readiness(config=cfg, requested_date="2026-05-26")

    assert result["status"] == "WARN"
    assert "position_path_missing" in result["warnings"]
    assert result["checks"]["data_status"]["status"] == "ready"


def test_check_live_readiness_can_write_json_out(tmp_path: Path, monkeypatch) -> None:
    db_path = _setup_ready_store(tmp_path)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    # Prevent loading the real field catalog artifact from cwd.
    monkeypatch.setattr(
        "alpha_mining.live.data_status.load_live_field_catalog",
        lambda *, config: {"status": "missing", "source": "", "fields": {}},
    )
    out = tmp_path / "readiness.json"

    result = check_live_readiness(config=cfg, requested_date="2026-05-26", json_out=out)

    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == result["status"]


def test_check_live_readiness_uses_execute_date_for_position_staleness(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.duckdb"
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-22", "2026-05-25"] * 2),
            "code": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "close": [10.0, 10.5, 20.0, 20.5],
            "can_buy": [1, 1, 1, 1],
            "can_sell": [1, 1, 1, 1],
            "is_st": [0, 0, 0, 0],
            "is_suspended": [0, 0, 0, 0],
            "up_limit": [11.0, 11.5, 21.0, 21.5],
            "down_limit": [9.0, 9.5, 19.0, 19.5],
            "is_limit_up_close": [0, 0, 0, 0],
            "is_limit_down_close": [0, 0, 0, 0],
            "circ_mv": [1000.0, 1001.0, 2000.0, 2001.0],
            "universe": [1, 1, 1, 1],
        }
    )
    catalog = pd.DataFrame(
        {
            "field_name": [
                "close",
                "can_buy",
                "can_sell",
                "is_st",
                "is_suspended",
                "up_limit",
                "down_limit",
                "is_limit_up_close",
                "is_limit_down_close",
                "circ_mv",
            ],
            "available_at": ["same_day_close_available"] * 10,
            "leakage_safe": [True] * 10,
            "field_role": ["live"] * 10,
            "source_table": ["panel"] * 10,
        }
    )
    conn = duckdb.connect(str(db_path))
    try:
        conn.register("panel_df", panel)
        conn.register("catalog_df", catalog)
        conn.execute("CREATE TABLE panel AS SELECT * FROM panel_df")
        conn.execute("CREATE TABLE v_project_field_catalog AS SELECT * FROM catalog_df")
    finally:
        conn.close()
    run_dir = tmp_path / "u1" / "superalphas" / "sa1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "superalpha_id": "sa1",
                "universe": "u1",
                "combo_expression": "1",
                "component_count": 1,
                "components": [
                    {
                        "factor": "alpha_a",
                        "expression": "ts_mean(close, 2)",
                        "weight": 1.0,
                    }
                ],
                "component_normalization": "none",
                "final_normalization": "none",
                "direction_adjustment": True,
                "period": 1,
                "layers": 10,
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    activate_superalpha(base_dir=tmp_path, universe="u1", superalpha_id="sa1")
    positions_path = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "shares": [100],
            "available_shares": [100],
            "last_price": [10.0],
            "market_value": [1000.0],
            "position_date": ["2026-05-22"],
        }
    ).to_csv(positions_path, index=False)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    cfg.account.max_position_staleness_trade_days = 1

    result = check_live_readiness(
        config=cfg,
        requested_date="2026-05-22",
        position_path=positions_path,
        account_overrides={"account_total_value": 10000.0, "cash": 9000.0},
    )

    assert result["signal_date"] == "2026-05-22"
    assert result["execute_date"] == "2026-05-25"
    assert "stale_positions" not in result["warnings"]
    assert "stale_positions" not in result["blocking_reasons"]
    assert result["checks"]["account"]["status"] == "ok"
