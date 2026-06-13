from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from alpha_mining.live.config import LiveConfig
from alpha_mining.live.calendar import resolve_execute_date
from alpha_mining.live.data_status import check_live_data_status
from alpha_mining.live.field_catalog import load_live_field_catalog
from alpha_mining.live.registry import activate_superalpha


def _prepare_duckdb(path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-21", "2026-05-22", "2026-05-25"] * 2),
            "code": ["000001.SZ"] * 3 + ["000002.SZ"] * 3,
            "close": [10.0, 11.0, None, 20.0, 21.0, None],
            "volume": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
            "can_buy": [1, 1, 1, 1, 1, 1],
            "can_sell": [1, 1, 1, 1, 1, 1],
            "is_st": [0, 0, 0, 0, 0, 0],
            "is_suspended": [0, 0, 0, 0, 0, 0],
            "up_limit": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
            "down_limit": [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
            "is_limit_up_close": [0, 0, 0, 0, 0, 0],
            "is_limit_down_close": [0, 0, 0, 0, 0, 0],
            "circ_mv": [1000.0] * 6,
            "universe": [1] * 6,
        }
    )
    conn = duckdb.connect(str(path))
    try:
        conn.register("panel_df", df)
        conn.execute("CREATE TABLE panel AS SELECT * FROM panel_df")
    finally:
        conn.close()


def _activate(root: Path) -> None:
    run_dir = root / "u1" / "superalphas" / "superalpha_demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        """
        {
          "superalpha_id": "superalpha_demo",
          "universe": "u1",
          "combo_expression": "1",
          "component_count": 1,
          "components": [{"factor": "alpha_a", "expression": "ts_mean(close, 2)", "weight": 1.0}],
          "component_normalization": "cs_zscore",
          "final_normalization": "cs_zscore",
          "direction_adjustment": true,
          "period": 1,
          "layers": 10,
          "summary": {}
        }
        """,
        encoding="utf-8",
    )
    activate_superalpha(base_dir=root, universe="u1", superalpha_id="superalpha_demo")


def test_common_ready_date_uses_required_field_latest_non_null_date(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "market.duckdb"
    _prepare_duckdb(db_path)
    _activate(tmp_path)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    # Prevent loading the real field catalog artifact from cwd.
    monkeypatch.setattr(
        "alpha_mining.live.data_status.load_live_field_catalog",
        lambda *, config: {"status": "missing", "source": "", "fields": {}},
    )

    status = check_live_data_status(config=cfg)

    assert status["status"] == "ready"
    assert status["common_ready_date"] == "2026-05-22"
    assert status["resolved_signal_date"] == "2026-05-22"
    assert status["ready_field_count"] >= 1
    close = next(row for row in status["fields"] if row["field"] == "close")
    assert close["field_latest_non_null_date"] == "2026-05-22"


def test_tradability_strict_missing_field_blocks(tmp_path: Path) -> None:
    db_path = tmp_path / "market.duckdb"
    _prepare_duckdb(db_path)
    _activate(tmp_path)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    cfg.tradability.critical_fields = ("close", "can_buy", "missing_flag")

    status = check_live_data_status(config=cfg)

    assert status["status"] == "data_not_ready"
    assert "missing_flag" in status["blocking_fields"]


def test_execute_date_uses_trading_day_sequence(tmp_path: Path) -> None:
    db_path = tmp_path / "market.duckdb"
    _prepare_duckdb(db_path)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")

    resolved = resolve_execute_date(config=cfg, signal_date="2026-05-22")
    specified = resolve_execute_date(config=cfg, signal_date="2026-05-22", requested_execute_date="2026-05-24")

    assert resolved["execute_date"] == "2026-05-25"
    assert specified["valid"] is False
    assert "not a trading date" in specified["warnings"][0]


def _prepare_duckdb_with_catalog(
    path: Path,
    *,
    leakage_safe: bool = True,
    available_at: str = "same_day_close_available",
) -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25"] * 3),
            "code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "close": [10.0, 20.0, 30.0],
            "can_buy": [1, 1, 1],
            "can_sell": [1, 1, 1],
            "is_st": [0, 0, 0],
            "is_suspended": [0, 0, 0],
            "up_limit": [11.0, 21.0, 31.0],
            "down_limit": [9.0, 19.0, 29.0],
            "is_limit_up_close": [0, 0, 0],
            "is_limit_down_close": [0, 0, 0],
            "circ_mv": [None, None, None],
            "total_mv": [1000.0, 2000.0, 3000.0],
            "universe": [1, 1, 1],
        }
    )
    catalog = pd.DataFrame(
        {
            "field_name": ["close", "total_mv", "circ_mv"],
            "available_at": [
                available_at,
                "same_day_close_available",
                "same_day_close_available",
            ],
            "leakage_safe": [leakage_safe, True, True],
            "field_role": ["signal_input", "tradability", "tradability"],
            "source_table": ["panel", "panel", "panel"],
        }
    )
    conn = duckdb.connect(str(path))
    try:
        conn.register("panel_df", df)
        conn.register("catalog_df", catalog)
        conn.execute("CREATE TABLE panel AS SELECT * FROM panel_df")
        conn.execute("CREATE TABLE v_project_field_catalog AS SELECT * FROM catalog_df")
    finally:
        conn.close()


def test_field_catalog_leakage_safe_false_blocks_live_data_status(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.duckdb"
    _prepare_duckdb_with_catalog(db_path, leakage_safe=False)
    _activate(tmp_path)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")

    status = check_live_data_status(config=cfg, requested_date="2026-05-25")

    close = next(row for row in status["fields"] if row["field"] == "close")
    assert status["status"] == "data_not_ready"
    assert close["catalog_status"] == "ok"
    assert close["leakage_safe"] is False
    assert "close" in status["blocking_fields"]


def test_strict_available_at_blocks_late_catalog_field(tmp_path: Path) -> None:
    db_path = tmp_path / "market.duckdb"
    _prepare_duckdb_with_catalog(db_path, available_at="next_day_after_close")
    _activate(tmp_path)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    cfg.data.strict_available_at = True

    status = check_live_data_status(config=cfg, requested_date="2026-05-25")

    assert status["status"] == "data_not_ready"
    assert "close" in status["blocking_fields"]


def test_market_value_any_of_selects_field_with_sufficient_non_null_rate(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "market.duckdb"
    _prepare_duckdb_with_catalog(db_path)
    _activate(tmp_path)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    cfg.tradability.market_value_any_of = ("circ_mv", "total_mv")

    status = check_live_data_status(config=cfg, requested_date="2026-05-25")

    assert status["status"] == "ready"
    assert status["selected_market_value_field"] == "total_mv"
    assert status["selected_market_value_non_null_rate"] == 1.0


def test_catalog_missing_policy_block_does_not_raise_name_error(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "market.duckdb"
    _prepare_duckdb(db_path)
    _activate(tmp_path)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")
    cfg.data.catalog_missing_policy = "block"
    monkeypatch.setattr(
        "alpha_mining.live.data_status.load_live_field_catalog",
        lambda *, config: {"status": "missing", "source": "", "fields": {}},
    )

    status = check_live_data_status(config=cfg)

    assert status["status"] == "data_not_ready"
    assert "field_catalog_missing" in status["blocking_fields"]


def test_live_field_catalog_artifact_source_uses_safe_label(tmp_path: Path) -> None:
    db_path = tmp_path / "market.duckdb"
    _prepare_duckdb(db_path)
    catalog_dir = tmp_path / "data" / "lake" / "meta"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "field_name": ["close"],
            "available_at": ["same_day_close_available"],
            "leakage_safe": [True],
            "field_role": ["signal_input"],
            "source_table": ["panel"],
        }
    ).to_parquet(catalog_dir / "field_catalog.parquet", index=False)
    cfg = LiveConfig(universe="u1", store_root=tmp_path, duckdb_path=db_path, source_view="panel")

    catalog = load_live_field_catalog(config=cfg)

    assert catalog["status"] == "ok"
    assert catalog["source"] == "artifact:field_catalog.parquet"
    assert "datasource.local.yaml" not in str(catalog)
