from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_mining.datasource.quality import (
    build_panel_quality_report,
    build_tushare_smoke_plan,
    render_panel_quality_markdown,
)


def test_panel_quality_report_fails_missing_required_and_high_nan_inf() -> None:
    df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2026-04-20"),
                pd.Timestamp("2026-04-20"),
                pd.Timestamp("2026-04-21"),
            ],
            "code": ["000001.SZ", "000002.SZ", "000001.SZ"],
            "close": [1.0, np.nan, np.inf],
            "moneyflow_net_mf_amount": [np.nan, np.nan, 3.0],
        }
    )

    report = build_panel_quality_report(
        df,
        expected_fields=["close", "moneyflow_net_mf_amount", "cyq_winner_rate"],
        required_fields=["close", "cyq_winner_rate"],
        max_missing_ratio=0.50,
        max_inf_ratio=0.0,
    )

    assert report["overall_status"] == "fail"
    by_field = {item["field"]: item for item in report["fields"]}
    assert by_field["close"]["inf_ratio"] > 0.0
    assert by_field["close"]["status"] == "fail"
    assert by_field["moneyflow_net_mf_amount"]["missing_ratio"] > 0.50
    assert by_field["moneyflow_net_mf_amount"]["status"] == "warn"
    assert by_field["cyq_winner_rate"]["present"] is False
    assert by_field["cyq_winner_rate"]["status"] == "fail"


def test_panel_quality_markdown_contains_summary_and_fields() -> None:
    report = build_panel_quality_report(
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-04-21")],
                "code": ["000001.SZ"],
                "close": [1.0],
            }
        ),
        expected_fields=["close"],
        required_fields=["close"],
    )

    text = render_panel_quality_markdown(report)

    assert "# Panel Quality Report" in text
    assert "overall_status" in text
    assert "close" in text


def test_tushare_smoke_plan_is_dry_run_and_marks_high_cost_tables() -> None:
    plan = build_tushare_smoke_plan(
        start_date="2026-04-20",
        end_date="2026-04-21",
        fact_groups="p3,p4,p5",
        fact_tables="",
        exclude_fact_tables="",
    )

    assert plan["mode"] == "dry_run"
    assert "moneyflow" in plan["selected_fact_tables"]
    assert "report_rc" in plan["selected_fact_tables"]
    high_cost = {item["table"] for item in plan["table_notes"] if item["cost"] == "high"}
    assert {"cyq_perf", "stk_factor_pro", "report_rc"} <= high_cost
    assert "stk_auction_o" not in plan["selected_fact_tables"]
    assert "stk_auction_c" not in plan["selected_fact_tables"]


def test_quality_cli_file_backend_writes_json_and_markdown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_panel_quality.py"
    spec = importlib.util.spec_from_file_location("check_panel_quality_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data_path = tmp_path / "panel.parquet"
    json_path = tmp_path / "quality.json"
    md_path = tmp_path / "quality.md"
    pd.DataFrame({"date": [pd.Timestamp("2026-04-21")], "code": ["000001.SZ"], "close": [1.0]}).to_parquet(
        data_path, index=False
    )

    exit_code = module.run_quality_check(
        source_backend="file",
        data_path=str(data_path),
        datasource_config="",
        duckdb_path="",
        source_view="",
        start_date="",
        end_date="",
        fields="close,moneyflow_net_mf_amount",
        required_fields="close",
        json_out=str(json_path),
        markdown_out=str(md_path),
        max_missing_ratio=0.80,
        max_inf_ratio=0.0,
        fail_on_warn=False,
    )

    assert exit_code == 0
    assert json_path.exists()
    assert md_path.exists()
