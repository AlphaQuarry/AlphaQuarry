from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_compare_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "compare_feedback_mutation_ab.py"
    spec = importlib.util.spec_from_file_location("compare_feedback_mutation_ab_under_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_mutation_budget_ratio_is_normalized_to_unit_interval() -> None:
    mod = _load_compare_module()
    assert mod._normalized_mutation_budget_ratio(0.20) == 0.20
    assert mod._normalized_mutation_budget_ratio(2.0) == 1.0
    assert mod._normalized_mutation_budget_ratio(-1.0) == 0.0
    assert mod._normalized_mutation_budget_ratio(0.08) == 0.08


def test_compare_build_variant_command_toggle_flag() -> None:
    mod = _load_compare_module()
    parser = mod._build_parser()
    args = parser.parse_args([])
    baseline_cmd = mod._build_variant_command(
        args=args,
        datasource_config="configs/datasource.local.yaml",
        duckdb_path="data/duckdb/market.duckdb",
        source_view="v_project_panel_cn_a",
        start_date="2025-01-01",
        end_date="2025-12-31",
        universe_name="u_baseline",
        enable_feedback_mutation=False,
        request_new_override=None,
        iterations=1,
    )
    mutation_cmd = mod._build_variant_command(
        args=args,
        datasource_config="configs/datasource.local.yaml",
        duckdb_path="data/duckdb/market.duckdb",
        source_view="v_project_panel_cn_a",
        start_date="2025-01-01",
        end_date="2025-12-31",
        universe_name="u_mutation",
        enable_feedback_mutation=True,
        request_new_override=None,
        iterations=1,
    )
    assert "--enable-feedback-mutation" not in baseline_cmd
    assert "--enable-feedback-mutation" in mutation_cmd
    assert "--source-chunk-loading" in baseline_cmd
    assert "--no-snapshot-input" in baseline_cmd


def test_compare_build_variant_command_with_warmup_overrides() -> None:
    mod = _load_compare_module()
    parser = mod._build_parser()
    args = parser.parse_args(
        [
            "--request-new",
            "10",
            "--mutation-min-selected-count",
            "2",
            "--mutation-min-selected-ratio",
            "0.25",
        ]
    )
    warmup_cmd = mod._build_variant_command(
        args=args,
        datasource_config="configs/datasource.local.yaml",
        duckdb_path="data/duckdb/market.duckdb",
        source_view="v_project_panel_cn_a",
        start_date="2025-01-01",
        end_date="2025-12-31",
        universe_name="u_warmup",
        enable_feedback_mutation=False,
        request_new_override=4,
        iterations=2,
    )
    req_idx = warmup_cmd.index("--request-new")
    it_idx = warmup_cmd.index("--iterations")
    min_count_idx = warmup_cmd.index("--mutation-min-selected-count")
    min_ratio_idx = warmup_cmd.index("--mutation-min-selected-ratio")
    assert warmup_cmd[req_idx + 1] == "4"
    assert warmup_cmd[it_idx + 1] == "2"
    assert warmup_cmd[min_count_idx + 1] == "2"
    assert warmup_cmd[min_ratio_idx + 1] == "0.25"


def test_compare_build_markdown_report_contains_window() -> None:
    mod = _load_compare_module()
    summary_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "status": "ok",
                "candidate_count": 10,
                "candidate_passed": 8,
                "candidate_rejected": 2,
                "sample_reject_count": 1,
                "mutation_candidate_count": 0,
                "mutation_ratio": 0.0,
                "selected_count": 5,
                "scoreboard_rows": 5,
                "topn_score_mean": 0.1,
                "topn_score_median": 0.1,
                "topn_positive_ratio": 0.8,
                "topn_turnover_mean": 0.2,
                "selected_source_dist_json": "{}",
            },
            {
                "variant": "mutation",
                "status": "ok",
                "candidate_count": 12,
                "candidate_passed": 9,
                "candidate_rejected": 3,
                "sample_reject_count": 1,
                "mutation_candidate_count": 2,
                "mutation_ratio": 0.1667,
                "selected_count": 5,
                "scoreboard_rows": 5,
                "topn_score_mean": 0.2,
                "topn_score_median": 0.2,
                "topn_positive_ratio": 0.9,
                "topn_turnover_mean": 0.3,
                "selected_source_dist_json": '{"feedback_mutation_v2": 1}',
            },
        ]
    )
    text = mod._build_markdown_report(
        summary_df=summary_df,
        top_n=20,
        start_date="2025-01-01",
        end_date="2025-12-31",
        source_view="v_project_panel_cn_a",
    )
    assert "date_range" in text
    assert "v_project_panel_cn_a" in text
    assert "Delta (mutation - baseline)" in text
    assert "Fragment Feedback" in text
