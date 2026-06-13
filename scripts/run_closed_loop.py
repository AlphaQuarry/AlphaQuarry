from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import shutil
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.config import AlphaMiningConfig, AlphaSimulationConfig
from alpha_mining.datasource import (
    build_snapshot_run_id,
    get_searchable_fields_from_field_catalog,
    load_datasource_settings,
    load_panel_from_duckdb,
    materialize_input_snapshot,
    plan_required_fields_for_closed_loop,
)
from alpha_mining.mining import DeepExploreConfig
from alpha_mining.mining.field_preprocessing import FieldPreprocessConfig
from alpha_mining.simulation.neutralization import (
    neutralization_group_field,
    normalize_neutralization_mode,
)
from alpha_mining.workflow.benchmark_binding import resolve_benchmark_binding
from alpha_mining.workflow.closed_loop import (
    ClosedLoopConfig,
    run_closed_loop,
    validate_universe_registries,
)
from factor_research import SampleSplitConfig, TransactionCostConfig


DEFAULT_DEEP_WINDOWS = (5, 10, 22, 66, 132)


def _default_apply_tradability_constraints() -> bool:
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run closed-loop alpha mining + analysis.")
    parser.add_argument("--source-backend", default="duckdb", choices=["duckdb", "file"])
    parser.add_argument(
        "--data-path",
        default="",
        help="Input panel path (.pkl/.parquet/.csv) for file backend",
    )
    parser.add_argument("--datasource-config", default="", help="Datasource config yaml path")
    parser.add_argument("--duckdb-path", default="")
    parser.add_argument("--source-view", default="")
    parser.add_argument("--duckdb-memory-limit", default="", help="DuckDB memory_limit, e.g. 6GB")
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=0,
        help="DuckDB threads, 0 means use DuckDB default",
    )
    parser.add_argument(
        "--duckdb-temp-directory",
        default="",
        help="DuckDB temp_directory for spill files",
    )
    parser.add_argument(
        "--duckdb-temp-isolate-run",
        action="store_true",
        help="Use a run-scoped subdirectory under DuckDB temp_directory for safer cleanup.",
    )
    parser.add_argument(
        "--no-duckdb-temp-isolate-run",
        action="store_true",
        help="Disable run-scoped DuckDB temp directory isolation.",
    )
    parser.add_argument(
        "--duckdb-temp-run-id",
        default="",
        help="Optional run id suffix used when --duckdb-temp-isolate-run is enabled.",
    )
    parser.add_argument(
        "--duckdb-max-temp-directory-size",
        default="",
        help="DuckDB max_temp_directory_size, e.g. 100GB",
    )
    parser.add_argument(
        "--duckdb-temp-cleanup-warn-gb",
        type=float,
        default=10.0,
        help="Warn when pre/post cleanup removes temp spill files above this size in GB.",
    )
    parser.add_argument(
        "--cleanup-duckdb-temp",
        action="store_true",
        help="Cleanup default DuckDB temp dir (<duckdb_path>.tmp) before/after run.",
    )
    parser.add_argument(
        "--no-cleanup-duckdb-temp",
        action="store_true",
        help="Disable auto cleanup of default DuckDB temp dir (<duckdb_path>.tmp).",
    )
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--snapshot-root", default="")
    parser.add_argument("--snapshot-input", action="store_true")
    parser.add_argument("--no-snapshot-input", action="store_true")
    parser.add_argument("--field-catalog-version", default="")
    parser.add_argument("--manifest-schema-version", default="v2")
    parser.add_argument(
        "--no-field-preprocessing",
        action="store_true",
        help="Disable scalar raw field preprocessing wrapper",
    )
    parser.add_argument("--field-preprocess-window", type=int, default=120)
    parser.add_argument("--field-preprocess-winsorize-std", type=float, default=4.0)
    parser.add_argument("--run-filters-json", default="", help="JSON object for source filters")
    parser.add_argument(
        "--source-chunk-loading",
        action="store_true",
        help="Enable chunk-level dynamic DuckDB loading",
    )
    parser.add_argument(
        "--no-source-chunk-loading",
        action="store_true",
        help="Disable chunk-level dynamic DuckDB loading",
    )
    parser.add_argument(
        "--source-chunk-mem-warn-mb",
        type=float,
        default=2560.0,
        help="Soft warning threshold for one DuckDB-loaded chunk_raw_df in MB",
    )
    parser.add_argument("--source-chunk-mem-hard-limit-mb", type=float, default=0.0)
    parser.add_argument("--universe", default="cn_all")
    parser.add_argument("--base-dir", default="data/alpha_universe_store")
    parser.add_argument(
        "--search-mode",
        default="layered_v2",
        choices=["template_only", "deep_hybrid", "operator_only", "layered_v2"],
    )
    parser.add_argument(
        "--layer-max-order",
        type=int,
        default=4,
        help="Maximum layered_v2 order to generate, from 0 to 4",
    )
    parser.add_argument(
        "--enable-stateful-phase2-ops",
        action="store_true",
        help="Enable stateful Phase2 operators (hump/trade_when_hold) in candidate generation.",
    )
    parser.add_argument(
        "--layer-max-candidates",
        type=int,
        default=0,
        help="Maximum layered_v2 candidates before prefilter; default is max(400, max_eval*4)",
    )
    parser.add_argument(
        "--layer-budget-json",
        default="",
        help='Optional JSON object for layered_v2 budgets, e.g. {"L0":32,"L1":160,"L2":160,"L3":80,"L4":60}',
    )
    parser.add_argument(
        "--layer-windows",
        default="5,10,22,66,132",
        help="Comma-separated time-series windows for layered_v2",
    )
    parser.add_argument(
        "--layer-gate-families",
        default="liquidity_activity,moneyflow_pressure,price_trend,industry_activity",
    )
    parser.add_argument("--layer-gate-max-total", type=int, default=24)
    parser.add_argument("--layer-gate-max-per-family", type=int, default=6)
    parser.add_argument("--layer-gate-seed-max", type=int, default=18)
    parser.add_argument("--layer-enable-event-gates", action="store_true")
    parser.add_argument("--layer-enable-bucket-groups", action="store_true", default=True)
    parser.add_argument("--no-layer-enable-bucket-groups", action="store_true")
    parser.add_argument("--layer-bucket-max-groups", type=int, default=12)
    parser.add_argument("--layer-bucket-max-composite-groups", type=int, default=6)
    parser.add_argument("--layer-bucket-ranges", default="0,1,0.2")
    parser.add_argument("--layer-bucket-l1-max-total", type=int, default=24)
    parser.add_argument("--layer-bucket-l2-max-total", type=int, default=20)
    parser.add_argument("--layer-enable-recipe-lite", action="store_true", default=True)
    parser.add_argument("--no-layer-enable-recipe-lite", action="store_true")
    parser.add_argument("--layer-recipe-max-total", type=int, default=80)
    parser.add_argument("--layer-recipe-max-per-family", type=int, default=16)
    parser.add_argument("--layer-role-pair-max-total", type=int, default=80)
    parser.add_argument("--layer-cross-family-pair-ratio", type=float, default=0.15)
    parser.add_argument("--field-profile-lite-enabled", action="store_true", default=True)
    parser.add_argument("--no-field-profile-lite", action="store_true")
    parser.add_argument("--field-profile-lite-min-coverage", type=float, default=0.20)
    parser.add_argument("--field-profile-lite-min-finite-rate", type=float, default=0.80)
    parser.add_argument("--field-profile-lite-top-fields-per-family", type=int, default=50)
    parser.add_argument("--feedback-policy-lite-enabled", action="store_true", default=True)
    parser.add_argument("--no-feedback-policy-lite", action="store_true")
    parser.add_argument("--bucket-quality-lite-enabled", action="store_true", default=True)
    parser.add_argument("--no-bucket-quality-lite", action="store_true")
    parser.add_argument("--bucket-quality-max-evaluations", type=int, default=80)
    parser.add_argument("--bucket-quality-min-coverage", type=float, default=0.50)
    parser.add_argument("--bucket-quality-min-median-group-size", type=int, default=5)
    parser.add_argument("--bucket-quality-min-group-count", type=int, default=3)
    parser.add_argument("--bucket-quality-max-nan-group-ratio", type=float, default=0.30)
    parser.add_argument(
        "--bucket-quality-reject-low-quality-composite",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-bucket-quality-reject-low-quality-composite", action="store_true")
    parser.add_argument("--layer-operator-tier", default="stable")
    parser.add_argument("--layer-operator-expansion-max-total", type=int, default=100)
    parser.add_argument("--layer-selection-min-ratio-json", default="")
    parser.add_argument("--layer-selection-max-ratio-json", default="")
    parser.add_argument("--structure-selection-min-ratio-json", default="")
    parser.add_argument("--generation-diagnostics-enabled", action="store_true", default=True)
    parser.add_argument("--no-generation-diagnostics", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--request-new", type=int, default=5)
    parser.add_argument("--max-eval", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-restart-retry", type=int, default=2)
    parser.add_argument("--lock-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--code-col", default="code")
    parser.add_argument("--group-fields", default="industry,sector")
    parser.add_argument("--vector-fields", default="")
    parser.add_argument(
        "--include-fields",
        default="",
        help="Comma-separated scalar/vector field whitelist for expression generation; group fields are controlled by --group-fields",
    )
    parser.add_argument("--exclude-fields", default="")
    parser.add_argument(
        "--include-factor-families",
        default="",
        help="Comma-separated factor families to include: price_volume,fundamental,moneyflow,analyst",
    )
    parser.add_argument(
        "--exclude-factor-families",
        default="",
        help="Comma-separated factor families to exclude",
    )
    parser.add_argument(
        "--family-quota",
        action="store_true",
        help="Enable factor-family quota during candidate ranking",
    )
    parser.add_argument(
        "--no-family-quota",
        action="store_true",
        help="Disable factor-family quota during candidate ranking",
    )
    parser.add_argument("--family-max-selected-ratio", type=float, default=0.45)
    parser.add_argument("--family-min-explore-ratio", type=float, default=0.25)
    parser.add_argument("--analysis-level", default="light_then_full_on_survivors")
    parser.add_argument(
        "--neutralization",
        default="NONE",
        help="Mining-stage neutralization mode: NONE, MARKET, SECTOR, INDUSTRY, SUBINDUSTRY",
    )
    parser.add_argument("--analysis-period", type=int, default=1)
    parser.add_argument("--analysis-layers", type=int, default=10)
    parser.add_argument(
        "--effectiveness-ic-min",
        type=float,
        default=0.015,
        help="Factor effectiveness: minimum |IC| threshold",
    )
    parser.add_argument(
        "--effectiveness-ir-min",
        type=float,
        default=0.25,
        help="Factor effectiveness: minimum |IR| threshold",
    )
    parser.add_argument(
        "--effectiveness-sharpe-min",
        type=float,
        default=0.40,
        help="Factor effectiveness: minimum Sharpe ratio",
    )
    parser.add_argument(
        "--effectiveness-coverage-min",
        type=float,
        default=0.60,
        help="Factor effectiveness: minimum data coverage rate",
    )
    parser.add_argument(
        "--effectiveness-turnover-max",
        type=float,
        default=0.80,
        help="Factor effectiveness: maximum turnover rate",
    )
    parser.add_argument(
        "--effectiveness-min-score",
        type=float,
        default=50.0,
        help="Factor effectiveness: minimum composite score",
    )
    parser.add_argument(
        "--long10-count",
        type=int,
        default=10,
        help="Number of top stocks in long-only portfolio",
    )
    parser.add_argument(
        "--include-double-sort",
        action="store_true",
        help="Persist 5x5 factor/control double-sort diagnostics",
    )
    parser.add_argument("--double-sort-control-col", default="total_mv")
    parser.add_argument("--double-sort-factor-bins", type=int, default=5)
    parser.add_argument("--double-sort-control-bins", type=int, default=5)
    parser.add_argument(
        "--double-sort-method",
        default="conditional",
        choices=["conditional", "independent"],
    )
    parser.add_argument(
        "--apply-tradability-constraints",
        dest="apply_tradability_constraints",
        action="store_true",
        default=_default_apply_tradability_constraints(),
        help="Apply can_buy/can_sell constraints to long-only analysis (default: enabled)",
    )
    parser.add_argument(
        "--no-tradability-constraints",
        dest="apply_tradability_constraints",
        action="store_false",
        help="Disable can_buy/can_sell constraints and restore legacy unrestricted long-only analysis",
    )
    parser.add_argument("--tradability-mode", default="entry_exit", choices=["entry_exit"])
    parser.add_argument(
        "--include-sample-split-analysis",
        action="store_true",
        help="Add legacy train/validation/oos split metrics",
    )
    parser.add_argument(
        "--no-phase-metrics",
        action="store_true",
        help="Disable default train/val/test phase metrics",
    )
    parser.add_argument("--phase-metric-min-obs", type=int, default=1)
    parser.add_argument("--feedback-phase", default="train")
    parser.add_argument(
        "--include-visualization-png",
        action="store_true",
        help="Generate legacy static PNG visualization artifacts",
    )
    parser.add_argument(
        "--benchmark-enabled",
        action="store_true",
        help="Compatibility flag; benchmark is attempted by default unless --no-benchmark is set",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Disable benchmark loading and benchmark-relative metrics",
    )
    parser.add_argument(
        "--benchmark-view",
        default="v_project_index_daily",
        help="DuckDB view/table for benchmark index daily data",
    )
    parser.add_argument(
        "--benchmark-code",
        default="",
        help="Benchmark index code; default auto-binds to --universe",
    )
    parser.add_argument("--benchmark-date-col", default="date")
    parser.add_argument("--benchmark-code-col", default="code")
    parser.add_argument("--benchmark-close-col", default="close")
    parser.add_argument(
        "--benchmark-return-col",
        default="return",
        help="Benchmark daily return column; if unavailable, set empty to use close pct_change",
    )
    parser.add_argument(
        "--transaction-cost-enabled",
        dest="transaction_cost_enabled",
        action="store_true",
        default=True,
        help="Generate fee-adjusted PnL for supported portfolios (default: enabled)",
    )
    parser.add_argument(
        "--no-transaction-cost",
        dest="transaction_cost_enabled",
        action="store_false",
        help="Disable fee-adjusted PnL generation for supported portfolios",
    )
    parser.add_argument("--transaction-cost-model-name", default="cn_a_linear_v1")
    parser.add_argument("--commission-bps-per-side", type=float, default=2.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=3.0)
    parser.add_argument("--stamp-tax-bps-sell", type=float, default=5.0)
    parser.add_argument("--transfer-fee-bps-per-side", type=float, default=0.1)
    parser.add_argument("--exchange-fee-bps-per-side", type=float, default=0.341)
    parser.add_argument("--regulatory-fee-bps-per-side", type=float, default=0.2)
    parser.add_argument(
        "--charge-initial-position",
        action="store_true",
        help="Charge initial portfolio establishment in after-fee PnL",
    )
    parser.add_argument(
        "--factor-library",
        action="store_true",
        help="Submit candidates that pass the base score into the simple factor library",
    )
    parser.add_argument("--factor-library-min-score", type=float, default=60.0)
    parser.add_argument("--factor-library-staging-min-score", type=float, default=50.0)
    parser.add_argument("--factor-library-max-signal-corr", type=float, default=0.80)
    parser.add_argument("--factor-library-max-ic-corr", type=float, default=0.80)
    parser.add_argument("--factor-library-max-pnl-corr", type=float, default=0.80)
    parser.add_argument("--factor-library-staging-max-corr", type=float, default=0.95)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--validation-start", default="2025-01-01")
    parser.add_argument("--validation-end", default="2025-12-31")
    parser.add_argument("--oos-start", default="2026-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-candidate-ranking",
        action="store_true",
        help="Disable V1 candidate ranking and fall back to deterministic order",
    )
    parser.add_argument(
        "--score-weights-json",
        default="",
        help='JSON object for candidate score weights, e.g. {"feedback":0.20,"novelty":0.18}',
    )
    parser.add_argument(
        "--complexity-weight",
        type=float,
        default=0.10,
        help="Weight for expression complexity penalty in candidate ranking",
    )
    parser.add_argument(
        "--sample-prefilter",
        action="store_true",
        help="Enable optional lightweight sample prefilter before full materialization",
    )
    parser.add_argument("--sample-min-coverage", type=float, default=0.30)
    parser.add_argument("--sample-max-inf-ratio", type=float, default=0.01)
    parser.add_argument("--sample-max-evaluations", type=int, default=60)
    parser.add_argument("--sample-lookback-days", type=int, default=120)
    parser.add_argument("--sample-prefilter-stratified", action="store_true", default=True)
    parser.add_argument("--no-sample-prefilter-stratified", action="store_true")
    parser.add_argument("--feedback-min-explore-ratio", type=float, default=0.30)
    parser.add_argument(
        "--feedback-exploit-ratio",
        type=float,
        default=0.55,
        help="Feedback sampler exploit/explore split ratio",
    )
    parser.add_argument(
        "--feedback-lookback-batches",
        type=int,
        default=50,
        help="Feedback sampler lookback window in batches",
    )
    parser.add_argument(
        "--enable-feedback-mutation",
        action="store_true",
        help="Enable feedback mutation source (feedback_mutation_v2).",
    )
    parser.add_argument("--mutation-budget-ratio", type=float, default=0.15)
    parser.add_argument("--mutation-max-children-per-parent", type=int, default=3)
    parser.add_argument(
        "--mutation-min-selected-count",
        type=int,
        default=0,
        help="Minimum number of feedback_mutation_v2 candidates to keep in final ranked list when mutation is enabled.",
    )
    parser.add_argument(
        "--mutation-min-selected-ratio",
        type=float,
        default=0.0,
        help="Minimum ratio of feedback_mutation_v2 candidates in final ranked list when mutation is enabled.",
    )
    parser.add_argument("--fragment-max-age-batches", type=int, default=50)
    parser.add_argument("--fragment-cooldown-batches", type=int, default=3)
    parser.add_argument("--no-purge", action="store_true")
    parser.add_argument("--panel-cache-max-size", type=int, default=64)
    parser.add_argument("--candidate-artifact-retention-enabled", action="store_true", default=True)
    parser.add_argument("--no-candidate-artifact-retention", action="store_true")
    parser.add_argument("--candidate-artifact-retention-max-batches", type=int, default=200)
    parser.add_argument("--candidate-artifact-retention-days", type=int, default=30)
    parser.add_argument("--analysis-artifact-retention-enabled", action="store_true", default=False)
    parser.add_argument("--no-analysis-artifact-retention", action="store_true")
    parser.add_argument("--analysis-artifact-retention-max-runs", type=int, default=120)
    parser.add_argument("--analysis-artifact-retention-days", type=int, default=90)
    parser.add_argument("--run-health-retention-enabled", action="store_true", default=True)
    parser.add_argument("--no-run-health-retention", action="store_true")
    parser.add_argument("--run-health-retention-max-lines", type=int, default=5000)
    parser.add_argument("--run-health-retention-days", type=int, default=90)
    parser.add_argument("--registry-health-check", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cleanup_duckdb_temp = bool(args.cleanup_duckdb_temp or (not args.no_cleanup_duckdb_temp))

    datasource_settings = load_datasource_settings(str(args.datasource_config or "") or None)

    group_fields = tuple([x.strip() for x in str(args.group_fields).split(",") if x.strip()])
    vector_fields = tuple([x.strip() for x in str(args.vector_fields).split(",") if x.strip()])
    include_fields = tuple([x.strip() for x in str(args.include_fields).split(",") if x.strip()])
    exclude_fields = tuple([x.strip() for x in str(args.exclude_fields).split(",") if x.strip()])
    include_factor_families = tuple([x.strip() for x in str(args.include_factor_families).split(",") if x.strip()])
    exclude_factor_families = tuple([x.strip() for x in str(args.exclude_factor_families).split(",") if x.strip()])
    enable_family_quota = bool(args.family_quota) or (not bool(args.no_family_quota))
    benchmark_binding = resolve_benchmark_binding(
        universe_name=str(args.universe or ""),
        explicit_code=str(args.benchmark_code or ""),
    )

    source_backend = str(args.source_backend or datasource_settings.source_backend).strip().lower()
    run_filters = _merge_filters(
        datasource_settings.run_filters,
        _parse_json_obj(args.run_filters_json, "--run-filters-json"),
    )
    layer_budgets = _parse_layer_budget_arg(args.layer_budget_json)
    layer_windows = (
        tuple(int(w) for w in str(args.layer_windows).split(",") if str(w).strip())
        if hasattr(args, "layer_windows") and args.layer_windows
        else DEFAULT_DEEP_WINDOWS
    )
    layer_selection_min_ratio = _parse_ratio_json_arg(
        args.layer_selection_min_ratio_json, "--layer-selection-min-ratio-json"
    )
    layer_selection_max_ratio = _parse_ratio_json_arg(
        args.layer_selection_max_ratio_json, "--layer-selection-max-ratio-json"
    )
    structure_selection_min_ratio = _parse_ratio_json_arg(
        args.structure_selection_min_ratio_json,
        "--structure-selection-min-ratio-json",
        normalize_layer_keys=False,
    )
    layer_max_candidates = (
        int(args.layer_max_candidates) if int(args.layer_max_candidates) > 0 else max(400, int(args.max_eval) * 4)
    )

    cfg = ClosedLoopConfig(
        universe_name=str(args.universe),
        universe_base_dir=str(args.base_dir),
        batch_size=int(args.batch_size),
        request_new_alphas=int(args.request_new),
        max_eval_expressions=int(args.max_eval),
        search_mode=str(args.search_mode),
        layer_max_order=int(args.layer_max_order),
        layer_max_candidates=int(layer_max_candidates),
        layer_budgets=layer_budgets or {"L0": 32, "L1": 160, "L2": 160, "L3": 100, "L4": 80},
        layer_windows=layer_windows,
        enable_stateful_phase2_ops=bool(args.enable_stateful_phase2_ops),
        layer_gate_families=tuple([x.strip() for x in str(args.layer_gate_families).split(",") if x.strip()]),
        layer_gate_max_total=_positive_int(args.layer_gate_max_total, 24),
        layer_gate_max_per_family=_positive_int(args.layer_gate_max_per_family, 6),
        layer_gate_seed_max=_positive_int(args.layer_gate_seed_max, 18),
        layer_enable_event_gates=bool(args.layer_enable_event_gates),
        layer_enable_bucket_groups=bool(args.layer_enable_bucket_groups and not args.no_layer_enable_bucket_groups),
        layer_bucket_max_groups=_positive_int(args.layer_bucket_max_groups, 12),
        layer_bucket_max_composite_groups=_positive_int(args.layer_bucket_max_composite_groups, 6),
        layer_bucket_ranges=tuple([x.strip() for x in str(args.layer_bucket_ranges).split(";") if x.strip()])
        or ("0,1,0.2",),
        layer_bucket_l1_max_total=_positive_int(args.layer_bucket_l1_max_total, 24),
        layer_bucket_l2_max_total=_positive_int(args.layer_bucket_l2_max_total, 20),
        layer_enable_recipe_lite=bool(args.layer_enable_recipe_lite and not args.no_layer_enable_recipe_lite),
        layer_recipe_max_total=_positive_int(args.layer_recipe_max_total, 80),
        layer_recipe_max_per_family=_positive_int(args.layer_recipe_max_per_family, 16),
        layer_role_pair_max_total=_positive_int(args.layer_role_pair_max_total, 80),
        layer_cross_family_pair_ratio=_normalized_unit_ratio(args.layer_cross_family_pair_ratio, 0.15),
        field_profile_lite_enabled=bool(args.field_profile_lite_enabled and not args.no_field_profile_lite),
        field_profile_lite_min_coverage=_normalized_unit_ratio(args.field_profile_lite_min_coverage, 0.20),
        field_profile_lite_min_finite_rate=_normalized_unit_ratio(args.field_profile_lite_min_finite_rate, 0.80),
        field_profile_lite_top_fields_per_family=_positive_int(args.field_profile_lite_top_fields_per_family, 50),
        feedback_policy_lite_enabled=bool(args.feedback_policy_lite_enabled and not args.no_feedback_policy_lite),
        bucket_quality_lite_enabled=bool(args.bucket_quality_lite_enabled and not args.no_bucket_quality_lite),
        bucket_quality_max_evaluations=_positive_int(args.bucket_quality_max_evaluations, 80),
        bucket_quality_min_coverage=_normalized_unit_ratio(args.bucket_quality_min_coverage, 0.50),
        bucket_quality_min_median_group_size=_positive_int(args.bucket_quality_min_median_group_size, 5),
        bucket_quality_min_group_count=_positive_int(args.bucket_quality_min_group_count, 3),
        bucket_quality_max_nan_group_ratio=_normalized_unit_ratio(args.bucket_quality_max_nan_group_ratio, 0.30),
        bucket_quality_reject_low_quality_composite=bool(
            args.bucket_quality_reject_low_quality_composite and not args.no_bucket_quality_reject_low_quality_composite
        ),
        layer_operator_tier=str(args.layer_operator_tier or "stable"),
        layer_operator_expansion_max_total=_positive_int(args.layer_operator_expansion_max_total, 100),
        layer_selection_min_ratio=layer_selection_min_ratio or None,
        layer_selection_max_ratio=layer_selection_max_ratio or None,
        structure_selection_min_ratio=structure_selection_min_ratio or None,
        generation_diagnostics_enabled=bool(args.generation_diagnostics_enabled and not args.no_generation_diagnostics),
        max_iterations=int(args.iterations),
        max_restart_retry=int(args.max_restart_retry),
        lock_timeout_seconds=float(args.lock_timeout_seconds),
        date_col=str(args.date_col),
        code_col=str(args.code_col),
        group_fields=group_fields,
        vector_fields=vector_fields,
        include_fields=include_fields,
        exclude_fields=exclude_fields,
        include_factor_families=include_factor_families,
        exclude_factor_families=exclude_factor_families,
        enable_family_quota=enable_family_quota,
        family_max_selected_ratio=_normalized_unit_ratio(args.family_max_selected_ratio, 0.45),
        family_min_explore_ratio=_normalized_unit_ratio(args.family_min_explore_ratio, 0.25),
        analysis_level_mode=str(args.analysis_level),
        analysis_period=int(args.analysis_period),
        analysis_layers=int(args.analysis_layers),
        include_double_sort=bool(args.include_double_sort),
        double_sort_control_col=str(args.double_sort_control_col),
        double_sort_factor_bins=_positive_int(args.double_sort_factor_bins, 5),
        double_sort_control_bins=_positive_int(args.double_sort_control_bins, 5),
        double_sort_method=str(args.double_sort_method),
        apply_tradability_constraints=bool(args.apply_tradability_constraints),
        tradability_mode=str(args.tradability_mode),
        include_sample_split_analysis=bool(args.include_sample_split_analysis),
        include_phase_metrics=not bool(args.no_phase_metrics),
        phase_metric_min_obs=_positive_int(args.phase_metric_min_obs, 1),
        effectiveness_ic_abs_min=max(0.0, float(args.effectiveness_ic_min)),
        effectiveness_ir_abs_min=max(0.0, float(args.effectiveness_ir_min)),
        effectiveness_sharpe_min=max(0.0, float(args.effectiveness_sharpe_min)),
        effectiveness_coverage_min=_normalized_unit_ratio(args.effectiveness_coverage_min, 0.60),
        effectiveness_turnover_max=_normalized_unit_ratio(args.effectiveness_turnover_max, 0.80),
        effectiveness_min_score=max(0.0, float(args.effectiveness_min_score)),
        long10_count=_positive_int(args.long10_count, 10),
        feedback_phase=str(args.feedback_phase or "train"),
        include_visualization_png=bool(args.include_visualization_png),
        benchmark_enabled=not bool(args.no_benchmark),
        benchmark_code=str(benchmark_binding.code or "000300.SH"),
        benchmark_view=str(args.benchmark_view or "v_project_index_daily"),
        benchmark_date_col=str(args.benchmark_date_col or "date"),
        benchmark_code_col=str(args.benchmark_code_col or "code"),
        benchmark_close_col=str(args.benchmark_close_col or "close"),
        benchmark_return_col=str(args.benchmark_return_col or ""),
        sample_split_config=SampleSplitConfig(
            train_start=str(args.train_start),
            train_end=str(args.train_end),
            validation_start=str(args.validation_start),
            validation_end=str(args.validation_end),
            oos_start=str(args.oos_start),
        ),
        transaction_cost_config=TransactionCostConfig(
            enabled=bool(args.transaction_cost_enabled),
            model_name=str(args.transaction_cost_model_name or "cn_a_linear_v1"),
            commission_bps_per_side=float(args.commission_bps_per_side),
            slippage_bps_per_side=float(args.slippage_bps_per_side),
            stamp_tax_bps_sell=float(args.stamp_tax_bps_sell),
            transfer_fee_bps_per_side=float(args.transfer_fee_bps_per_side),
            exchange_fee_bps_per_side=float(args.exchange_fee_bps_per_side),
            regulatory_fee_bps_per_side=float(args.regulatory_fee_bps_per_side),
            charge_initial_position=bool(args.charge_initial_position),
        ),
        factor_library_enabled=bool(args.factor_library),
        factor_library_min_score=float(args.factor_library_min_score),
        factor_library_staging_min_score=float(args.factor_library_staging_min_score),
        factor_library_max_signal_corr=float(args.factor_library_max_signal_corr),
        factor_library_max_ic_corr=float(args.factor_library_max_ic_corr),
        factor_library_max_pnl_corr=float(args.factor_library_max_pnl_corr),
        factor_library_staging_max_corr=float(args.factor_library_staging_max_corr),
        enable_candidate_ranking=not bool(args.no_candidate_ranking),
        score_weights_json=str(args.score_weights_json or ""),
        complexity_weight=max(0.0, float(args.complexity_weight)),
        enable_sample_prefilter=bool(args.sample_prefilter),
        sample_prefilter_min_coverage=float(args.sample_min_coverage),
        sample_prefilter_max_inf_ratio=float(args.sample_max_inf_ratio),
        sample_prefilter_max_evaluations=int(args.sample_max_evaluations),
        sample_prefilter_lookback_days=int(args.sample_lookback_days),
        sample_prefilter_stratified=bool(args.sample_prefilter_stratified and not args.no_sample_prefilter_stratified),
        feedback_min_explore_ratio=float(args.feedback_min_explore_ratio),
        feedback_exploit_ratio=_normalized_unit_ratio(args.feedback_exploit_ratio, 0.55),
        feedback_lookback_batches=_positive_int(args.feedback_lookback_batches, 50),
        enable_feedback_mutation=bool(args.enable_feedback_mutation),
        mutation_budget_ratio=_normalized_mutation_budget_ratio(args.mutation_budget_ratio),
        mutation_max_children_per_parent=_positive_int(args.mutation_max_children_per_parent, 3),
        mutation_min_selected_count=_non_negative_int(args.mutation_min_selected_count, 0),
        mutation_min_selected_ratio=_normalized_unit_ratio(args.mutation_min_selected_ratio, 0.0),
        mutation_fragment_cooldown_batches=_positive_int(args.fragment_cooldown_batches, 3),
        mutation_fragment_max_age_batches=_positive_int(args.fragment_max_age_batches, 50),
        enable_purge_after_analysis=not bool(args.no_purge),
        panel_cache_max_size=max(0, int(args.panel_cache_max_size)),
        candidate_artifact_retention_enabled=bool(
            args.candidate_artifact_retention_enabled and not args.no_candidate_artifact_retention
        ),
        candidate_artifact_retention_max_batches=_positive_int(args.candidate_artifact_retention_max_batches, 200),
        candidate_artifact_retention_days=_positive_int(args.candidate_artifact_retention_days, 30),
        analysis_artifact_retention_enabled=bool(
            args.analysis_artifact_retention_enabled and not args.no_analysis_artifact_retention
        ),
        analysis_artifact_retention_max_runs=_positive_int(args.analysis_artifact_retention_max_runs, 120),
        analysis_artifact_retention_days=_positive_int(args.analysis_artifact_retention_days, 90),
        run_health_retention_enabled=bool(args.run_health_retention_enabled and not args.no_run_health_retention),
        run_health_retention_max_lines=_positive_int(args.run_health_retention_max_lines, 5000),
        run_health_retention_days=_positive_int(args.run_health_retention_days, 90),
        registry_health_check=bool(args.registry_health_check),
        mining_config=AlphaMiningConfig(
            simulation=AlphaSimulationConfig(
                delay=1,
                decay=0,
                neutralization=normalize_neutralization_mode(args.neutralization),
                truncation=None,
                pasteurization=True,
                universe="universe",
            )
        ),
        deep_explore_config=DeepExploreConfig(
            windows=layer_windows,
            max_depth=2,
            max_candidates=max(50, int(args.max_eval)),
            enable_stateful_phase2_ops=bool(args.enable_stateful_phase2_ops),
            random_seed=int(args.seed),
        ),
        source_backend=source_backend,
        field_catalog_version=str(args.field_catalog_version or datasource_settings.field_catalog_version or "v1"),
        moneyflow_source=str(datasource_settings.moneyflow_source or "moneyflow"),
        manifest_schema_version=str(args.manifest_schema_version or "v2"),
        field_preprocessing_config=FieldPreprocessConfig(
            enabled=not bool(args.no_field_preprocessing),
            ts_backfill_window=max(1, int(args.field_preprocess_window)),
            winsorize_std=float(args.field_preprocess_winsorize_std),
        ),
        run_filters=run_filters,
        source_chunk_mem_warn_mb=float(args.source_chunk_mem_warn_mb),
        source_chunk_mem_hard_limit_mb=max(0.0, float(args.source_chunk_mem_hard_limit_mb)),
        duckdb_memory_limit=str(args.duckdb_memory_limit or "").strip(),
        duckdb_threads=_non_negative_int(args.duckdb_threads),
        duckdb_temp_directory=str(args.duckdb_temp_directory or "").strip(),
        duckdb_max_temp_directory_size=str(args.duckdb_max_temp_directory_size or "").strip(),
    )

    cleanup_duckdb_path = ""
    cleanup_temp_dir = ""
    cleanup_enabled = False
    if source_backend == "file":
        data_path_text = str(args.data_path or "").strip()
        if not data_path_text:
            raise ValueError("--data-path is required when --source-backend=file")
        data_path = Path(data_path_text)
        raw_df = _load_dataframe(data_path)
        if bool(cfg.benchmark_enabled):
            cfg = replace(
                cfg,
                benchmark_status={
                    "status": "missing",
                    "reason": "benchmark loading is only wired for duckdb source backend",
                    "row_count": 0,
                },
            )
        cfg = _cfg_with_source(
            cfg,
            input_source_path=str(data_path.as_posix()),
            snapshot_path=str(data_path.as_posix()),
            duckdb_path="",
            source_view="",
            source_backend="file",
            date_range=("", ""),
        )
        print(f"[closed_loop] file backend loaded rows={len(raw_df)} cols={len(raw_df.columns)}")
    else:
        duckdb_path = str(args.duckdb_path or datasource_settings.paths.duckdb_path).strip()
        source_view = str(args.source_view or datasource_settings.source_view).strip() or "v_project_panel_cn_a"
        if not duckdb_path:
            raise ValueError("duckdb backend requires --duckdb-path or PROJECT_DUCKDB_PATH")
        temp_run_label = _build_temp_run_label(args=args)
        duckdb_runtime_settings = _build_duckdb_runtime_settings(
            args=args,
            duckdb_path=duckdb_path,
            run_label=temp_run_label,
        )
        cfg = replace(
            cfg,
            duckdb_memory_limit=str(duckdb_runtime_settings.get("memory_limit", "") or ""),
            duckdb_threads=_non_negative_int(duckdb_runtime_settings.get("threads", 0)),
            duckdb_temp_directory=str(duckdb_runtime_settings.get("temp_directory", "") or ""),
            duckdb_max_temp_directory_size=str(duckdb_runtime_settings.get("max_temp_directory_size", "") or ""),
        )
        cleanup_duckdb_path = str(duckdb_path)
        cleanup_temp_dir = str(duckdb_runtime_settings.get("temp_directory", "") or "")
        cleanup_enabled = bool(cleanup_duckdb_temp)
        cleanup_allow_nested_default = bool(duckdb_runtime_settings.get("cleanup_allow_nested_default", False))
        cleanup_warn_bytes = int(max(0.0, float(args.duckdb_temp_cleanup_warn_gb or 0.0)) * 1024.0 * 1024.0 * 1024.0)
        if cleanup_enabled:
            pre = _safe_cleanup_duckdb_temp_dir(
                duckdb_path=cleanup_duckdb_path,
                temp_directory=cleanup_temp_dir,
                allow_nested_default=cleanup_allow_nested_default,
            )
            _print_cleanup_summary(stage="pre", result=pre, warn_bytes=cleanup_warn_bytes)

        enable_source_chunk_loading = not bool(args.no_source_chunk_loading)
        if bool(args.source_chunk_loading):
            enable_source_chunk_loading = True

        plan = plan_required_fields_for_closed_loop(
            duckdb_path=duckdb_path,
            source_view=source_view,
            closed_loop_config=cfg,
            universe_base_dir=str(args.base_dir),
            universe_name=str(args.universe),
        )
        required_fields = plan.get("required_fields", [])
        preview_exprs = plan.get("selected_expressions", [])
        field_source = str(plan.get("field_source", "view_columns"))
        available_columns = {str(x) for x in plan.get("available_columns", [])}
        searchable_fields = sorted(
            get_searchable_fields_from_field_catalog(
                duckdb_path=duckdb_path,
                catalog_view="v_project_field_catalog",
                include_fields=include_fields,
                include_factor_families=include_factor_families,
                exclude_factor_families=exclude_factor_families,
            )
        )
        if available_columns:
            searchable_fields = [field for field in searchable_fields if field in available_columns]
        if searchable_fields:
            cfg = replace(
                cfg,
                search_field_universe=tuple(searchable_fields),
                search_field_source="field_catalog",
                enable_source_chunk_loading=enable_source_chunk_loading,
            )
        else:
            cfg = replace(
                cfg,
                search_field_universe=tuple(),
                search_field_source=str(field_source),
                enable_source_chunk_loading=enable_source_chunk_loading,
            )

        bootstrap_fields = _collect_bootstrap_required_fields(cfg=cfg)
        if available_columns:
            bootstrap_fields = [x for x in bootstrap_fields if x in available_columns]
        if not bool(enable_source_chunk_loading):
            bootstrap_fields = list(required_fields)

        raw_df = load_panel_from_duckdb(
            duckdb_path=duckdb_path,
            source_view=source_view,
            required_fields=bootstrap_fields,
            start_date=str(args.start_date or "") or None,
            end_date=str(args.end_date or "") or None,
            date_col=str(args.date_col),
            code_col=str(args.code_col),
            base_fields=cfg.base_frame_cols,
            group_fields=cfg.group_fields,
            run_filters=run_filters,
            duckdb_settings=duckdb_runtime_settings,
        )
        print(
            f"[closed_loop] duckdb backend loaded rows={len(raw_df)} cols={len(raw_df.columns)} "
            f"required_fields={len(bootstrap_fields)} preview_exprs={len(preview_exprs)} "
            f"field_source={field_source} search_field_universe={len(cfg.search_field_universe)} "
            f"source_chunk_loading={bool(cfg.enable_source_chunk_loading)} "
            f"effective_source_view={raw_df.attrs.get('duckdb_effective_source_view', source_view)}"
        )
        if bool(cfg.benchmark_enabled):
            benchmark_records, benchmark_status = _load_benchmark_returns_from_duckdb(
                duckdb_path=duckdb_path,
                view=str(args.benchmark_view or "v_project_index_daily"),
                benchmark_code=str(cfg.benchmark_code or "000300.SH"),
                date_col=str(args.benchmark_date_col or "date"),
                code_col=str(args.benchmark_code_col or "code"),
                close_col=str(args.benchmark_close_col or "close"),
                return_col=str(args.benchmark_return_col or ""),
                start_date=str(args.start_date or ""),
                end_date=str(args.end_date or ""),
                duckdb_settings=duckdb_runtime_settings,
            )
            cfg = replace(
                cfg,
                benchmark_returns=tuple(benchmark_records),
                benchmark_status={
                    **dict(benchmark_status or {}),
                    "binding_source": benchmark_binding.source,
                    "binding_universe": benchmark_binding.universe_name,
                    "binding_reason": benchmark_binding.reason,
                },
            )
            print(
                f"[closed_loop] benchmark status={benchmark_status.get('status')} "
                f"code={benchmark_status.get('code', cfg.benchmark_code)} "
                f"rows={benchmark_status.get('row_count', 0)}"
            )

        if bool(args.snapshot_input) and bool(args.no_snapshot_input):
            print(
                "[closed_loop][warn] both --snapshot-input and --no-snapshot-input were supplied; snapshot is disabled."
            )
        do_snapshot = bool(args.snapshot_input) and not bool(args.no_snapshot_input)
        snapshot_path = ""
        if do_snapshot:
            snapshot_root = str(args.snapshot_root).strip() or str(datasource_settings.paths.snapshots_path.as_posix())
            snapshot_run_id = build_snapshot_run_id(
                universe_name=str(args.universe),
                source_view=source_view,
                start_date=str(args.start_date or "") or None,
                end_date=str(args.end_date or "") or None,
                fields=[str(x) for x in raw_df.columns],
            )
            snapshot_meta = materialize_input_snapshot(
                raw_df=raw_df,
                snapshot_root=snapshot_root,
                universe_name=str(args.universe),
                run_id=snapshot_run_id,
                metadata={
                    "source_backend": "duckdb",
                    "duckdb_path": duckdb_path,
                    "source_view": source_view,
                    "start_date": str(args.start_date or ""),
                    "end_date": str(args.end_date or ""),
                    "run_filters": run_filters,
                    "planned_required_fields": bootstrap_fields,
                    "planned_preview_expressions": preview_exprs,
                    "field_source": field_source,
                    "search_field_universe_count": int(len(cfg.search_field_universe)),
                    "source_chunk_loading": bool(cfg.enable_source_chunk_loading),
                },
            )
            snapshot_path = str(snapshot_meta.get("snapshot_path", ""))

        cfg = _cfg_with_source(
            cfg,
            input_source_path=f"duckdb://{Path(duckdb_path).as_posix()}::{source_view}",
            snapshot_path=snapshot_path,
            duckdb_path=str(Path(duckdb_path).as_posix()),
            source_view=source_view,
            source_backend="duckdb",
            date_range=(str(args.start_date or ""), str(args.end_date or "")),
        )

    try:
        if bool(cfg.registry_health_check):
            registry_health = validate_universe_registries(
                base_dir=cfg.universe_base_dir, universe_name=cfg.universe_name
            )
            print("[closed_loop] registry_health_check")
            print(json.dumps(registry_health, ensure_ascii=False, sort_keys=True, indent=2))
        out = run_closed_loop(raw_df=raw_df, config=cfg)
        print("closed_loop completed")
        print(out)
    finally:
        if cleanup_enabled:
            post = _safe_cleanup_duckdb_temp_dir(
                duckdb_path=cleanup_duckdb_path,
                temp_directory=cleanup_temp_dir,
                allow_nested_default=cleanup_allow_nested_default,
            )
            _print_cleanup_summary(stage="post", result=post, warn_bytes=cleanup_warn_bytes)


def _cfg_with_source(
    cfg: ClosedLoopConfig,
    input_source_path: str,
    snapshot_path: str,
    duckdb_path: str,
    source_view: str,
    source_backend: str,
    date_range: tuple[str, str],
) -> ClosedLoopConfig:
    return replace(
        cfg,
        input_source_path=str(input_source_path),
        snapshot_path=str(snapshot_path),
        duckdb_path=str(duckdb_path),
        source_view=str(source_view),
        source_backend=str(source_backend),
        source_date_range=(str(date_range[0] or ""), str(date_range[1] or "")),
    )


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".pkl":
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
    raise ValueError(f"Unsupported data file extension: {path}")


def _load_benchmark_returns_from_duckdb(
    *,
    duckdb_path: str,
    view: str,
    benchmark_code: str,
    date_col: str,
    code_col: str,
    close_col: str,
    return_col: str,
    start_date: str,
    end_date: str,
    duckdb_settings: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status_base = {
        "enabled": True,
        "code": str(benchmark_code or "000300.SH"),
        "view": str(view or "v_project_index_daily"),
        "row_count": 0,
    }
    try:
        import duckdb
    except Exception as exc:
        return [], {
            **status_base,
            "status": "missing",
            "reason": f"duckdb import failed: {exc}",
        }

    try:
        con = duckdb.connect(database=str(duckdb_path), read_only=True)
    except Exception as exc:
        return [], {
            **status_base,
            "status": "missing",
            "reason": f"duckdb connect failed: {exc}",
        }

    try:
        settings = dict(duckdb_settings or {})
        for key in [
            "memory_limit",
            "threads",
            "temp_directory",
            "max_temp_directory_size",
        ]:
            value = settings.get(key)
            if value not in (None, ""):
                try:
                    con.execute(f"SET {key} = ?", [value])
                except Exception:
                    pass
        fields = [_sql_ident(date_col), _sql_ident(close_col)]
        aliases = ["trade_date", "close"]
        if str(return_col or "").strip():
            fields.append(_sql_ident(return_col))
            aliases.append("benchmark_return")
        select_expr = ", ".join(f"{field} AS {alias}" for field, alias in zip(fields, aliases))
        where = [f"{_sql_ident(code_col)} = ?"]
        params: list[Any] = [str(benchmark_code or "000300.SH")]
        if str(start_date or "").strip():
            where.append(f"{_sql_ident(date_col)} >= ?")
            params.append(str(start_date))
        if str(end_date or "").strip():
            where.append(f"{_sql_ident(date_col)} <= ?")
            params.append(str(end_date))
        sql = (
            f"SELECT {select_expr} FROM {_sql_relation(view)} "
            f"WHERE {' AND '.join(where)} ORDER BY {_sql_ident(date_col)}"
        )
        frame = con.execute(sql, params).fetchdf()
    except Exception as exc:
        return [], {
            **status_base,
            "status": "missing",
            "reason": f"query failed: {exc}",
        }
    finally:
        try:
            con.close()
        except Exception:
            pass

    if frame.empty:
        return [], {
            **status_base,
            "status": "missing",
            "reason": "no benchmark rows matched configured code/date range",
        }
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if "benchmark_return" in frame.columns:
        frame["return"] = pd.to_numeric(frame["benchmark_return"], errors="coerce")
    else:
        close = pd.to_numeric(frame["close"], errors="coerce")
        frame["return"] = close.pct_change()
    frame = frame.dropna(subset=["trade_date", "return"]).sort_values("trade_date", kind="mergesort")
    if frame.empty:
        return [], {
            **status_base,
            "status": "missing",
            "reason": "benchmark rows exist but no usable returns were produced",
        }
    rows = [
        {
            "trade_date": row.trade_date.strftime("%Y-%m-%d"),
            "return": float(row.return_),
        }
        for row in frame.rename(columns={"return": "return_"}).itertuples(index=False)
    ]
    return rows, {
        **status_base,
        "status": "ok",
        "row_count": int(len(rows)),
        "start": str(rows[0]["trade_date"]),
        "end": str(rows[-1]["trade_date"]),
        "return_source": str(return_col or "close_pct_change"),
    }


def _sql_ident(name: str) -> str:
    text = str(name or "").strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{text}"'


def _sql_relation(name: str) -> str:
    parts = [part.strip() for part in str(name or "").split(".") if part.strip()]
    if not parts:
        raise ValueError("Benchmark view must not be empty")
    return ".".join(_sql_ident(part).replace('"', '"') for part in parts)


def _parse_json_obj(raw: str, option_name: str = "--run-filters-json") -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{option_name} must be a JSON object")
    return loaded


def _parse_int_budget(raw: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in dict(raw or {}).items():
        layer = str(key).strip().upper()
        if layer and not layer.startswith("L"):
            layer = f"L{layer}"
        if layer:
            out[layer] = max(0, int(value))
    return out


def _parse_layer_budget_arg(raw: str) -> dict[str, int]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        return _parse_int_budget(_parse_json_obj(text, "--layer-budget-json"))
    except json.JSONDecodeError:
        repaired = _quote_unquoted_json_keys(text)
        return _parse_int_budget(_parse_json_obj(repaired, "--layer-budget-json"))


def _parse_layer_min_ratio_arg(raw: str) -> dict[str, float]:
    return _parse_ratio_json_arg(raw, "--layer-selection-min-ratio-json")


def _parse_ratio_json_arg(raw: str, option_name: str, *, normalize_layer_keys: bool = True) -> dict[str, float]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        loaded = _parse_json_obj(text, option_name)
    except json.JSONDecodeError:
        loaded = _parse_json_obj(_quote_unquoted_json_keys(text), option_name)
    out: dict[str, float] = {}
    for key, value in dict(loaded or {}).items():
        name = str(key).strip()
        if normalize_layer_keys:
            name = name.upper()
            if name and not name.startswith("L"):
                name = f"L{name}"
        else:
            name = name.lower()
        if not name:
            continue
        try:
            ratio = float(value)
        except Exception:
            ratio = 0.0
        out[name] = max(0.0, min(1.0, ratio))
    return out


def _quote_unquoted_json_keys(text: str) -> str:
    return re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', str(text or ""))


def _merge_filters(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for k, v in (patch or {}).items():
        out[str(k)] = v
    return out


def _collect_bootstrap_required_fields(cfg: ClosedLoopConfig) -> list[str]:
    needed: list[str] = []

    def _append(name: str) -> None:
        key = str(name or "").strip()
        if not key:
            return
        if key not in needed:
            needed.append(key)

    for col in cfg.base_frame_cols:
        _append(str(col))
    if bool(getattr(cfg, "include_double_sort", False)):
        _append(str(getattr(cfg, "double_sort_control_col", "total_mv")))
        _append("circ_mv")
    if bool(getattr(cfg, "apply_tradability_constraints", False)):
        _append("can_buy")
        _append("can_sell")
    _append(str(cfg.date_col))
    _append(str(cfg.code_col))
    for col in cfg.group_fields:
        _append(str(col))

    simulation_universe = str(getattr(cfg.mining_config.simulation, "universe", "") or "").strip()
    if simulation_universe:
        _append(simulation_universe)
    group_field = neutralization_group_field(getattr(cfg.mining_config.simulation, "neutralization", "NONE"))
    if group_field:
        _append(group_field)
    return needed


def _normalized_mutation_budget_ratio(raw: float) -> float:
    try:
        value = float(raw)
    except Exception:
        return 0.15
    return max(0.0, min(1.0, value))


def _normalized_unit_ratio(raw: float, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return max(0.0, min(1.0, value))


def _positive_int(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except Exception:
        return int(default)
    return max(1, value)


def _non_negative_int(raw: Any, default: int = 0) -> int:
    try:
        value = int(raw)
    except Exception:
        return int(default)
    return max(0, value)


def _build_duckdb_runtime_settings(
    args: argparse.Namespace,
    duckdb_path: str,
    run_label: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    memory_limit = str(args.duckdb_memory_limit or "").strip()
    if memory_limit:
        out["memory_limit"] = memory_limit

    threads = _non_negative_int(args.duckdb_threads, 0)
    if threads > 0:
        out["threads"] = threads

    temp_text = str(args.duckdb_temp_directory or "").strip()
    if temp_text:
        temp_path = Path(temp_text)
        if not temp_path.is_absolute():
            temp_path = Path.cwd() / temp_path
    else:
        temp_path = Path(f"{duckdb_path}.tmp")
    isolate_temp = bool(getattr(args, "duckdb_temp_isolate_run", False)) and (
        not bool(getattr(args, "no_duckdb_temp_isolate_run", False))
    )
    if isolate_temp:
        run_id_text = str(getattr(args, "duckdb_temp_run_id", "") or "").strip() or str(run_label or "").strip()
        run_id_safe = _safe_temp_run_id(run_id_text)
        temp_path = temp_path / f"run_{run_id_safe}"
    out["temp_directory"] = str(temp_path.as_posix())
    out["cleanup_allow_nested_default"] = bool(isolate_temp)

    max_temp_size = str(args.duckdb_max_temp_directory_size or "").strip()
    if max_temp_size:
        out["max_temp_directory_size"] = max_temp_size
    return out


def _safe_cleanup_duckdb_temp_dir(
    duckdb_path: str,
    temp_directory: str,
    allow_nested_default: bool = False,
) -> dict[str, Any]:
    db_path = Path(str(duckdb_path or "")).resolve()
    temp_path = Path(str(temp_directory or "")).resolve()
    expected_default = Path(f"{str(db_path)}.tmp").resolve()
    if allow_nested_default:
        if temp_path != expected_default and expected_default not in temp_path.parents:
            return {
                "deleted_files": 0,
                "deleted_bytes": 0,
                "skipped": "outside_default_temp_tree",
            }
    else:
        if temp_path != expected_default:
            return {
                "deleted_files": 0,
                "deleted_bytes": 0,
                "skipped": "non_default_temp_directory",
            }
    if not temp_path.exists() or not temp_path.is_dir():
        return {"deleted_files": 0, "deleted_bytes": 0, "skipped": "missing"}

    deleted_files = 0
    deleted_bytes = 0
    for file_path in temp_path.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            deleted_bytes += int(file_path.stat().st_size)
        except Exception:
            pass
        try:
            file_path.unlink()
            deleted_files += 1
        except Exception:
            continue
    try:
        shutil.rmtree(temp_path, ignore_errors=True)
    except Exception:
        pass
    return {
        "deleted_files": int(deleted_files),
        "deleted_bytes": int(deleted_bytes),
        "skipped": "",
    }


def _safe_temp_run_id(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "").strip())
    text = text.strip("._-")
    if not text:
        return "default"
    return text[:96]


def _build_temp_run_label(args: argparse.Namespace) -> str:
    override = str(getattr(args, "duckdb_temp_run_id", "") or "").strip()
    if override:
        return override
    universe = str(getattr(args, "universe", "") or "").strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if universe:
        return f"{universe}_{timestamp}"
    return f"run_{timestamp}"


def _print_cleanup_summary(stage: str, result: dict[str, Any], warn_bytes: int) -> None:
    deleted_files = int(result.get("deleted_files", 0) or 0)
    deleted_bytes = int(result.get("deleted_bytes", 0) or 0)
    skipped = str(result.get("skipped", "") or "").strip()
    if deleted_files > 0 or deleted_bytes > 0:
        print(
            f"[closed_loop][cleanup-{stage}] deleted_files={deleted_files} "
            f"deleted_mb={float(deleted_bytes) / (1024.0 * 1024.0):.2f}"
        )
    elif skipped:
        print(f"[closed_loop][cleanup-{stage}] skipped={skipped}")
    if warn_bytes > 0 and deleted_bytes >= warn_bytes:
        print(
            f"[closed_loop][cleanup-{stage}][warning] reclaimed_large_tmp="
            f"{float(deleted_bytes) / (1024.0 * 1024.0 * 1024.0):.2f}GB"
        )


if __name__ == "__main__":
    main()
