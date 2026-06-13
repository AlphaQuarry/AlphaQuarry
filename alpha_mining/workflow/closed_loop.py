from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from factor_research import SampleSplitConfig, TransactionCostConfig
from factor_research.screening import FactorEffectivenessConfig

from ..atomic_io import atomic_write_dataframe_csv, atomic_write_json, atomic_write_text
from ..config import AlphaMiningConfig
from ..mining.field_preprocessing import FieldPreprocessConfig
from ..mining.operator_signatures import build_default_operator_signature_registry
from ..registry import build_default_registry
from ..simulation.neutralization import (
    neutralization_group_field,
    normalize_neutralization_mode,
)
from ..mining import (
    apply_candidate_feedback_to_registry,
    CandidateRankerConfig,
    compute_adaptive_explore_ratio,
    DeepExploreConfig,
    AlphaMiningPipeline,
    FeedbackSampler,
    FeedbackSamplerConfig,
    FragmentRegistryConfig,
    fragment_registry_path,
    load_fragment_registry,
    plan_candidates,
    prune_candidate_artifacts,
    refresh_fragment_registry,
    save_fragment_registry,
    save_candidate_artifacts,
    select_active_fragments,
)
from ..panel_store import PanelStore
from .analysis_cycle import (
    AnalysisLevelConfig,
    BatchAnalysisConfig,
    run_factor_analysis_batch,
)
from .closed_loop_config_summary import closed_loop_config_hash
from .factor_library import REGISTRY_COLUMNS as FACTOR_LIBRARY_REGISTRY_COLUMNS
from .factor_library import (
    FactorLibraryConfig,
    compute_scoreboard_health,
    submit_factor_library_candidates,
)
from .lifecycle import (
    load_lifecycle_registry,
    mark_analyzed,
    mark_failed,
    mark_materialized,
    register_alpha_batch,
)
from .purge import purge_alpha_values
from .universe_store import (
    DEFAULT_UNIVERSE_BASE_DIR,
    append_universe_expressions,
    build_dashboard_factor_metrics,
    canonical_simulation_config_json,
    get_universe_paths,
    init_universe_workspace,
    load_factor_metrics_registry,
    load_seen_expression_hashes_for_universe,
    load_seen_signal_hashes_for_universe,
    load_universe_alpha_batch,
    load_universe_base_frame,
    load_universe_expression_registry,
    save_universe_alpha_values,
    save_universe_analysis_run,
    save_universe_base_frame,
    save_universe_input_manifest,
    signal_hash_for_expression,
)
from .visualization_artifacts import (
    attach_visualization_manifest_to_analysis_meta,
    save_factor_visualization_artifacts,
)


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Configuration for the closed-loop alpha mining workflow.

    This frozen dataclass contains all parameters controlling the closed-loop
    mining process, including search space configuration, analysis settings,
    feedback policies, and resource limits.

    Key parameter groups:
    - Search space: search_mode, layer_budgets, layer_windows, etc.
    - Analysis: analysis_period, analysis_layers, effectiveness thresholds
    - Feedback: feedback_enabled, feedback_exploit_ratio, mutation settings
    - Data source: source_backend, duckdb_path, source_view
    - Resource limits: max_eval_expressions, chunk loading settings

    Attributes:
        universe_name: Target universe name (e.g., 'cn_all', 'csi500').
        batch_size: Number of factors per evaluation batch.
        request_new_alphas: Number of new alpha candidates per iteration.
        max_eval_expressions: Maximum expressions to evaluate per run.
        search_mode: Search strategy ('layered_v2', 'template_only', etc.).
        layer_budgets: Budget allocation per layer (L0-L4).
        layer_windows: Rolling windows to use for time-series operators.
        analysis_period: Forward return period for IC calculation.
        analysis_layers: Number of quantile layers for layer analysis.
        effectiveness_min_score: Minimum effectiveness score for factor selection.
        factor_library_enabled: Whether to enable factor library admission.

    Example:
        >>> config = ClosedLoopConfig(
        ...     universe_name='csi500',
        ...     batch_size=10,
        ...     max_eval_expressions=200,
        ... )
        >>> run_closed_loop(config=config, ...)
    """

    universe_name: str = "cn_all"
    universe_base_dir: str = DEFAULT_UNIVERSE_BASE_DIR

    batch_size: int = 5
    request_new_alphas: int = 5
    max_new_alphas_per_chunk: int = 5
    compute_chunk_size: int = 5
    max_eval_expressions: int = 80
    search_mode: str = "layered_v2"  # template_only / deep_hybrid / operator_only / layered_v2
    use_signature_generator: bool = True
    layer_max_order: int = 4
    layer_max_candidates: int = 400
    layer_budgets: dict[str, int] = field(default_factory=lambda: {"L0": 32, "L1": 160, "L2": 160, "L3": 100, "L4": 80})
    layer_windows: tuple[int, ...] = (5, 10, 22, 66, 132)
    layer_include_gates: bool = True
    enable_stateful_phase2_ops: bool = False
    layer_gate_families: tuple[str, ...] = (
        "liquidity_activity",
        "moneyflow_pressure",
        "price_trend",
        "industry_activity",
    )
    layer_gate_max_total: int = 24
    layer_gate_max_per_family: int = 6
    layer_gate_seed_max: int = 18
    layer_gate_templates: tuple[str, ...] = ("if_else_zero",)
    layer_enable_event_gates: bool = False
    layer_enable_bucket_groups: bool = True
    layer_bucket_max_groups: int = 12
    layer_bucket_max_composite_groups: int = 6
    layer_bucket_ranges: tuple[str, ...] = ("0,1,0.2",)
    layer_bucket_field_families: tuple[str, ...] = (
        "size",
        "liquidity",
        "valuation",
        "chip",
        "technical",
    )
    layer_bucket_use_composite_industry: bool = True
    layer_bucket_l1_max_total: int = 24
    layer_bucket_l2_max_total: int = 20
    layer_enable_recipe_lite: bool = True
    layer_recipe_max_total: int = 80
    layer_recipe_max_per_family: int = 16
    layer_role_pair_max_total: int = 80
    layer_cross_family_pair_ratio: float = 0.15
    field_profile_lite_enabled: bool = True
    field_profile_lite_min_coverage: float = 0.20
    field_profile_lite_min_finite_rate: float = 0.80
    field_profile_lite_top_fields_per_family: int = 50
    feedback_policy_lite_enabled: bool = True
    bucket_quality_lite_enabled: bool = True
    bucket_quality_max_evaluations: int = 80
    bucket_quality_min_coverage: float = 0.50
    bucket_quality_min_median_group_size: int = 5
    bucket_quality_min_group_count: int = 3
    bucket_quality_max_nan_group_ratio: float = 0.30
    bucket_quality_reject_low_quality_composite: bool = True
    bucket_quality_reject_low_quality_plain: bool = False
    layer_operator_tier: str = "stable"
    layer_operator_expansion_max_total: int = 100
    layer_selection_min_ratio: dict[str, float] | None = field(
        default_factory=lambda: {"L1": 0.20, "L2": 0.20, "L3": 0.15, "L4": 0.15}
    )
    layer_selection_max_ratio: dict[str, float] | None = field(default_factory=lambda: {"L0": 0.08})
    structure_selection_min_ratio: dict[str, float] | None = field(
        default_factory=lambda: {
            "bucket": 0.05,
            "gate": 0.05,
            "recipe": 0.05,
            "role_pair": 0.05,
        }
    )
    generation_diagnostics_enabled: bool = True
    enable_recall_validation: bool = False

    output_alpha_dtype: str = "float32"
    drop_all_nan_alpha_rows: bool = True
    enable_purge_after_analysis: bool = True
    panel_cache_max_size: int = 64
    candidate_artifact_retention_enabled: bool = True
    candidate_artifact_retention_max_batches: int = 200
    candidate_artifact_retention_days: int = 30
    analysis_artifact_retention_enabled: bool = False
    analysis_artifact_retention_max_runs: int = 120
    analysis_artifact_retention_days: int = 90
    run_health_retention_enabled: bool = True
    run_health_retention_max_lines: int = 5000
    run_health_retention_days: int = 90
    registry_health_check: bool = False

    date_col: str = "date"
    code_col: str = "code"
    group_fields: tuple[str, ...] = ("industry", "sector")
    vector_fields: tuple[str, ...] = ()
    include_fields: tuple[str, ...] = ()
    exclude_fields: tuple[str, ...] = ()
    include_factor_families: tuple[str, ...] = ()
    exclude_factor_families: tuple[str, ...] = ()
    enable_family_quota: bool = True
    family_max_selected_ratio: float = 0.45
    family_min_explore_ratio: float = 0.25
    base_frame_cols: tuple[str, ...] = ("date", "code", "pct_chg", "circ_mv")

    analysis_level_mode: str = "light_then_full_on_survivors"
    analysis_period: int = 1
    analysis_layers: int = 10
    analysis_is_timeseries: bool = True
    analysis_return_col: str = "pct_chg"
    analysis_market_value_column: str = "circ_mv"
    analysis_do_neutralize: bool = False
    analysis_do_standardize: bool = False
    analysis_max_lag: int = 10
    include_full_ic_lag_analysis: bool = False
    analysis_include_robustness: bool = True
    analysis_robust_periods: tuple[int, ...] = (1, 5, 10, 20)
    include_double_sort: bool = False
    double_sort_control_col: str = "total_mv"
    double_sort_factor_bins: int = 5
    double_sort_control_bins: int = 5
    double_sort_method: str = "conditional"
    apply_tradability_constraints: bool = True
    tradability_mode: str = "entry_exit"
    include_sample_split_analysis: bool = False
    sample_split_config: SampleSplitConfig = field(default_factory=SampleSplitConfig)
    include_phase_metrics: bool = True
    phase_metric_min_obs: int = 1
    effectiveness_ic_abs_min: float = 0.015
    effectiveness_ir_abs_min: float = 0.25
    effectiveness_sharpe_min: float = 0.40
    effectiveness_coverage_min: float = 0.60
    effectiveness_turnover_max: float = 0.80
    effectiveness_min_score: float = 50.0
    long10_count: int = 10
    feedback_phase: str = "train"
    include_visualization_png: bool = False
    benchmark_enabled: bool = True
    benchmark_code: str = ""
    benchmark_view: str = "v_project_index_daily"
    benchmark_date_col: str = "date"
    benchmark_code_col: str = "code"
    benchmark_close_col: str = "close"
    benchmark_return_col: str = "return"
    benchmark_returns: tuple[dict[str, Any], ...] = ()
    benchmark_status: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "pending",
            "reason": "benchmark not loaded yet",
        }
    )
    transaction_cost_config: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    factor_library_enabled: bool = False
    factor_library_min_score: float = 60.0
    factor_library_staging_min_score: float = 50.0
    factor_library_max_signal_corr: float = 0.80
    factor_library_max_ic_corr: float = 0.80
    factor_library_max_pnl_corr: float = 0.80
    factor_library_staging_max_corr: float = 0.95

    idle_sleep_seconds: float = 5.0
    error_backoff_seconds: float = 10.0
    max_iterations: int = 1
    max_restart_retry: int = 2
    lock_timeout_seconds: float = 3600.0

    template_include_families: tuple[str, ...] = (
        "single_ts",
        "single_cross",
        "single_group",
        "price_volume",
        "composite",
        "fundamental",
    )
    template_pool_override: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    deep_explore_config: DeepExploreConfig = field(default_factory=DeepExploreConfig)
    field_preprocessing_config: FieldPreprocessConfig = field(default_factory=FieldPreprocessConfig)
    mining_config: AlphaMiningConfig = field(default_factory=AlphaMiningConfig)
    input_source_path: str = ""
    source_backend: str = "file"
    duckdb_path: str = ""
    source_view: str = ""
    snapshot_path: str = ""
    source_date_range: tuple[str, str] = ("", "")
    field_catalog_version: str = "v1"
    moneyflow_source: str = "moneyflow"
    manifest_schema_version: str = "v2"
    run_filters: dict[str, Any] = field(default_factory=dict)
    search_field_universe: tuple[str, ...] = ()
    search_field_source: str = "panel_store"
    enable_source_chunk_loading: bool = True
    source_chunk_mem_warn_mb: float = 2560.0
    source_chunk_mem_hard_limit_mb: float = 0.0
    duckdb_memory_limit: str = ""
    duckdb_threads: int = 0
    duckdb_temp_directory: str = ""
    duckdb_max_temp_directory_size: str = ""
    enable_candidate_ranking: bool = True
    score_weights_json: str = ""
    complexity_weight: float = 0.10
    enable_sample_prefilter: bool = False
    sample_prefilter_min_coverage: float = 0.30
    sample_prefilter_max_inf_ratio: float = 0.01
    sample_prefilter_max_evaluations: int = 60
    sample_prefilter_lookback_days: int = 120
    sample_prefilter_stratified: bool = True
    feedback_enabled: bool = True
    feedback_min_explore_ratio: float = 0.30
    feedback_exploit_ratio: float = 0.55
    feedback_lookback_batches: int = 50
    enable_feedback_mutation: bool = False
    field_rotation_focus_count: int = 0  # 0 = 不轮转, >0 = 每轮聚焦 N 个字段 (建议 3-5)
    budget_rotation_mode: str = "none"  # none / round_robin
    mutation_budget_ratio: float = 0.15
    mutation_max_children_per_parent: int = 3
    mutation_fragment_cooldown_batches: int = 3
    mutation_fragment_max_age_batches: int = 50
    mutation_stateful_ratio_cap: float = 0.10
    mutation_min_selected_count: int = 0
    mutation_min_selected_ratio: float = 0.0


def validate_closed_loop_config(config: ClosedLoopConfig) -> None:
    """验证 ClosedLoopConfig 参数的依赖关系和有效性。"""
    valid_modes = {"template_only", "deep_hybrid", "operator_only", "layered_v2"}
    if config.search_mode not in valid_modes:
        raise ValueError(f"search_mode must be one of {valid_modes}, got '{config.search_mode}'")

    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")

    if config.analysis_period < 1:
        raise ValueError(f"analysis_period must be >= 1, got {config.analysis_period}")

    if config.max_eval_expressions < 1:
        raise ValueError(f"max_eval_expressions must be >= 1, got {config.max_eval_expressions}")

    if config.layer_budgets:
        for key in config.layer_budgets:
            if not key.startswith("L"):
                raise ValueError(f"layer_budgets key must start with 'L', got '{key}'")
        max_layer = max(int(k.replace("L", "")) for k in config.layer_budgets)
        if max_layer > config.layer_max_order:
            raise ValueError(f"layer_budgets has L{max_layer} but layer_max_order={config.layer_max_order}")

    if config.mutation_budget_ratio < 0 or config.mutation_budget_ratio > 1:
        raise ValueError(f"mutation_budget_ratio must be in [0, 1], got {config.mutation_budget_ratio}")

    if config.mutation_stateful_ratio_cap < 0 or config.mutation_stateful_ratio_cap > 1:
        raise ValueError(f"mutation_stateful_ratio_cap must be in [0, 1], got {config.mutation_stateful_ratio_cap}")

    if config.family_max_selected_ratio < 0 or config.family_max_selected_ratio > 1:
        raise ValueError(f"family_max_selected_ratio must be in [0, 1], got {config.family_max_selected_ratio}")

    if config.family_min_explore_ratio < 0 or config.family_min_explore_ratio > 1:
        raise ValueError(f"family_min_explore_ratio must be in [0, 1], got {config.family_min_explore_ratio}")

    if config.feedback_min_explore_ratio < 0 or config.feedback_min_explore_ratio > 1:
        raise ValueError(f"feedback_min_explore_ratio must be in [0, 1], got {config.feedback_min_explore_ratio}")


def run_closed_loop(
    raw_df: pd.DataFrame,
    config: ClosedLoopConfig,
) -> dict[str, Any]:
    """Run the closed-loop alpha mining workflow.

    Executes the complete closed-loop mining process:
    1. Generate candidate expressions using search space exploration
    2. Evaluate candidates against market data
    3. Analyze factor effectiveness (IC, layers, portfolios)
    4. Feed results back to guide next iteration

    The process runs for the configured number of iterations or until
    no new candidates are found.

    Args:
        raw_df: Long-format DataFrame with market data. Must contain
            date, code, and price/volume columns.
        config: ClosedLoopConfig with all mining parameters.

    Returns:
        Dictionary with keys:
        - 'workspace': Universe workspace context
        - 'iterations': List of iteration results
        - 'lock_owner': Lock ownership info

    Raises:
        ValueError: If raw_df is empty or config is invalid.

    Example:
        >>> from alpha_mining import ClosedLoopConfig, run_closed_loop
        >>> config = ClosedLoopConfig(
        ...     universe_name='csi500',
        ...     batch_size=10,
        ...     max_eval_expressions=200,
        ... )
        >>> result = run_closed_loop(raw_df=df, config=config)
        >>> print(len(result['iterations']))
    """
    validate_closed_loop_config(config)
    if raw_df is None or raw_df.empty:
        raise ValueError("raw_df must not be empty")
    prepared_raw_df = _normalize_closed_loop_input(raw_df=raw_df, config=config)
    ctx = init_universe_workspace(base_dir=config.universe_base_dir, universe_name=config.universe_name)
    _save_or_update_input_manifest(prepared_raw_df, config)

    lock_path = (
        get_universe_paths(base_dir=config.universe_base_dir, universe_name=config.universe_name)["root"]
        / ".closed_loop.lock"
    )
    lock_owner = _acquire_loop_lock(
        lock_path=lock_path,
        timeout_seconds=float(config.lock_timeout_seconds),
        universe_name=str(config.universe_name),
        config_hash=closed_loop_config_hash(config),
    )
    results: list[dict[str, Any]] = []
    heartbeat_stop, heartbeat_thread = _start_loop_lock_heartbeat(
        lock_path=lock_path,
        owner_id=str(lock_owner.get("owner_id", "")),
        timeout_seconds=float(config.lock_timeout_seconds),
    )
    try:
        resume_result = resume_incomplete_batches(raw_df=prepared_raw_df, config=config)
        if resume_result:
            results.extend(resume_result)

        max_iterations = int(config.max_iterations)
        run_forever = max_iterations <= 0
        loop_count = 0
        while run_forever or loop_count < max(1, max_iterations):
            loop_count += 1
            out = run_one_loop_iteration(raw_df=prepared_raw_df, config=config, iteration=int(loop_count))
            out["iteration"] = int(loop_count)
            results.append(out)

            if out.get("status") in {"idle", "no_new_expression"}:
                time.sleep(max(0.0, float(config.idle_sleep_seconds)))
                if not run_forever:
                    break
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2.0)
        _release_loop_lock(lock_path=lock_path, owner_id=str(lock_owner.get("owner_id", "")))
    return {"workspace": ctx, "iterations": results, "lock_owner": lock_owner}


def run_one_loop_iteration(
    raw_df: pd.DataFrame,
    config: ClosedLoopConfig,
    iteration: int = 0,
) -> dict[str, Any]:
    processing_alpha_names: list[str] = []
    iteration_started = time.perf_counter()
    retention_summary: dict[str, Any] = {}
    try:
        prepared_raw_df = _normalize_closed_loop_input(raw_df=raw_df, config=config)
        init_universe_workspace(base_dir=config.universe_base_dir, universe_name=config.universe_name)
        _ensure_base_frame(raw_df=prepared_raw_df, config=config)

        panel_store = _build_panel_store(prepared_raw_df, config)
        pipeline = None
        if not _use_source_chunk_loading(config):
            pipeline = AlphaMiningPipeline.from_panel_store(panel_store, config=config.mining_config)

        candidates, candidate_meta = _generate_candidates(panel_store=panel_store, config=config, iteration=iteration)
        print(
            f"[closed_loop] candidate_field_source={candidate_meta.get('field_source', 'panel_store')} "
            f"scalar_fields={candidate_meta.get('scalar_field_count', 0)} "
            f"group_fields={candidate_meta.get('group_field_count', 0)}"
        )
        _log_candidate_distributions(candidate_meta)
        source_chunk_metas: list[dict[str, Any]] = []
        selected_meta = pd.DataFrame()
        if not candidates:
            result = {
                "status": "idle",
                "reason": "no_candidates",
                "artifact_retention_summary": _prune_closed_loop_artifacts(config),
                "registry_health_summary": validate_universe_registries(
                    base_dir=config.universe_base_dir,
                    universe_name=config.universe_name,
                ),
                "failure_status_counts": _failure_status_counts(config),
                "source_chunk_hard_limit_triggered": False,
            }
            result["run_health_path"] = _append_run_health(
                config=config,
                result=result,
                candidate_meta=candidate_meta,
                selected_meta=selected_meta,
                source_chunk_metas=source_chunk_metas,
                retention_summary=dict(result.get("artifact_retention_summary", {})),
                elapsed_seconds=time.perf_counter() - iteration_started,
            )
            return result

        simulation_cfg_json = canonical_simulation_config_json(
            _canonical_simulation_config_dict(config.mining_config.simulation)
        )
        seen_signal_hashes = load_seen_signal_hashes_for_universe(
            base_dir=config.universe_base_dir,
            universe_name=config.universe_name,
            simulation_config_json=simulation_cfg_json,
        )
        fresh_candidates = _filter_new_signals(candidates, seen_signal_hashes, simulation_cfg_json)
        if not fresh_candidates:
            result = {
                "status": "no_new_expression",
                "reason": "all_candidates_seen",
                "artifact_retention_summary": _prune_closed_loop_artifacts(config),
                "registry_health_summary": validate_universe_registries(
                    base_dir=config.universe_base_dir,
                    universe_name=config.universe_name,
                ),
                "failure_status_counts": _failure_status_counts(config),
                "source_chunk_hard_limit_triggered": False,
            }
            result["run_health_path"] = _append_run_health(
                config=config,
                result=result,
                candidate_meta=candidate_meta,
                selected_meta=selected_meta,
                source_chunk_metas=source_chunk_metas,
                retention_summary=dict(result.get("artifact_retention_summary", {})),
                elapsed_seconds=time.perf_counter() - iteration_started,
            )
            return result

        request_n = max(1, int(config.request_new_alphas))
        selected_expr = fresh_candidates[:request_n]
        candidate_df = candidate_meta.get("candidate_df", pd.DataFrame())
        selected_meta = pd.DataFrame({"expression": selected_expr})
        if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty and "expression" in candidate_df.columns:
            meta_cols = [c for c in candidate_df.columns if c != "expression"]
            selected_meta = pd.merge(
                selected_meta,
                candidate_df[["expression"] + meta_cols],
                on="expression",
                how="left",
            )
        _log_selected_candidate_distributions(selected_meta)
        manifest = _save_or_update_input_manifest(prepared_raw_df, config)
        panel_sig = _panel_signature_hash(prepared_raw_df)
        expr_payload = selected_meta.copy()
        if "source" in expr_payload.columns:
            expr_payload["source"] = expr_payload["source"].fillna("closed_loop").astype(str)
        else:
            expr_payload["source"] = "closed_loop"
        expr_payload["simulation_config_json"] = simulation_cfg_json
        expr_payload["input_manifest_id"] = str(manifest.get("manifest_id", ""))
        expr_payload["input_source_path"] = str(config.input_source_path or "")
        expr_payload["panel_signature_hash"] = panel_sig
        expr_payload["search_mode"] = str(config.search_mode)
        added = append_universe_expressions(
            expression_df=expr_payload,
            base_dir=config.universe_base_dir,
            universe_name=config.universe_name,
        )
        if added.empty:
            result = {
                "status": "no_new_expression",
                "reason": "dedup_after_registry",
                "artifact_retention_summary": _prune_closed_loop_artifacts(config),
                "registry_health_summary": validate_universe_registries(
                    base_dir=config.universe_base_dir,
                    universe_name=config.universe_name,
                ),
                "failure_status_counts": _failure_status_counts(config),
                "source_chunk_hard_limit_triggered": False,
            }
            result["run_health_path"] = _append_run_health(
                config=config,
                result=result,
                candidate_meta=candidate_meta,
                selected_meta=selected_meta,
                source_chunk_metas=source_chunk_metas,
                retention_summary=dict(result.get("artifact_retention_summary", {})),
                elapsed_seconds=time.perf_counter() - iteration_started,
            )
            return result

        alpha_names = added["alpha_name"].astype(str).tolist()
        expressions = added["expression"].astype(str).tolist()
        register_alpha_batch(
            expression_df=added[
                [
                    "alpha_name",
                    "expression",
                    "expression_hash",
                    "source",
                    "simulation_config_json",
                    "input_manifest_id",
                ]
            ],
            base_dir=config.universe_base_dir,
            universe_name=config.universe_name,
            simulation_config_json=simulation_cfg_json,
            input_manifest_id=str(manifest.get("manifest_id", "")),
        )

        chunk_size = max(
            1,
            min(
                int(config.max_new_alphas_per_chunk),
                int(config.compute_chunk_size),
                int(config.batch_size),
            ),
        )
        chunk_results: list[dict[str, Any]] = []
        all_alpha_names: list[str] = []
        for expr_chunk, name_chunk in _chunk_parallel_lists(expressions, alpha_names, chunk_size):
            if not expr_chunk or not name_chunk:
                continue
            processing_alpha_names = list(name_chunk)

            if _use_source_chunk_loading(config):
                source_result = _materialize_alpha_batch_from_source(
                    expressions=expr_chunk,
                    alpha_names=name_chunk,
                    config=config,
                )
                materialized_paths = dict(source_result.get("paths", {}))
                source_chunk_meta = dict(source_result.get("chunk_meta", {}))
                source_chunk_metas.append(source_chunk_meta)
            else:
                source_chunk_meta = {}
                if pipeline is None:
                    raise ValueError("pipeline is unavailable for non-chunk loading mode")
                materialized_paths = _materialize_alpha_batch(
                    pipeline=pipeline,
                    expressions=expr_chunk,
                    alpha_names=name_chunk,
                    config=config,
                )
            success_names = [name for name in name_chunk if materialized_paths.get(name)]
            failed_names = [name for name in name_chunk if name not in set(success_names)]
            if failed_names:
                failure = _classify_closed_loop_failure(
                    RuntimeError("materialize_failed: no alpha value artifact produced"),
                    stage="materialize",
                )
                mark_failed(
                    alpha_names=failed_names,
                    error_message="materialize_failed: no alpha value artifact produced",
                    status=str(failure["status"]),
                    failure_kind=str(failure["failure_kind"]),
                    last_error_stage=str(failure["stage"]),
                    base_dir=config.universe_base_dir,
                    universe_name=config.universe_name,
                )
            if not success_names:
                processing_alpha_names = []
                continue
            all_alpha_names.extend(success_names)
            for name in success_names:
                mark_materialized(
                    alpha_names=[name],
                    alpha_value_path=str(materialized_paths.get(name, "")),
                    base_dir=config.universe_base_dir,
                    universe_name=config.universe_name,
                )

            analysis_meta = _analyze_alpha_batch(
                alpha_names=success_names,
                raw_df=prepared_raw_df,
                config=config,
            )
            analysis_run_id = str(analysis_meta.get("analysis_run_id", "")) if isinstance(analysis_meta, dict) else ""
            mark_analyzed(
                alpha_names=success_names,
                analysis_run_id=analysis_run_id,
                base_dir=config.universe_base_dir,
                universe_name=config.universe_name,
            )

            purge_result = None
            if config.enable_purge_after_analysis:
                purge_result = purge_alpha_values(
                    alpha_names=success_names,
                    base_dir=config.universe_base_dir,
                    universe_name=config.universe_name,
                    update_lifecycle=True,
                )
            chunk_results.append(
                {
                    "alpha_names": list(success_names),
                    "failed_alpha_names": list(failed_names),
                    "analysis_meta": analysis_meta,
                    "purge_result": purge_result,
                    "source_chunk_meta": source_chunk_meta,
                }
            )
            processing_alpha_names = []

        scoreboard = compile_universe_feedback_scoreboard(
            base_dir=config.universe_base_dir,
            universe_name=config.universe_name,
        )
        retention_summary = _prune_closed_loop_artifacts(config)
        mem_summary = _summarize_source_chunk_memory(source_chunk_metas)
        result = {
            "status": "ok",
            "alpha_names": all_alpha_names,
            "chunk_results": chunk_results,
            "scoreboard_rows": int(len(scoreboard)) if isinstance(scoreboard, pd.DataFrame) else 0,
            "artifact_retention_summary": retention_summary,
            "registry_health_summary": validate_universe_registries(
                base_dir=config.universe_base_dir, universe_name=config.universe_name
            ),
            "failure_status_counts": _failure_status_counts(config),
            "source_chunk_hard_limit_triggered": any(
                bool(x.get("hard_limit_triggered", False)) for x in source_chunk_metas
            ),
            **mem_summary,
        }
        result["run_health_path"] = _append_run_health(
            config=config,
            result=result,
            candidate_meta=candidate_meta,
            selected_meta=selected_meta,
            source_chunk_metas=source_chunk_metas,
            retention_summary=retention_summary,
            elapsed_seconds=time.perf_counter() - iteration_started,
        )
        return result
    except Exception as exc:
        if processing_alpha_names:
            failure = _classify_closed_loop_failure(exc, stage="iteration")
            mark_failed(
                alpha_names=processing_alpha_names,
                error_message=str(exc),
                status=str(failure["status"]),
                failure_kind=str(failure["failure_kind"]),
                last_error_stage=str(failure["stage"]),
                base_dir=config.universe_base_dir,
                universe_name=config.universe_name,
            )
        time.sleep(max(0.0, float(config.error_backoff_seconds)))
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
            "alpha_names": processing_alpha_names,
            "universe_name": str(config.universe_name),
            "analysis_period": int(config.analysis_period),
            "analysis_layers": int(config.analysis_layers),
            "artifact_retention_summary": retention_summary,
            "registry_health_summary": validate_universe_registries(
                base_dir=config.universe_base_dir, universe_name=config.universe_name
            ),
            "failure_status_counts": _failure_status_counts(config),
            "source_chunk_hard_limit_triggered": isinstance(exc, SourceChunkMemoryLimitError),
        }
        result["run_health_path"] = _append_run_health(
            config=config,
            result=result,
            candidate_meta={},
            selected_meta=pd.DataFrame(),
            source_chunk_metas=[],
            retention_summary=retention_summary,
            elapsed_seconds=time.perf_counter() - iteration_started,
        )
        return result


def resume_incomplete_batches(
    raw_df: pd.DataFrame,
    config: ClosedLoopConfig,
) -> list[dict[str, Any]]:
    prepared_raw_df = _normalize_closed_loop_input(raw_df=raw_df, config=config)
    lifecycle = load_lifecycle_registry(
        base_dir=config.universe_base_dir,
        universe_name=config.universe_name,
    )
    if lifecycle.empty or "status" not in lifecycle.columns:
        return []

    outputs: list[dict[str, Any]] = []
    chunk_size = max(
        1,
        min(
            int(config.max_new_alphas_per_chunk),
            int(config.compute_chunk_size),
            int(config.batch_size),
        ),
    )

    materialized = lifecycle[lifecycle["status"].astype(str) == "MATERIALIZED"]["alpha_name"].astype(str).tolist()
    for chunk in _chunk_list(materialized, chunk_size):
        if not chunk:
            continue
        meta = _analyze_alpha_batch(alpha_names=chunk, raw_df=prepared_raw_df, config=config)
        mark_analyzed(
            alpha_names=chunk,
            analysis_run_id=str(meta.get("analysis_run_id", "")) if isinstance(meta, dict) else "",
            base_dir=config.universe_base_dir,
            universe_name=config.universe_name,
        )
        purge_result = None
        if config.enable_purge_after_analysis:
            purge_result = purge_alpha_values(
                alpha_names=chunk,
                base_dir=config.universe_base_dir,
                universe_name=config.universe_name,
                update_lifecycle=True,
            )
        outputs.append(
            {
                "status": "resumed_materialized",
                "alpha_names": chunk,
                "analysis_meta": meta,
                "purge_result": purge_result,
            }
        )

    eligible_registered = lifecycle[lifecycle["status"].astype(str) == "REGISTERED"]["alpha_name"].astype(str).tolist()
    if "retry_count" in lifecycle.columns:
        retry_count = pd.to_numeric(lifecycle["retry_count"], errors="coerce").fillna(0)
        failed_mask = (lifecycle["status"].astype(str) == "FAILED") & (retry_count < int(config.max_restart_retry))
        eligible_registered.extend(lifecycle[failed_mask]["alpha_name"].astype(str).tolist())

    eligible_registered = list(dict.fromkeys([x for x in eligible_registered if x]))
    if eligible_registered:
        expr_registry = load_universe_expression_registry(
            base_dir=config.universe_base_dir,
            universe_name=config.universe_name,
        )
        expr_map = {
            str(r["alpha_name"]): str(r.get("expression", ""))
            for _, r in expr_registry.iterrows()
            if str(r.get("alpha_name", ""))
        }
        panel_store = _build_panel_store(prepared_raw_df, config)
        pipeline = None
        if not _use_source_chunk_loading(config):
            pipeline = AlphaMiningPipeline.from_panel_store(panel_store, config=config.mining_config)

        for chunk in _chunk_list(eligible_registered, chunk_size):
            chunk_names = [x for x in chunk if x in expr_map and expr_map.get(x)]
            missing_expr = [x for x in chunk if x not in expr_map or not expr_map.get(x)]
            if missing_expr:
                failure = _classify_closed_loop_failure(ValueError("expression missing from registry"), stage="resume")
                mark_failed(
                    alpha_names=missing_expr,
                    error_message="resume_failed: expression missing from registry",
                    status=str(failure["status"]),
                    failure_kind=str(failure["failure_kind"]),
                    last_error_stage=str(failure["stage"]),
                    base_dir=config.universe_base_dir,
                    universe_name=config.universe_name,
                )
            if not chunk_names:
                continue

            chunk_exprs = [expr_map[x] for x in chunk_names]
            try:
                if _use_source_chunk_loading(config):
                    source_result = _materialize_alpha_batch_from_source(
                        expressions=chunk_exprs,
                        alpha_names=chunk_names,
                        config=config,
                    )
                    materialized_paths = dict(source_result.get("paths", {}))
                    source_chunk_meta = dict(source_result.get("chunk_meta", {}))
                else:
                    source_chunk_meta = {}
                    if pipeline is None:
                        raise ValueError("pipeline is unavailable for non-chunk loading mode")
                    materialized_paths = _materialize_alpha_batch(
                        pipeline=pipeline,
                        expressions=chunk_exprs,
                        alpha_names=chunk_names,
                        config=config,
                    )
                for name in chunk_names:
                    mark_materialized(
                        alpha_names=[name],
                        alpha_value_path=str(materialized_paths.get(name, "")),
                        base_dir=config.universe_base_dir,
                        universe_name=config.universe_name,
                    )

                meta = _analyze_alpha_batch(alpha_names=chunk_names, raw_df=prepared_raw_df, config=config)
                mark_analyzed(
                    alpha_names=chunk_names,
                    analysis_run_id=str(meta.get("analysis_run_id", "")) if isinstance(meta, dict) else "",
                    base_dir=config.universe_base_dir,
                    universe_name=config.universe_name,
                )
                purge_result = None
                if config.enable_purge_after_analysis:
                    purge_result = purge_alpha_values(
                        alpha_names=chunk_names,
                        base_dir=config.universe_base_dir,
                        universe_name=config.universe_name,
                        update_lifecycle=True,
                    )
                outputs.append(
                    {
                        "status": "resumed_registered",
                        "alpha_names": chunk_names,
                        "analysis_meta": meta,
                        "purge_result": purge_result,
                        "source_chunk_meta": source_chunk_meta,
                        **_summarize_source_chunk_memory([source_chunk_meta]),
                    }
                )
            except Exception as exc:
                failure = _classify_closed_loop_failure(exc, stage="resume")
                mark_failed(
                    alpha_names=chunk_names,
                    error_message=f"resume_failed: {type(exc).__name__}: {exc}",
                    status=str(failure["status"]),
                    failure_kind=str(failure["failure_kind"]),
                    last_error_stage=str(failure["stage"]),
                    base_dir=config.universe_base_dir,
                    universe_name=config.universe_name,
                )
                outputs.append(
                    {
                        "status": "resume_error",
                        "error": str(exc),
                        "alpha_names": chunk_names,
                    }
                )
                continue

    return outputs


def compile_universe_feedback_scoreboard(
    base_dir: str | Path,
    universe_name: str,
    enable_factor_health: bool = False,
) -> pd.DataFrame:
    metrics = load_factor_metrics_registry(base_dir=base_dir, universe_name=universe_name)
    expressions = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
    if metrics.empty or expressions.empty:
        return pd.DataFrame()
    if "factor" not in metrics.columns:
        return pd.DataFrame()
    work_m = metrics.copy()
    work_m["factor"] = work_m["factor"].astype(str)
    work_e = expressions.copy()
    work_e["alpha_name"] = work_e["alpha_name"].astype(str)
    expr_cols = [
        c
        for c in [
            "alpha_name",
            "expression",
            "source",
            "fields",
            "operators",
            "family",
            "factor_family",
            "factor_family_mix_json",
            "groups",
            "windows",
            "layer",
            "layer_family",
            "parent_expression",
            "parent_hash",
            "mutation_type",
            "fragment_hash",
            "feedback_source",
            "builder_source",
            "template_id",
            "pair_key",
            "structural_hash",
            "original_expression",
            "simplified_expression",
            "canonical_expression",
            "canonical_hash",
            "neutralization",
            "simulation_hash",
            "signal_hash",
            "lint_passed",
            "lint_reject_reason",
        ]
        if c in work_e.columns
    ]
    merged = pd.merge(
        work_m,
        work_e[expr_cols],
        left_on="factor",
        right_on="alpha_name",
        how="left",
    )
    if "alpha_name_y" in merged.columns:
        merged = merged.drop(columns=["alpha_name_y"])
    merged["scoreboard_score"] = _scoreboard_score(merged)
    if bool(enable_factor_health):
        merged = compute_scoreboard_health(merged)
    merged = merged.sort_values("scoreboard_score", ascending=False).reset_index(drop=True)
    out_path = get_universe_paths(base_dir=base_dir, universe_name=universe_name)["feedback_scoreboard_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe_csv(out_path, merged, index=False, backup=True)
    return merged


def _generate_candidates(
    panel_store: PanelStore,
    config: ClosedLoopConfig,
    iteration: int = 0,
) -> tuple[list[str], dict[str, Any]]:
    simulation_cfg_json = canonical_simulation_config_json(
        _canonical_simulation_config_dict(config.mining_config.simulation)
    )
    seen_hashes = load_seen_expression_hashes_for_universe(
        base_dir=config.universe_base_dir,
        universe_name=config.universe_name,
        simulation_config_json=simulation_cfg_json,
    )
    batch_id = f"batch_{int(time.time() * 1000)}"
    paths = get_universe_paths(base_dir=config.universe_base_dir, universe_name=config.universe_name)
    feedback_hints = _load_feedback_hints(config=config, paths=paths)

    # 自适应探索率：根据最近迭代结果动态调整
    if bool(getattr(config, "adaptive_exploration", False)):
        recent_results = _load_recent_iteration_results(config, window=10)
        adaptive_ratio = compute_adaptive_explore_ratio(
            recent_results,
            CandidateRankerConfig(
                adaptive_exploration=True,
                exploration_window=int(getattr(config, "exploration_window", 10)),
                exploration_base_ratio=float(getattr(config, "feedback_min_explore_ratio", 0.30)),
                exploration_max_ratio=float(getattr(config, "exploration_max_ratio", 0.60)),
                exploration_boost_threshold=int(getattr(config, "exploration_boost_threshold", 3)),
            ),
        )
        feedback_hints["adaptive_explore_ratio"] = adaptive_ratio
    expressions, candidate_df, rejected_df, meta = plan_candidates(
        panel_store=panel_store,
        config=config,
        existing_hashes=seen_hashes,
        batch_id=batch_id,
        feedback_hints=feedback_hints,
        iteration=iteration,
        sample_panel_store_loader=(
            (lambda candidate_df: _build_sample_prefilter_panel_store(candidate_df=candidate_df, config=config))
            if bool(config.enable_sample_prefilter) and _use_source_chunk_loading(config)
            else None
        ),
    )
    artifact_paths = save_candidate_artifacts(
        candidate_df=candidate_df,
        rejected_df=rejected_df,
        sample_df=meta.get("sample_df", pd.DataFrame()),
        root=paths["root"],
        batch_id=batch_id,
        generation_diagnostics=meta.get("layered_generation_diagnostics", {})
        if bool(getattr(config, "generation_diagnostics_enabled", True))
        else None,
    )
    feedback_update_summary: dict[str, int] = {
        "positive_updates": 0,
        "negative_updates": 0,
        "rejected_updates": 0,
    }
    if bool(config.enable_feedback_mutation):
        reg_path = str(feedback_hints.get("fragment_registry_path", "") or "")
        current_batch = int(feedback_hints.get("fragment_current_batch", 0) or 0)
        if reg_path and current_batch > 0:
            registry_df = load_fragment_registry(reg_path)
            evaluated_set = {str(x).strip() for x in expressions if str(x).strip()}
            updated_registry, feedback_update_summary = apply_candidate_feedback_to_registry(
                registry_df=registry_df,
                candidate_df=candidate_df,
                current_batch=current_batch,
                cooldown_batches=max(1, int(config.mutation_fragment_cooldown_batches)),
                evaluated_expressions=evaluated_set,
            )
            save_fragment_registry(updated_registry, reg_path)
            feedback_hints["fragment_feedback_updates"] = dict(feedback_update_summary)

    feedback_hints_path = paths["feedback_dir"] / "feedback_hints.json"
    feedback_hints_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(feedback_hints_path, feedback_hints, backup=True)
    artifact_paths["feedback_hints_path"] = str(feedback_hints_path.as_posix())
    meta.update(artifact_paths)
    meta["fragment_feedback_update_positive"] = int(feedback_update_summary.get("positive_updates", 0))
    meta["fragment_feedback_update_negative"] = int(feedback_update_summary.get("negative_updates", 0))
    meta["fragment_feedback_update_rejected"] = int(feedback_update_summary.get("rejected_updates", 0))
    return expressions, meta


def _build_sample_prefilter_panel_store(candidate_df: pd.DataFrame, config: ClosedLoopConfig) -> PanelStore | None:
    if not _use_source_chunk_loading(config):
        return None
    if candidate_df is None or candidate_df.empty or "expression" not in candidate_df.columns:
        return None

    working = candidate_df.copy()
    if "prefilter_status" in working.columns:
        working = working[working["prefilter_status"].astype(str) == "pass"]
    expressions = [
        str(expr).strip() for expr in working["expression"].dropna().astype(str).tolist() if str(expr).strip()
    ]
    expressions = list(dict.fromkeys(expressions))
    if not expressions:
        return None

    from ..datasource.loader import (
        collect_required_fields_from_expressions,
        load_panel_from_duckdb,
    )

    extra_fields = _simulation_required_fields(config)
    required_fields = collect_required_fields_from_expressions(
        expressions=expressions,
        base_fields=config.base_frame_cols,
        group_fields=config.group_fields,
        extra_fields=extra_fields,
    )
    start_tm = time.perf_counter()
    start_date, end_date = _sample_prefilter_date_range(config)
    sample_raw_df = load_panel_from_duckdb(
        duckdb_path=str(config.duckdb_path),
        source_view=str(config.source_view),
        required_fields=required_fields,
        start_date=start_date,
        end_date=end_date,
        date_col=str(config.date_col),
        code_col=str(config.code_col),
        base_fields=config.base_frame_cols,
        group_fields=config.group_fields,
        run_filters=config.run_filters,
        duckdb_settings=_duckdb_settings_from_config(config),
    )
    if sample_raw_df.empty:
        return None
    elapsed = time.perf_counter() - start_tm
    mem_mb = float(sample_raw_df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)
    print(
        f"[closed_loop][sample_prefilter] source=duckdb expressions={len(expressions)} "
        f"fields={len(sample_raw_df.columns)} rows={len(sample_raw_df)} "
        f"start_date={start_date or ''} end_date={end_date or ''} "
        f"elapsed_seconds={elapsed:.2f} mem_mb={mem_mb:.2f} "
        f"effective_source_view={sample_raw_df.attrs.get('duckdb_effective_source_view', config.source_view)}"
    )
    return _build_panel_store(sample_raw_df, config)


def _sample_prefilter_date_range(
    config: ClosedLoopConfig,
) -> tuple[str | None, str | None]:
    start_date = str(config.source_date_range[0] if len(config.source_date_range) >= 1 else "").strip() or None
    end_date = str(config.source_date_range[1] if len(config.source_date_range) >= 2 else "").strip() or None
    lookback_days = int(getattr(config, "sample_prefilter_lookback_days", 0) or 0)
    if lookback_days <= 0 or not end_date:
        return start_date, end_date
    try:
        end_ts = pd.to_datetime(end_date)
        lookback_start = end_ts - pd.Timedelta(days=lookback_days)
        if start_date:
            source_start = pd.to_datetime(start_date)
            if lookback_start < source_start:
                lookback_start = source_start
        return lookback_start.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")
    except Exception:
        return start_date, end_date


def _load_feedback_hints(config: ClosedLoopConfig, paths: dict[str, Path]) -> dict[str, Any]:
    scoreboard_path = paths["feedback_scoreboard_csv"]
    scoreboard_df = pd.DataFrame()
    if scoreboard_path.exists():
        try:
            scoreboard_df = pd.read_csv(scoreboard_path)
        except Exception:
            scoreboard_df = pd.DataFrame()
    min_explore_ratio = float(config.feedback_min_explore_ratio)
    if bool(config.enable_feedback_mutation):
        min_explore_ratio = max(0.30, min(1.0, min_explore_ratio))
    sampler = FeedbackSampler(
        FeedbackSamplerConfig(
            enabled=bool(config.feedback_enabled),
            exploit_ratio=float(getattr(config, "feedback_exploit_ratio", 0.55)),
            min_explore_ratio=float(min_explore_ratio),
            lookback_batches=int(getattr(config, "feedback_lookback_batches", 50)),
        )
    )
    hints = sampler.build_weight_hints(scoreboard_df=scoreboard_df)
    hints["scoreboard_path"] = str(scoreboard_path.as_posix())
    hints["min_explore_ratio"] = float(min_explore_ratio)

    frag_cfg = FragmentRegistryConfig(
        cooldown_batches=max(1, int(config.mutation_fragment_cooldown_batches)),
        max_age_batches=max(1, int(config.mutation_fragment_max_age_batches)),
        top_k=max(1, int(config.max_eval_expressions) * 4),
    )
    reg_path = fragment_registry_path(paths["feedback_dir"])
    registry_df, current_batch, saved_path = refresh_fragment_registry(
        scoreboard_df=scoreboard_df,
        registry_path=reg_path,
        config=frag_cfg,
        current_batch=None,
    )
    if registry_df is None or registry_df.empty:
        registry_df = load_fragment_registry(saved_path)
    active = select_active_fragments(
        registry_df=registry_df,
        current_batch=int(current_batch),
        max_age_batches=max(1, int(config.mutation_fragment_max_age_batches)),
        limit=max(1, int(config.max_eval_expressions) * 4),
    )
    hints["fragment_registry_path"] = str(saved_path)
    hints["fragment_current_batch"] = int(current_batch)
    hints["fragment_count"] = int(len(registry_df))
    hints["active_fragment_count"] = int(len(active))
    if bool(config.enable_feedback_mutation):
        hints["active_fragments"] = active.to_dict(orient="records")
    return hints


def _materialize_alpha_batch(
    pipeline: AlphaMiningPipeline,
    expressions: list[str],
    alpha_names: list[str],
    config: ClosedLoopConfig,
) -> dict[str, str]:
    if len(expressions) != len(alpha_names):
        raise ValueError("expressions and alpha_names length mismatch")
    saved_paths: dict[str, str] = {}
    for expression, alpha_name in zip(expressions, alpha_names):
        try:
            alpha_wide = pipeline.run_prepared_expressions(
                expressions=[expression],
                output_dtype=config.output_alpha_dtype,
                drop_all_nan_rows=config.drop_all_nan_alpha_rows,
            )
            renamed = alpha_wide.copy()
            expr_cols = [c for c in renamed.columns if c.startswith("alpha_")]
            if len(expr_cols) != 1:
                raise ValueError("alpha output column count mismatch")
            renamed = renamed.rename(columns={expr_cols[0]: alpha_name})
            saved = save_universe_alpha_values(
                alpha_df=renamed[["date", "code", alpha_name]].copy(),
                alpha_name=alpha_name,
                base_dir=config.universe_base_dir,
                universe_name=config.universe_name,
                date_col="date",
                code_col="code",
            )
            saved_paths[alpha_name] = str(saved.get("path", ""))
        except Exception as exc:
            failure = _classify_closed_loop_failure(exc, stage="materialize")
            mark_failed(
                alpha_names=[alpha_name],
                error_message=f"materialize_failed: {type(exc).__name__}: {exc}",
                status=str(failure["status"]),
                failure_kind=str(failure["failure_kind"]),
                last_error_stage=str(failure["stage"]),
                base_dir=config.universe_base_dir,
                universe_name=config.universe_name,
            )
    return saved_paths


def _materialize_alpha_batch_from_source(
    expressions: list[str],
    alpha_names: list[str],
    config: ClosedLoopConfig,
) -> dict[str, Any]:
    if len(expressions) != len(alpha_names):
        raise ValueError("expressions and alpha_names length mismatch")
    if not _use_source_chunk_loading(config):
        raise ValueError("source chunk loading is disabled or source settings are incomplete")

    from ..datasource.loader import (
        collect_required_fields_from_expressions,
        load_panel_from_duckdb,
    )

    start_tm = time.perf_counter()
    extra_fields = _simulation_required_fields(config)
    required_fields = collect_required_fields_from_expressions(
        expressions=expressions,
        base_fields=config.base_frame_cols,
        group_fields=config.group_fields,
        extra_fields=extra_fields,
    )

    start_date = str(config.source_date_range[0] if len(config.source_date_range) >= 1 else "").strip() or None
    end_date = str(config.source_date_range[1] if len(config.source_date_range) >= 2 else "").strip() or None

    chunk_raw_df = load_panel_from_duckdb(
        duckdb_path=str(config.duckdb_path),
        source_view=str(config.source_view),
        required_fields=required_fields,
        start_date=start_date,
        end_date=end_date,
        date_col=str(config.date_col),
        code_col=str(config.code_col),
        base_fields=config.base_frame_cols,
        group_fields=config.group_fields,
        run_filters=config.run_filters,
        duckdb_settings=_duckdb_settings_from_config(config),
    )
    if chunk_raw_df.empty:
        raise ValueError("chunk source load returned empty frame")

    mem_mb = float(chunk_raw_df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)
    warn_threshold_mb = float(config.source_chunk_mem_warn_mb)
    mem_warning = bool(warn_threshold_mb > 0 and mem_mb > warn_threshold_mb)
    hard_limit_mb = float(getattr(config, "source_chunk_mem_hard_limit_mb", 0.0) or 0.0)
    hard_limit_triggered = bool(hard_limit_mb > 0 and mem_mb > hard_limit_mb)
    chunk_meta = {
        "alpha_names": list(alpha_names),
        "required_fields": list(required_fields),
        "field_count": int(len(chunk_raw_df.columns)),
        "row_count": int(len(chunk_raw_df)),
        "elapsed_seconds": 0.0,
        "mem_mb": float(mem_mb),
        "mem_warn_threshold_mb": float(warn_threshold_mb),
        "mem_warning": bool(mem_warning),
        "mem_hard_limit_mb": float(hard_limit_mb),
        "hard_limit_triggered": bool(hard_limit_triggered),
        "requested_source_view": str(config.source_view),
        "effective_source_view": str(chunk_raw_df.attrs.get("duckdb_effective_source_view", config.source_view)),
    }
    print(
        f"[closed_loop][chunk] source=duckdb fields={len(chunk_raw_df.columns)} "
        f"rows={len(chunk_raw_df)} required_fields={len(required_fields)} "
        f"elapsed_seconds={time.perf_counter() - start_tm:.2f} mem_mb={mem_mb:.2f} "
        f"effective_source_view={chunk_meta['effective_source_view']}"
    )
    if mem_warning:
        print(
            f"[closed_loop][warn] chunk memory {mem_mb:.2f} MB exceeds "
            f"source_chunk_mem_warn_mb={warn_threshold_mb:.2f} MB "
            f"alpha_names={','.join(alpha_names)}"
        )
    if hard_limit_triggered:
        raise SourceChunkMemoryLimitError(
            f"source chunk memory {mem_mb:.2f} MB exceeds hard limit {hard_limit_mb:.2f} MB"
        )
    panel_store = _build_panel_store(chunk_raw_df, config)
    pipeline = AlphaMiningPipeline.from_panel_store(panel_store, config=config.mining_config)
    out = _materialize_alpha_batch(
        pipeline=pipeline,
        expressions=expressions,
        alpha_names=alpha_names,
        config=config,
    )
    chunk_meta["elapsed_seconds"] = float(time.perf_counter() - start_tm)
    return {"paths": out, "chunk_meta": chunk_meta}


def _analyze_alpha_batch(
    alpha_names: list[str],
    raw_df: pd.DataFrame,
    config: ClosedLoopConfig,
) -> dict[str, Any]:
    _ensure_base_frame(raw_df=raw_df, config=config)
    alpha_df = load_universe_alpha_batch(
        alpha_names=alpha_names,
        base_dir=config.universe_base_dir,
        universe_name=config.universe_name,
    )
    base_df = load_universe_base_frame(
        base_dir=config.universe_base_dir,
        universe_name=config.universe_name,
    )
    from ..adapters import to_factor_research_frame

    fr_input = to_factor_research_frame(
        raw_df=base_df,
        alpha_wide_df=alpha_df,
        code_col="code",
        date_col="date",
    )
    factor_cols = [x for x in alpha_names if x in fr_input.columns]
    batch_out = run_factor_analysis_batch(
        df_raw=fr_input,
        factor_cols=factor_cols,
        config=BatchAnalysisConfig(
            period=config.analysis_period,
            layers=config.analysis_layers,
            is_timeseries=config.analysis_is_timeseries,
            return_col=config.analysis_return_col,
            market_value_column=config.analysis_market_value_column,
            do_neutralize=config.analysis_do_neutralize,
            do_standardize=config.analysis_do_standardize,
            max_lag=config.analysis_max_lag,
            include_full_ic_lag_analysis=bool(config.include_visualization_png or config.include_full_ic_lag_analysis),
            include_robustness=config.analysis_include_robustness,
            robust_periods=tuple(int(x) for x in config.analysis_robust_periods),
            analysis_level=AnalysisLevelConfig(mode=config.analysis_level_mode),
            apply_filtering=True,
            signal_delay=int(getattr(config.mining_config.simulation, "delay", 0)),
            include_double_sort=bool(config.include_double_sort),
            double_sort_control_col=str(config.double_sort_control_col),
            double_sort_factor_bins=int(config.double_sort_factor_bins),
            double_sort_control_bins=int(config.double_sort_control_bins),
            double_sort_method=str(config.double_sort_method),
            apply_tradability_constraints=bool(config.apply_tradability_constraints),
            tradability_mode=str(config.tradability_mode),
            long10_count=int(getattr(config, "long10_count", 10)),
            include_sample_split_analysis=bool(config.include_sample_split_analysis),
            sample_split_config=config.sample_split_config,
            include_phase_metrics=bool(config.include_phase_metrics),
            phase_metric_min_obs=int(config.phase_metric_min_obs),
            feedback_phase=str(config.feedback_phase or "train"),
            benchmark_enabled=bool(config.benchmark_enabled),
            benchmark_code=str(config.benchmark_code or "000300.SH"),
            benchmark_returns=tuple(config.benchmark_returns or ()),
            transaction_cost_config=config.transaction_cost_config,
            enable_recall_validation=bool(config.enable_recall_validation),
            effectiveness_config=FactorEffectivenessConfig(
                ic_abs_stage_a_min=config.effectiveness_ic_abs_min,
                ic_abs_stage_b_min=config.effectiveness_ic_abs_min + 0.005,
                ir_abs_stage_a_min=config.effectiveness_ir_abs_min,
                ir_abs_stage_b_min=config.effectiveness_ir_abs_min + 0.05,
                sharpe_stage_a_min=config.effectiveness_sharpe_min,
                sharpe_stage_b_min=config.effectiveness_sharpe_min + 0.10,
                coverage_hard_reject_min=config.effectiveness_coverage_min,
                coverage_stage_b_min=config.effectiveness_coverage_min,
                turnover_stage_a_max=config.effectiveness_turnover_max,
                turnover_stage_b_max=config.effectiveness_turnover_max - 0.05,
                effective_min_score=config.effectiveness_min_score,
            ),
        ),
    )
    return_semantics = dict(batch_out.get("return_semantics", {}))
    expression_registry_df = load_universe_expression_registry(
        base_dir=config.universe_base_dir,
        universe_name=config.universe_name,
    )
    dashboard_factor_metrics = build_dashboard_factor_metrics(
        factor_metrics_df=batch_out["factor_metrics_df"],
        expression_registry_df=expression_registry_df,
        period=config.analysis_period,
        layers=config.analysis_layers,
    )
    analysis_meta = save_universe_analysis_run(
        base_dir=config.universe_base_dir,
        universe_name=config.universe_name,
        alpha_names=factor_cols,
        period=config.analysis_period,
        layers=config.analysis_layers,
        is_timeseries=config.analysis_is_timeseries,
        factor_metrics_df=batch_out["factor_metrics_df"],
        tables={
            "factor_effectiveness_table": batch_out.get("factor_effectiveness_table", pd.DataFrame()),
            "ic_yearly_df": batch_out.get("ic_yearly_df", pd.DataFrame()),
            "ic_monthly_df": batch_out.get("ic_monthly_df", pd.DataFrame()),
            "period_comparison_df": batch_out.get("period_comparison_df", pd.DataFrame()),
            "double_sort_matrix_returns_df": batch_out.get("double_sort_matrix_returns_df", pd.DataFrame()),
            "double_sort_spread_returns_df": batch_out.get("double_sort_spread_returns_df", pd.DataFrame()),
            "double_sort_summary_df": batch_out.get("double_sort_summary_df", pd.DataFrame()),
            "sample_split_metrics_df": batch_out.get("sample_split_metrics_df", pd.DataFrame()),
            "phase_metrics_df": batch_out.get("phase_metrics_df", pd.DataFrame()),
            "ic_df": batch_out.get("ic_df", pd.DataFrame()),
            "portfolio_pnl_df": batch_out.get("portfolio_pnl_df", pd.DataFrame()),
            "benchmark_pnl_df": batch_out.get("benchmark_pnl_df", pd.DataFrame()),
            "analysis_distribution_histogram": batch_out.get("analysis_distribution_histogram_df", pd.DataFrame()),
            "analysis_ic_decay": batch_out.get("analysis_ic_decay_df", pd.DataFrame()),
            "analysis_factor_coverage_by_date": batch_out.get("analysis_factor_coverage_by_date_df", pd.DataFrame()),
            "direction_policy_df": batch_out.get("direction_policy_df", pd.DataFrame()),
            "phase_local_direction_df": batch_out.get("phase_local_direction_df", pd.DataFrame()),
            "dashboard_factor_metrics": dashboard_factor_metrics,
        },
        extra_meta={
            "analysis_level_mode": config.analysis_level_mode,
            "factor_count": int(len(factor_cols)),
            "closed_loop": True,
            "return_semantics": return_semantics,
            "include_double_sort": bool(config.include_double_sort),
            "apply_tradability_constraints": bool(config.apply_tradability_constraints),
            "include_sample_split_analysis": bool(config.include_sample_split_analysis),
            "sample_split_config": asdict(config.sample_split_config),
            "include_phase_metrics": bool(config.include_phase_metrics),
            "phase_metric_min_obs": int(config.phase_metric_min_obs),
            "feedback_phase": str(config.feedback_phase or "train"),
            "include_visualization_png": bool(config.include_visualization_png),
            "include_full_ic_lag_analysis": bool(
                config.include_visualization_png or config.include_full_ic_lag_analysis
            ),
            "benchmark_config": {
                "enabled": bool(config.benchmark_enabled),
                "code": str(config.benchmark_code or "000300.SH"),
                "view": str(config.benchmark_view or "v_project_index_daily"),
                "date_col": str(config.benchmark_date_col or "date"),
                "code_col": str(config.benchmark_code_col or "code"),
                "close_col": str(config.benchmark_close_col or "close"),
                "return_col": str(config.benchmark_return_col or ""),
            },
            "benchmark_status": dict(config.benchmark_status or {}),
            "transaction_cost_config": config.transaction_cost_config.to_dict(),
            "factor_library_config": {
                "enabled": bool(config.factor_library_enabled),
                "min_score": float(config.factor_library_min_score),
                "staging_min_score": float(config.factor_library_staging_min_score),
                "max_signal_corr": float(config.factor_library_max_signal_corr),
                "max_ic_corr": float(config.factor_library_max_ic_corr),
                "max_pnl_corr": float(config.factor_library_max_pnl_corr),
                "staging_max_corr": float(config.factor_library_staging_max_corr),
            },
            "phase_config": batch_out.get("phase_meta", {}),
        },
    )
    if bool(config.include_visualization_png):
        try:
            manifest_df = save_factor_visualization_artifacts(
                analysis_dir=analysis_meta["analysis_dir"],
                factor_cols=factor_cols,
                df_step2=batch_out.get("df_step2"),
                ic_df=batch_out.get("ic_df"),
                summary_df=batch_out.get("summary_df"),
                lag_analysis_results=batch_out.get("lag_analysis_results"),
                layer_results=batch_out.get("layer_results"),
            )
            manifest_path = Path(analysis_meta["analysis_dir"]) / "visualization_manifest.csv"
            if isinstance(manifest_df, pd.DataFrame) and manifest_path.exists():
                analysis_meta = attach_visualization_manifest_to_analysis_meta(
                    Path(analysis_meta["analysis_dir"]) / "analysis_meta.json",
                    manifest_path,
                )
        except Exception as exc:
            print(f"[closed_loop][warn] visualization artifact export failed: {exc}")
    if bool(config.factor_library_enabled):
        try:
            library_result = submit_factor_library_candidates(
                base_dir=config.universe_base_dir,
                universe_name=config.universe_name,
                run_id=str(analysis_meta.get("analysis_run_id", "")),
                factor_metrics_df=batch_out.get("factor_metrics_df", pd.DataFrame()),
                ic_df=batch_out.get("ic_df", pd.DataFrame()),
                portfolio_pnl_df=batch_out.get("portfolio_pnl_df", pd.DataFrame()),
                signal_df=batch_out.get("df_step2", pd.DataFrame()),
                config=FactorLibraryConfig(
                    enabled=True,
                    min_score=float(config.factor_library_min_score),
                    staging_min_score=float(config.factor_library_staging_min_score),
                    max_signal_corr=float(config.factor_library_max_signal_corr),
                    max_ic_corr=float(config.factor_library_max_ic_corr),
                    max_pnl_corr=float(config.factor_library_max_pnl_corr),
                    staging_max_corr=float(config.factor_library_staging_max_corr),
                    transaction_cost_enabled=bool(config.transaction_cost_config.enabled),
                ),
            )
            analysis_meta["factor_library_result"] = library_result
        except Exception as exc:
            print(f"[closed_loop][warn] factor library submit failed: {exc}")
    return analysis_meta


def _build_panel_store(raw_df: pd.DataFrame, config: ClosedLoopConfig) -> PanelStore:
    date_col = str(config.date_col)
    code_col = str(config.code_col)
    id_alias_excluded: set[str] = set()
    for candidate in ["date", "trade_date"]:
        if candidate in raw_df.columns and candidate != date_col:
            id_alias_excluded.add(candidate)
    for candidate in ["code", "znz_code"]:
        if candidate in raw_df.columns and candidate != code_col:
            id_alias_excluded.add(candidate)

    vector_fields = [v for v in config.vector_fields if v]
    group_fields = [g for g in config.group_fields if g in raw_df.columns]
    neutral_group = neutralization_group_field(getattr(config.mining_config.simulation, "neutralization", "NONE"))
    if neutral_group:
        if neutral_group not in raw_df.columns:
            mode = normalize_neutralization_mode(getattr(config.mining_config.simulation, "neutralization", "NONE"))
            raise ValueError(f"neutralization={mode} requires group field '{neutral_group}' in source data")
        if neutral_group not in group_fields:
            group_fields.append(neutral_group)
    group_set = set(group_fields)
    vector_set = set(vector_fields)
    scalar_fields = [
        c
        for c in raw_df.columns
        if c not in {date_col, code_col} and c not in id_alias_excluded and c not in group_set and c not in vector_set
    ]

    return PanelStore.from_long_frame(
        raw_df.copy(),
        date_col=date_col,
        code_col=code_col,
        scalar_fields=scalar_fields,
        group_fields=group_fields,
        vector_fields=vector_fields,
        max_panel_cache_size=config.panel_cache_max_size,
    )


def _ensure_base_frame(raw_df: pd.DataFrame, config: ClosedLoopConfig) -> None:
    needed = _base_frame_columns_for_config(config)
    missing = [c for c in needed if c not in raw_df.columns]
    work = raw_df
    if missing and bool(config.apply_tradability_constraints):
        tradability_defaults = [c for c in ["can_buy", "can_sell"] if c in missing]
        if tradability_defaults:
            work = raw_df.copy()
            for col in tradability_defaults:
                work[col] = 1
            missing = [c for c in missing if c not in tradability_defaults]
            print(
                "[closed_loop][warn] tradability constraints enabled but "
                f"{tradability_defaults} are missing; assuming all rows are tradable"
            )
    if missing:
        raise ValueError(f"Missing base_frame_cols in raw_df: {missing}")
    base_df = work[needed].copy()
    save_universe_base_frame(
        base_df=base_df,
        base_dir=config.universe_base_dir,
        universe_name=config.universe_name,
    )


def _save_or_update_input_manifest(raw_df: pd.DataFrame, config: ClosedLoopConfig) -> dict[str, Any]:
    impl_registry = build_default_registry()
    sig_registry = build_default_operator_signature_registry()
    payload = {
        "manifest_schema_version": str(config.manifest_schema_version or "v2"),
        "date_col": config.date_col,
        "code_col": config.code_col,
        "group_fields": list(config.group_fields),
        "vector_fields": list(config.vector_fields),
        "include_fields": list(config.include_fields),
        "exclude_fields": list(config.exclude_fields),
        "include_factor_families": list(config.include_factor_families),
        "exclude_factor_families": list(config.exclude_factor_families),
        "enable_family_quota": bool(config.enable_family_quota),
        "family_max_selected_ratio": float(config.family_max_selected_ratio),
        "family_min_explore_ratio": float(config.family_min_explore_ratio),
        "base_frame_cols": list(config.base_frame_cols),
        "include_double_sort": bool(config.include_double_sort),
        "double_sort_control_col": str(config.double_sort_control_col),
        "double_sort_factor_bins": int(config.double_sort_factor_bins),
        "double_sort_control_bins": int(config.double_sort_control_bins),
        "double_sort_method": str(config.double_sort_method),
        "apply_tradability_constraints": bool(config.apply_tradability_constraints),
        "tradability_mode": str(config.tradability_mode),
        "include_sample_split_analysis": bool(config.include_sample_split_analysis),
        "sample_split_config": asdict(config.sample_split_config),
        "include_phase_metrics": bool(config.include_phase_metrics),
        "phase_metric_min_obs": int(config.phase_metric_min_obs),
        "feedback_phase": str(config.feedback_phase or "train"),
        "include_visualization_png": bool(config.include_visualization_png),
        "benchmark_config": {
            "enabled": bool(config.benchmark_enabled),
            "code": str(config.benchmark_code or "000300.SH"),
            "view": str(config.benchmark_view or "v_project_index_daily"),
            "date_col": str(config.benchmark_date_col or "date"),
            "code_col": str(config.benchmark_code_col or "code"),
            "close_col": str(config.benchmark_close_col or "close"),
            "return_col": str(config.benchmark_return_col or ""),
        },
        "benchmark_status": dict(config.benchmark_status or {}),
        "source_backend": str(config.source_backend or "file"),
        "duckdb_path": str(config.duckdb_path or ""),
        "source_view": str(config.source_view or ""),
        "date_range": {
            "start": str(config.source_date_range[0] if len(config.source_date_range) >= 1 else ""),
            "end": str(config.source_date_range[1] if len(config.source_date_range) >= 2 else ""),
        },
        "field_catalog_version": str(config.field_catalog_version or ""),
        "moneyflow_source": str(config.moneyflow_source or "moneyflow"),
        "field_preprocessing_config": asdict(config.field_preprocessing_config),
        "simulation_config": _canonical_simulation_config_dict(config.mining_config.simulation),
        "operator_registry": {
            "implemented_count": int(len(impl_registry.list_names())),
            "operators": impl_registry.list_names(),
        },
        "signature_registry": {
            "signed_count": int(len(sig_registry.names())),
            "operators": sig_registry.names(),
        },
        "run_filters": dict(config.run_filters or {}),
        "search_field_source": str(config.search_field_source or ""),
        "search_field_universe_count": int(len(config.search_field_universe)),
        "search_mode": str(config.search_mode or ""),
        "layer_max_order": int(config.layer_max_order),
        "layer_max_candidates": int(config.layer_max_candidates),
        "layer_budgets": dict(config.layer_budgets or {}),
        "enable_stateful_phase2_ops": bool(config.enable_stateful_phase2_ops),
        "enable_feedback_mutation": bool(config.enable_feedback_mutation),
        "mutation_budget_ratio": float(config.mutation_budget_ratio),
        "mutation_max_children_per_parent": int(config.mutation_max_children_per_parent),
        "mutation_fragment_cooldown_batches": int(config.mutation_fragment_cooldown_batches),
        "mutation_fragment_max_age_batches": int(config.mutation_fragment_max_age_batches),
        "mutation_stateful_ratio_cap": float(config.mutation_stateful_ratio_cap),
        "mutation_min_selected_count": int(config.mutation_min_selected_count),
        "mutation_min_selected_ratio": float(config.mutation_min_selected_ratio),
        "enable_source_chunk_loading": bool(config.enable_source_chunk_loading),
        "source_chunk_mem_warn_mb": float(config.source_chunk_mem_warn_mb),
        "duckdb_memory_limit": str(config.duckdb_memory_limit or ""),
        "duckdb_threads": int(config.duckdb_threads),
        "duckdb_temp_directory": str(config.duckdb_temp_directory or ""),
        "duckdb_max_temp_directory_size": str(config.duckdb_max_temp_directory_size or ""),
        "source_path": str(config.input_source_path or ""),
        "snapshot_path": str(config.snapshot_path or ""),
        "columns": [str(c) for c in raw_df.columns],
        "dtypes": {str(c): str(raw_df[c].dtype) for c in raw_df.columns},
        "row_count": int(len(raw_df)),
    }
    manifest_id = _derive_manifest_id(raw_df=raw_df, config=config)
    saved = save_universe_input_manifest(
        manifest=payload,
        base_dir=config.universe_base_dir,
        universe_name=config.universe_name,
        manifest_id=manifest_id,
    )
    return {**payload, **saved}


def _base_frame_columns_for_config(config: ClosedLoopConfig) -> list[str]:
    needed: list[str] = []

    def _append(name: str) -> None:
        text = str(name or "").strip()
        if text and text not in needed:
            needed.append(text)

    for col in config.base_frame_cols:
        _append(str(col))
    if bool(config.include_double_sort):
        _append(str(config.double_sort_control_col))
        _append("circ_mv")
    if bool(config.apply_tradability_constraints):
        _append("can_buy")
        _append("can_sell")
    return needed


def _simulation_required_fields(config: ClosedLoopConfig) -> list[str]:
    needed: list[str] = []

    def _append(name: str) -> None:
        text = str(name or "").strip()
        if text and text not in needed:
            needed.append(text)

    simulation = config.mining_config.simulation
    simulation_universe = str(getattr(simulation, "universe", "") or "").strip()
    if simulation_universe:
        _append(simulation_universe)
    group_field = neutralization_group_field(getattr(simulation, "neutralization", "NONE"))
    if group_field:
        _append(group_field)
    return needed


def _canonical_simulation_config_dict(simulation: Any) -> dict[str, Any]:
    payload = asdict(simulation) if hasattr(simulation, "__dataclass_fields__") else dict(simulation or {})
    payload["neutralization"] = normalize_neutralization_mode(str(payload.get("neutralization", "NONE")))
    return payload


def _scoreboard_score(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    # 优先使用 effectiveness_score（0-100 分制，语义最清晰）
    for col in [
        "effectiveness_score",
        "feedback_score",
        "feedback_score_net",
        "train_score_total",
        "train_score_total_net",
        "train_score",
        "score_total",
        "score_total_net",
        "feedback_score_gross",
        "score_total_gross",
        "scoreboard_score",
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return values.fillna(values.median(skipna=True) if values.notna().any() else 0.0)
    score = np.zeros(len(df), dtype=np.float64)
    for col, w in [
        ("ir", 0.20),
        ("ic_mean", 0.20),
        ("best_layer_annualized_return", 0.18),
        ("best_layer_sharpe", 0.16),
        ("best_minus_benchmark_annualized_return", 0.14),
        ("best_minus_universe_annualized_return", 0.06),
        ("margin_long_only", 0.08),
        ("turnover_long_only_mean", -0.04),
    ]:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        std = float(s.std(skipna=True))
        if std > 0:
            z = (s - s.mean(skipna=True)) / std
        else:
            z = s * 0.0
        score = score + float(w) * np.asarray(z.fillna(0.0), dtype=np.float64)
    return pd.Series(score, index=df.index)


def _chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    size = max(1, int(chunk_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _chunk_parallel_lists(
    expressions: list[str],
    alpha_names: list[str],
    chunk_size: int,
) -> list[tuple[list[str], list[str]]]:
    if len(expressions) != len(alpha_names):
        raise ValueError("expressions and alpha_names length mismatch")
    size = max(1, int(chunk_size))
    out: list[tuple[list[str], list[str]]] = []
    for i in range(0, len(expressions), size):
        out.append((expressions[i : i + size], alpha_names[i : i + size]))
    return out


def _filter_new_signals(
    expressions: list[str],
    seen_signal_hashes: set[str],
    simulation_config_json: str,
) -> list[str]:
    out: list[str] = []
    local_seen: set[str] = set()
    for expr in expressions:
        text = str(expr or "").strip()
        if not text:
            continue
        sig_hash = signal_hash_for_expression(text, simulation_config_json)
        if sig_hash in seen_signal_hashes or sig_hash in local_seen:
            continue
        out.append(text)
        local_seen.add(sig_hash)
    return out


def _panel_signature_hash(raw_df: pd.DataFrame) -> str:
    payload = {
        "columns": [str(c) for c in raw_df.columns],
        "dtypes": {str(c): str(raw_df[c].dtype) for c in raw_df.columns},
        "row_count": int(len(raw_df)),
    }
    txt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()


def _derive_manifest_id(raw_df: pd.DataFrame, config: ClosedLoopConfig) -> str:
    base = {
        "universe_name": str(config.universe_name),
        "date_col": str(config.date_col),
        "code_col": str(config.code_col),
        "group_fields": list(config.group_fields),
        "vector_fields": list(config.vector_fields),
        "include_fields": list(config.include_fields),
        "exclude_fields": list(config.exclude_fields),
        "include_factor_families": list(config.include_factor_families),
        "exclude_factor_families": list(config.exclude_factor_families),
        "enable_family_quota": bool(config.enable_family_quota),
        "family_max_selected_ratio": float(config.family_max_selected_ratio),
        "family_min_explore_ratio": float(config.family_min_explore_ratio),
        "base_frame_cols": list(config.base_frame_cols),
        "include_double_sort": bool(config.include_double_sort),
        "double_sort_control_col": str(config.double_sort_control_col),
        "double_sort_factor_bins": int(config.double_sort_factor_bins),
        "double_sort_control_bins": int(config.double_sort_control_bins),
        "double_sort_method": str(config.double_sort_method),
        "apply_tradability_constraints": bool(config.apply_tradability_constraints),
        "tradability_mode": str(config.tradability_mode),
        "include_sample_split_analysis": bool(config.include_sample_split_analysis),
        "sample_split_config": asdict(config.sample_split_config),
        "include_phase_metrics": bool(config.include_phase_metrics),
        "phase_metric_min_obs": int(config.phase_metric_min_obs),
        "feedback_phase": str(config.feedback_phase or "train"),
        "include_visualization_png": bool(config.include_visualization_png),
        "source_backend": str(config.source_backend or ""),
        "duckdb_path": str(config.duckdb_path or ""),
        "source_view": str(config.source_view or ""),
        "source_date_range": list(config.source_date_range),
        "field_catalog_version": str(config.field_catalog_version or ""),
        "manifest_schema_version": str(config.manifest_schema_version or "v2"),
        "run_filters": dict(config.run_filters or {}),
        "search_field_source": str(config.search_field_source or ""),
        "search_field_universe_count": int(len(config.search_field_universe)),
        "search_mode": str(config.search_mode or ""),
        "layer_max_order": int(config.layer_max_order),
        "layer_max_candidates": int(config.layer_max_candidates),
        "layer_budgets": dict(config.layer_budgets or {}),
        "enable_stateful_phase2_ops": bool(config.enable_stateful_phase2_ops),
        "enable_feedback_mutation": bool(config.enable_feedback_mutation),
        "mutation_budget_ratio": float(config.mutation_budget_ratio),
        "mutation_max_children_per_parent": int(config.mutation_max_children_per_parent),
        "mutation_fragment_cooldown_batches": int(config.mutation_fragment_cooldown_batches),
        "mutation_fragment_max_age_batches": int(config.mutation_fragment_max_age_batches),
        "mutation_stateful_ratio_cap": float(config.mutation_stateful_ratio_cap),
        "mutation_min_selected_count": int(config.mutation_min_selected_count),
        "mutation_min_selected_ratio": float(config.mutation_min_selected_ratio),
        "enable_source_chunk_loading": bool(config.enable_source_chunk_loading),
        "source_chunk_mem_warn_mb": float(config.source_chunk_mem_warn_mb),
        "duckdb_memory_limit": str(config.duckdb_memory_limit or ""),
        "duckdb_threads": int(config.duckdb_threads),
        "duckdb_temp_directory": str(config.duckdb_temp_directory or ""),
        "duckdb_max_temp_directory_size": str(config.duckdb_max_temp_directory_size or ""),
        "source_path": str(config.input_source_path or ""),
        "snapshot_path": str(config.snapshot_path or ""),
        "simulation_config": _canonical_simulation_config_dict(config.mining_config.simulation),
        "panel_signature_hash": _panel_signature_hash(raw_df),
    }
    digest = hashlib.sha1(json.dumps(base, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"manifest_{digest}"


def _use_source_chunk_loading(config: ClosedLoopConfig) -> bool:
    if not bool(config.enable_source_chunk_loading):
        return False
    if str(config.source_backend or "").strip().lower() != "duckdb":
        return False
    if not str(config.duckdb_path or "").strip():
        return False
    if not str(config.source_view or "").strip():
        return False
    return True


def _duckdb_settings_from_config(config: ClosedLoopConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    memory_limit = str(getattr(config, "duckdb_memory_limit", "") or "").strip()
    if memory_limit:
        out["memory_limit"] = memory_limit
    try:
        threads = int(getattr(config, "duckdb_threads", 0) or 0)
    except Exception:
        threads = 0
    if threads > 0:
        out["threads"] = threads
    temp_directory = str(getattr(config, "duckdb_temp_directory", "") or "").strip()
    if temp_directory:
        out["temp_directory"] = temp_directory
    max_temp = str(getattr(config, "duckdb_max_temp_directory_size", "") or "").strip()
    if max_temp:
        out["max_temp_directory_size"] = max_temp
    return out


class SourceChunkMemoryLimitError(MemoryError):
    pass


def _classify_closed_loop_failure(exc: BaseException, stage: str = "") -> dict[str, str]:
    text = f"{type(exc).__name__}: {exc}".lower()
    permanent_tokens = (
        "missing required columns",
        "missing_sample_fields",
        "expression missing",
        "not found in panelstore",
        "unsupported neutralization",
        "parse",
        "type",
        "invalid group",
    )
    retryable_tokens = (
        "permission",
        "locked",
        "duckdb",
        "memory",
        "ioerror",
        "oserror",
        "empty frame",
        "temporar",
    )
    if any(token in text for token in permanent_tokens) and not any(token in text for token in retryable_tokens):
        return {
            "status": "PERMANENT_FAILED",
            "failure_kind": "permanent",
            "stage": str(stage or ""),
        }
    return {"status": "FAILED", "failure_kind": "retryable", "stage": str(stage or "")}


def _failure_status_counts(config: ClosedLoopConfig) -> dict[str, int]:
    lifecycle = load_lifecycle_registry(base_dir=config.universe_base_dir, universe_name=config.universe_name)
    if lifecycle.empty or "status" not in lifecycle.columns:
        return {}
    return {
        str(status): int(count)
        for status, count in lifecycle["status"].fillna("").astype(str).value_counts().to_dict().items()
        if str(status)
    }


def _prune_closed_loop_artifacts(config: ClosedLoopConfig) -> dict[str, Any]:
    root = get_universe_paths(base_dir=config.universe_base_dir, universe_name=config.universe_name)["root"]
    candidate_summary = (
        prune_candidate_artifacts(
            root=root,
            max_batches=int(config.candidate_artifact_retention_max_batches),
            retention_days=int(config.candidate_artifact_retention_days),
        )
        if bool(config.candidate_artifact_retention_enabled)
        else {"enabled": False}
    )
    analysis_summary = (
        prune_analysis_artifacts(
            root=root,
            max_runs=int(config.analysis_artifact_retention_max_runs),
            retention_days=int(config.analysis_artifact_retention_days),
        )
        if bool(config.analysis_artifact_retention_enabled)
        else {"enabled": False}
    )
    return {"candidate": candidate_summary, "analysis": analysis_summary}


def prune_analysis_artifacts(root: str | Path, *, max_runs: int = 120, retention_days: int = 90) -> dict[str, Any]:
    root_path = Path(root)
    analysis_dir = root_path / "analysis"
    summary: dict[str, Any] = {
        "enabled": True,
        "scanned_dirs": 0,
        "deleted_dirs": 0,
        "deleted_bytes": 0,
        "failed_paths": [],
        "retained_runs": 0,
    }
    if not analysis_dir.exists():
        return summary
    run_dirs = [p for p in analysis_dir.glob("period_*/analysis_*") if p.is_dir()]
    summary["scanned_dirs"] = int(len(run_dirs))
    now = time.time()
    cutoff_age = max(0, int(retention_days)) * 86400
    ordered = sorted(
        run_dirs,
        key=lambda p: (_latest_mtime(p), p.as_posix()),
        reverse=True,
    )
    keep: set[Path] = set()
    for idx, path in enumerate(ordered):
        keep_by_count = idx < max(0, int(max_runs))
        keep_by_age = cutoff_age > 0 and (now - _latest_mtime(path)) <= cutoff_age
        if keep_by_count or keep_by_age:
            keep.add(path)
    summary["retained_runs"] = int(len(keep))
    for path in ordered:
        if path in keep:
            continue
        try:
            size = _path_size_bytes(path)
            shutil.rmtree(path)
            summary["deleted_dirs"] = int(summary["deleted_dirs"]) + 1
            summary["deleted_bytes"] = int(summary["deleted_bytes"]) + int(size)
        except Exception:
            summary.setdefault("failed_paths", []).append(str(path.as_posix()))
    return summary


def validate_universe_registries(base_dir: str | Path, universe_name: str) -> dict[str, Any]:
    paths = get_universe_paths(base_dir=base_dir, universe_name=universe_name)
    root = paths["root"]
    registry_specs = [
        ("expression", paths["expression_csv"], ["alpha_name", "expression"]),
        ("lifecycle", paths["lifecycle_csv"], ["alpha_name", "status"]),
        ("analysis", paths["analysis_registry_csv"], ["alpha_name", "analysis_run_id"]),
        (
            "factor_metrics",
            paths["factor_metrics_registry_csv"],
            ["factor", "analysis_run_id"],
        ),
        (
            "factor_library",
            root / "library" / "factor_library_registry.csv",
            list(FACTOR_LIBRARY_REGISTRY_COLUMNS[:3]),
        ),
    ]
    items: list[dict[str, Any]] = []
    recovered = False
    for name, path, required_cols in registry_specs:
        items.append(_validate_registry_file(name=name, path=Path(path), required_cols=required_cols))
        recovered = recovered or items[-1]["status"] == "recovered_from_backup"
    status_counts = {
        str(status): int(count)
        for status, count in pd.Series([x["status"] for x in items]).value_counts().to_dict().items()
    }
    return {
        "registry_count": int(len(items)),
        "registry_recovered_from_backup": bool(recovered),
        "status_counts": status_counts,
        "registries": items,
    }


def _validate_registry_file(*, name: str, path: Path, required_cols: list[str]) -> dict[str, Any]:
    backup = path.with_suffix(path.suffix + ".bak")
    if not path.exists() and not backup.exists():
        return {
            "name": name,
            "path": str(path.as_posix()),
            "status": "missing",
            "missing_columns": list(required_cols),
            "row_count": 0,
        }
    try:
        frame = pd.read_csv(path)
        source = "current"
        status = "ok"
    except Exception as exc:
        if not backup.exists():
            return {
                "name": name,
                "path": str(path.as_posix()),
                "status": "read_error",
                "error": str(exc),
                "missing_columns": list(required_cols),
                "row_count": 0,
            }
        try:
            frame = pd.read_csv(backup)
            source = "backup"
            status = "recovered_from_backup"
        except Exception as backup_exc:
            return {
                "name": name,
                "path": str(path.as_posix()),
                "status": "read_error",
                "error": str(backup_exc),
                "missing_columns": list(required_cols),
                "row_count": 0,
            }
    missing = [col for col in required_cols if col not in frame.columns]
    if missing and status == "ok":
        status = "missing_columns"
    return {
        "name": name,
        "path": str(path.as_posix()),
        "status": status,
        "source": source,
        "missing_columns": missing,
        "row_count": int(len(frame)),
    }


def _latest_mtime(path: Path) -> float:
    try:
        if path.is_file():
            return float(path.stat().st_mtime)
        mtimes = [float(p.stat().st_mtime) for p in path.rglob("*") if p.exists()]
        mtimes.append(float(path.stat().st_mtime))
        return max(mtimes)
    except Exception:
        return 0.0


def _load_recent_iteration_results(config: ClosedLoopConfig, window: int = 10) -> list[dict[str, Any]]:
    """从 run_health.jsonl 读取最近 N 次迭代结果，用于自适应探索率计算。"""
    paths = get_universe_paths(base_dir=config.universe_base_dir, universe_name=config.universe_name)
    health_path = paths["feedback_dir"] / "run_health.jsonl"
    if not health_path.exists():
        return []
    try:
        lines = health_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception:
        return []
    results: list[dict[str, Any]] = []
    for line in reversed(lines[-window:]):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        status = str(record.get("status", ""))
        selected_count = int(record.get("selected_count", 0) or 0)
        results.append(
            {
                "status": status,
                "alpha_names": [f"alpha_{i}" for i in range(selected_count)] if selected_count > 0 else [],
            }
        )
    return list(reversed(results))


def _path_size_bytes(path: Path) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
        return int(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()))
    except Exception:
        return 0


def _append_run_health(
    *,
    config: ClosedLoopConfig,
    result: dict[str, Any],
    candidate_meta: dict[str, Any],
    selected_meta: pd.DataFrame,
    source_chunk_metas: list[dict[str, Any]],
    retention_summary: dict[str, Any],
    elapsed_seconds: float,
) -> str:
    paths = get_universe_paths(base_dir=config.universe_base_dir, universe_name=config.universe_name)
    path = paths["feedback_dir"] / "run_health.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    registry_health = validate_universe_registries(
        base_dir=config.universe_base_dir, universe_name=config.universe_name
    )
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": str(result.get("status", "")),
        "universe": str(config.universe_name),
        "elapsed_seconds": float(elapsed_seconds),
        "candidate_count": int(candidate_meta.get("candidate_count", 0) or 0)
        if isinstance(candidate_meta, dict)
        else 0,
        "passed_candidate_count": int(candidate_meta.get("passed_candidate_count", 0) or 0)
        if isinstance(candidate_meta, dict)
        else 0,
        "selected_count": int(len(selected_meta)) if isinstance(selected_meta, pd.DataFrame) else 0,
        "selected_layer_counts": _value_counts_dict(selected_meta.get("layer"))
        if isinstance(selected_meta, pd.DataFrame) and "layer" in selected_meta.columns
        else {},
        "sample_prefilter": dict(candidate_meta.get("layered_generation_diagnostics", {}).get("sample_prefilter", {}))
        if isinstance(candidate_meta, dict)
        else {},
        "source_chunk_memory": _summarize_source_chunk_memory(source_chunk_metas),
        "duckdb_settings": _duckdb_settings_from_config(config),
        "artifact_retention_summary": dict(retention_summary or {}),
        "analysis_retention_summary": dict((retention_summary or {}).get("analysis", {}))
        if isinstance(retention_summary, dict)
        else {},
        "registry_health_summary": registry_health,
        "registry_recovered_from_backup": bool(registry_health.get("registry_recovered_from_backup", False)),
        "failure_status_counts": dict(result.get("failure_status_counts", {}) or {}),
        "scoreboard_rows": int(result.get("scoreboard_rows", 0) or 0),
        "source_chunk_hard_limit_triggered": bool(result.get("source_chunk_hard_limit_triggered", False)),
    }
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_lines = [line for line in existing.splitlines() if line.strip()]
    lines, health_retention_summary = _trim_run_health_lines(
        existing_lines=existing_lines, new_record=record, config=config
    )
    record["run_health_retention_summary"] = health_retention_summary
    lines[-1] = json.dumps(record, ensure_ascii=False, sort_keys=True)
    text = "\n".join(lines) + "\n"
    atomic_write_text(path, text, encoding="utf-8", backup=True)
    return str(path.as_posix())


def _trim_run_health_lines(
    *, existing_lines: list[str], new_record: dict[str, Any], config: ClosedLoopConfig
) -> tuple[list[str], dict[str, Any]]:
    new_line = json.dumps(new_record, ensure_ascii=False, sort_keys=True)
    lines = list(existing_lines) + [new_line]
    if not bool(config.run_health_retention_enabled):
        return lines, {
            "enabled": False,
            "scanned_lines": len(lines),
            "pruned_lines": 0,
            "retained_lines": len(lines),
        }
    scanned = len(lines)
    cutoff_age = max(0, int(config.run_health_retention_days)) * 86400
    if cutoff_age > 0:
        now = time.time()
        retained_by_age: list[str] = []
        for line in lines:
            try:
                payload = json.loads(line)
                ts = datetime.fromisoformat(str(payload.get("created_at_utc", ""))).timestamp()
                if (now - ts) <= cutoff_age:
                    retained_by_age.append(line)
            except Exception:
                retained_by_age.append(line)
        lines = retained_by_age
    max_lines = max(1, int(config.run_health_retention_max_lines))
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return lines, {
        "enabled": True,
        "scanned_lines": int(scanned),
        "pruned_lines": int(max(0, scanned - len(lines))),
        "retained_lines": int(len(lines)),
        "max_lines": int(max_lines),
        "retention_days": int(config.run_health_retention_days),
    }


def _value_counts_dict(series: Any) -> dict[str, int]:
    if series is None:
        return {}
    try:
        return {
            str(key): int(value)
            for key, value in pd.Series(series).fillna("").astype(str).value_counts().to_dict().items()
            if str(key)
        }
    except Exception:
        return {}


def _summarize_source_chunk_memory(chunk_metas: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [dict(x) for x in chunk_metas if isinstance(x, dict) and x]
    if not valid:
        return {
            "max_chunk_mem_mb": 0.0,
            "source_chunk_mem_warning_count": 0,
            "source_chunk_mem_warning_chunks": [],
        }
    max_mem = max(float(x.get("mem_mb", 0.0) or 0.0) for x in valid)
    warning_chunks = [x for x in valid if bool(x.get("mem_warning", False))]
    return {
        "max_chunk_mem_mb": float(max_mem),
        "source_chunk_mem_warning_count": int(len(warning_chunks)),
        "source_chunk_mem_warning_chunks": warning_chunks,
    }


def _log_candidate_distributions(candidate_meta: dict[str, Any]) -> None:
    candidate_df = candidate_meta.get("candidate_df", pd.DataFrame())
    if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty and "layer" in candidate_df.columns:
        generated = _format_value_counts(candidate_df.get("layer"))
        passed_df = candidate_df
        rejected_df = candidate_df
        if "prefilter_status" in candidate_df.columns:
            passed_df = candidate_df[candidate_df["prefilter_status"].astype(str) == "pass"]
            rejected_df = candidate_df[candidate_df["prefilter_status"].astype(str) != "pass"]
        print(
            f"[closed_loop][layers] generated={generated or '{}'} "
            f"passed={_format_value_counts(passed_df.get('layer')) or '{}'} "
            f"rejected={_format_value_counts(rejected_df.get('layer')) or '{}'}"
        )

    sample_df = candidate_meta.get("sample_df", pd.DataFrame())
    if isinstance(sample_df, pd.DataFrame) and not sample_df.empty and "sample_status" in sample_df.columns:
        print(f"[closed_loop][sample_prefilter] status={_format_value_counts(sample_df.get('sample_status')) or '{}'}")


def _log_selected_candidate_distributions(selected_meta: pd.DataFrame) -> None:
    if selected_meta is None or selected_meta.empty:
        return
    layer_counts = _format_value_counts(selected_meta.get("layer")) if "layer" in selected_meta.columns else ""
    family_counts = _format_value_counts(selected_meta.get("family")) if "family" in selected_meta.columns else ""
    factor_family_counts = (
        _format_value_counts(selected_meta.get("factor_family")) if "factor_family" in selected_meta.columns else ""
    )
    if layer_counts or family_counts or factor_family_counts:
        print(
            f"[closed_loop][selected] layers={layer_counts or '{}'} "
            f"families={family_counts or '{}'} "
            f"factor_families={factor_family_counts or '{}'}"
        )


def _format_value_counts(series: pd.Series | None) -> str:
    if series is None:
        return ""
    values = [str(x).strip() for x in series.dropna().astype(str).tolist() if str(x).strip()]
    if not values:
        return ""
    counts = pd.Series(values).value_counts().sort_index()
    return ",".join(f"{idx}:{int(value)}" for idx, value in counts.items())


def _normalize_closed_loop_input(
    raw_df: pd.DataFrame,
    config: ClosedLoopConfig,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        raise ValueError("raw_df must not be empty")

    date_col = str(config.date_col)
    code_col = str(config.code_col)
    cols = set(str(c) for c in raw_df.columns)
    work = raw_df
    changed = False

    date_source = _resolve_column_alias(
        columns=cols,
        preferred=date_col,
        aliases=("date", "trade_date"),
    )
    if not date_source:
        raise ValueError(f"Could not resolve date column. Expected '{date_col}' or aliases ['date', 'trade_date'].")
    if date_col not in work.columns:
        if not changed:
            work = raw_df.copy(deep=False)
            changed = True
        work[date_col] = work[date_source]
        cols.add(date_col)

    code_source = _resolve_column_alias(
        columns=cols,
        preferred=code_col,
        aliases=("code", "znz_code"),
    )
    if not code_source:
        raise ValueError(f"Could not resolve code column. Expected '{code_col}' or aliases ['code', 'znz_code'].")
    if code_col not in work.columns:
        if not changed:
            work = raw_df.copy(deep=False)
            changed = True
        work[code_col] = work[code_source]
        cols.add(code_col)

    # Keep canonical aliases for downstream modules that rely on date/code.
    if "date" not in work.columns and date_col in work.columns:
        if not changed:
            work = raw_df.copy(deep=False)
            changed = True
        work["date"] = work[date_col]
        cols.add("date")
    if "code" not in work.columns and code_col in work.columns:
        if not changed:
            work = raw_df.copy(deep=False)
            changed = True
        work["code"] = work[code_col]
        cols.add("code")

    # Normalize date dtype once so downstream sorting/groupby are stable.
    if not pd.api.types.is_datetime64_any_dtype(work[date_col]):
        if not changed:
            work = raw_df.copy(deep=False)
            changed = True
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        if "date" in work.columns and date_col != "date":
            work["date"] = work[date_col]

    need_pct_chg = ("pct_chg" in set(config.base_frame_cols)) or (str(config.analysis_return_col).strip() == "pct_chg")
    if need_pct_chg and "pct_chg" not in work.columns:
        close_col = _resolve_column_alias(
            columns=set(str(c) for c in work.columns),
            preferred="close",
            aliases=("close", "adj_close", "close_price", "last_price"),
        )
        if not close_col:
            raise ValueError(
                "Missing 'pct_chg' and could not auto-build it because no close-like column "
                "was found (tried: close/adj_close/close_price/last_price)."
            )
        if not changed:
            work = raw_df.copy(deep=False)
            changed = True
        close_num = pd.to_numeric(work[close_col], errors="coerce")
        order_index = work[[code_col, date_col]].sort_values([code_col, date_col], kind="mergesort").index
        pct_sorted = close_num.loc[order_index].groupby(work.loc[order_index, code_col], sort=False).pct_change()
        pct_sorted = pct_sorted.replace([np.inf, -np.inf], np.nan)
        work["pct_chg"] = np.nan
        work.loc[order_index, "pct_chg"] = np.asarray(pct_sorted, dtype=np.float64)

    return work


def _resolve_column_alias(
    columns: set[str],
    preferred: str,
    aliases: tuple[str, ...],
) -> str | None:
    pref = str(preferred or "").strip()
    if pref and pref in columns:
        return pref
    for name in aliases:
        if str(name) in columns:
            return str(name)
    return None


def _normalized_excluded_fields(config: ClosedLoopConfig) -> set[str]:
    return {str(x).strip() for x in config.exclude_fields if str(x).strip()}


def _acquire_loop_lock(
    lock_path: Path,
    timeout_seconds: float,
    universe_name: str = "",
    config_hash: str = "",
) -> dict[str, Any]:
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and lock_path.is_file():
        _remove_stale_lock(lock_path)
    owner = _lock_owner_payload(universe_name=universe_name, config_hash=config_hash)
    try:
        lock_path.mkdir()
    except FileExistsError:
        current = _read_lock_owner(lock_path)
        age = _lock_age_seconds(current)
        if float(timeout_seconds) > 0 and age <= float(timeout_seconds):
            raise RuntimeError(
                f"closed_loop lock exists: {lock_path} (age={age:.1f}s <= timeout={float(timeout_seconds):.1f}s)"
            )
        _remove_stale_lock(lock_path)
        lock_path.mkdir()
    atomic_write_json(lock_path / "owner.json", owner)
    return owner


def _release_loop_lock(lock_path: Path, owner_id: str = "") -> None:
    lock_path = Path(lock_path)
    try:
        if not lock_path.exists():
            return
        if lock_path.is_file():
            lock_path.unlink()
            return
        if owner_id:
            current = _read_lock_owner(lock_path)
            if str(current.get("owner_id", "")) != str(owner_id):
                return
        for child in lock_path.iterdir():
            if child.is_file():
                child.unlink()
        lock_path.rmdir()
    except Exception:
        pass


def _lock_owner_payload(*, universe_name: str, config_hash: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "owner_id": uuid.uuid4().hex,
        "pid": int(os.getpid()),
        "hostname": socket.gethostname(),
        "started_at_utc": now,
        "heartbeat_at_utc": now,
        "universe": str(universe_name or ""),
        "config_hash": str(config_hash or ""),
    }


def _start_loop_lock_heartbeat(
    *,
    lock_path: Path,
    owner_id: str,
    timeout_seconds: float,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    interval = 30.0
    if timeout_seconds > 0:
        interval = max(5.0, min(60.0, float(timeout_seconds) / 4.0))

    def _worker() -> None:
        while not stop.wait(interval):
            _refresh_loop_lock_heartbeat(lock_path=lock_path, owner_id=owner_id)

    thread = threading.Thread(target=_worker, name="closed-loop-lock-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def _refresh_loop_lock_heartbeat(*, lock_path: Path, owner_id: str) -> bool:
    if not owner_id:
        return False
    lock_path = Path(lock_path)
    owner_path = lock_path / "owner.json"
    try:
        owner = _read_lock_owner(lock_path)
        if str(owner.get("owner_id", "")) != str(owner_id):
            return False
        owner["heartbeat_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(owner_path, owner)
        return True
    except Exception:
        return False


def _read_lock_owner(lock_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads((Path(lock_path) / "owner.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _lock_age_seconds(owner: dict[str, Any]) -> float:
    value = str(owner.get("heartbeat_at_utc") or owner.get("started_at_utc") or "")
    try:
        ts = datetime.fromisoformat(value).timestamp()
    except Exception:
        return float("inf")
    return max(0.0, time.time() - ts)


def _remove_stale_lock(lock_path: Path) -> None:
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return
    if lock_path.is_file():
        lock_path.unlink()
        return
    for child in lock_path.iterdir():
        if child.is_file():
            child.unlink()
    lock_path.rmdir()
