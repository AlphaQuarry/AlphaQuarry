from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from factor_research import (
    FactorEffectivenessConfig,
    TransactionCostConfig,
    analyze_holding_period_robustness,
    SampleSplitConfig,
    assign_sample_split,
    build_phase_windows,
    build_portfolio_pnl_table,
    build_return_semantics_metadata,
    calculate_best_layer_metrics,
    calculate_factor_coverage,
    calculate_ic_stability,
    calculate_ic_time_breakdown,
    calculate_icir,
    calculate_layer_monotonicity,
    calculate_layer_portfolio_turnover,
    calculate_long10_portfolio_returns,
    calculate_long_short_metrics,
    calculate_long_only_portfolio_turnover,
    calculate_margin_metrics,
    calculate_turnover_rate,
    double_sort_analysis,
    evaluate_factor_effectiveness,
    factor_layer_analysis,
    process_factor_data,
    process_future_return,
    summarize_split_metrics,
    summarize_long10_portfolio_returns,
    summarize_long_only_turnover,
)
from factor_research.screening import (
    compute_effectiveness_score_for_basis,
    compute_effectiveness_score_parts,
)
from factor_research.utils import calculate_risk_metrics, get_logger

from ..adapters import to_factor_research_frame
from .analysis_data_artifacts import (
    build_distribution_histogram_table,
    build_phase_ic_decay_table,
)
from .direction_policy import build_direction_policy_tables, direction_sign_map


@dataclass(frozen=True)
class AnalysisLevelConfig:
    mode: str = "full"  # light / full / light_then_full_on_survivors


@dataclass(frozen=True)
class RecallValidationConfig:
    """召回率验证配置。"""

    enabled: bool = False
    sample_ratio: float = 0.05
    min_sample_size: int = 2
    max_sample_size: int = 10


@dataclass(frozen=True)
class BatchAnalysisConfig:
    period: int = 1
    layers: int = 10
    is_timeseries: bool = True
    return_col: str = "pct_chg"
    market_value_column: str = "circ_mv"
    # Keep defaults aligned with notebook behavior.
    do_neutralize: bool = False
    do_standardize: bool = False
    max_lag: int = 10
    include_full_ic_lag_analysis: bool = True

    robust_periods: tuple[int, ...] = (1, 5, 10, 20)
    include_robustness: bool = True
    robustness_store_detailed_results: bool = False
    robustness_run_gc_per_period: bool = True

    analysis_level: AnalysisLevelConfig = AnalysisLevelConfig()
    apply_filtering: bool = True
    effectiveness_config: FactorEffectivenessConfig | None = None
    signal_delay: int = 1
    include_double_sort: bool = False
    double_sort_control_col: str = "total_mv"
    double_sort_fallback_control_col: str = "circ_mv"
    double_sort_factor_bins: int = 5
    double_sort_control_bins: int = 5
    double_sort_method: str = "conditional"
    apply_tradability_constraints: bool = True
    tradability_mode: str = "entry_exit"
    can_buy_col: str = "can_buy"
    can_sell_col: str = "can_sell"
    long10_count: int = 10
    include_sample_split_analysis: bool = False
    sample_split_config: SampleSplitConfig = SampleSplitConfig()
    include_phase_metrics: bool = True
    phase_metric_min_obs: int = 1
    feedback_phase: str = "train"
    benchmark_enabled: bool = True
    benchmark_code: str = "000300.SH"
    benchmark_returns: tuple[dict[str, Any], ...] = ()
    transaction_cost_config: TransactionCostConfig = TransactionCostConfig()
    enable_recall_validation: bool = False
    recall_validation_config: RecallValidationConfig = RecallValidationConfig()


@contextmanager
def _analysis_stage_timer(stage: str, *, rows: int | None = None, factors: int | None = None):
    start = perf_counter()
    try:
        yield
    finally:
        get_logger().info(
            "[analysis_timing] stage=%s elapsed_seconds=%.2f rows=%s factors=%s",
            stage,
            perf_counter() - start,
            rows if rows is not None else "-",
            factors if factors is not None else "-",
        )


def build_factor_research_input(
    base_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    date_col: str = "date",
    code_col: str = "code",
) -> pd.DataFrame:
    return to_factor_research_frame(
        raw_df=base_df,
        alpha_wide_df=alpha_df,
        code_col=code_col,
        date_col=date_col,
    )


def run_factor_analysis_batch(
    df_raw: pd.DataFrame,
    factor_cols: Sequence[str],
    config: BatchAnalysisConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BatchAnalysisConfig()
    mode = str(cfg.analysis_level.mode or "full").strip().lower()
    if mode not in {"light", "full", "light_then_full_on_survivors"}:
        raise ValueError(f"Unsupported analysis_level.mode: {cfg.analysis_level.mode}")

    factors = [str(c) for c in factor_cols if str(c)]
    if not factors:
        raise ValueError("factor_cols must not be empty")
    _validate_required_columns(df_raw, factors, cfg)

    light_outputs = run_factor_analysis_batch_light(df_raw=df_raw, factor_cols=factors, config=cfg)
    outputs = dict(light_outputs)

    if mode == "light":
        outputs["analysis_level_mode"] = mode
        return outputs

    if mode == "full":
        full_factors = factors
    else:
        full_factors = (
            outputs["effective_factors_df"]["factor"].astype(str).tolist()
            if isinstance(outputs.get("effective_factors_df"), pd.DataFrame)
            and not outputs["effective_factors_df"].empty
            else []
        )

    # 轻量评估召回验证：从被过滤因子中抽样做 full 评估
    if mode == "light_then_full_on_survivors" and bool(cfg.enable_recall_validation):
        recall_result = validate_light_filter_recall(
            df_raw=df_raw,
            all_factors=factors,
            light_survivors=full_factors,
            config=cfg,
            validation_config=cfg.recall_validation_config,
        )
        outputs["recall_validation"] = recall_result

    full_outputs = run_factor_analysis_batch_full(
        df_raw=df_raw,
        base_outputs=outputs,
        full_factor_cols=full_factors,
        config=cfg,
    )
    outputs.update(full_outputs)
    outputs["factor_metrics_df"] = _merge_optional_factor_metrics(
        outputs.get("factor_metrics_df", pd.DataFrame()),
        [
            outputs.get("double_sort_summary_df", pd.DataFrame()),
            outputs.get("sample_split_metrics_df", pd.DataFrame()),
            outputs.get("phase_metrics_df", pd.DataFrame()),
        ],
    )
    outputs["analysis_level_mode"] = mode
    return outputs


def run_factor_analysis_batch_light(
    df_raw: pd.DataFrame,
    factor_cols: Sequence[str],
    config: BatchAnalysisConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BatchAnalysisConfig()
    factors = [str(c) for c in factor_cols if str(c)]
    total_start = perf_counter()
    raw_rows = int(len(df_raw))
    factor_count = int(len(factors))

    with _analysis_stage_timer("process_future_return", rows=raw_rows, factors=factor_count):
        df_step1 = process_future_return(df_raw.copy(), return_col=cfg.return_col, period=cfg.period)
    future_col = f"{cfg.return_col}_{cfg.period}d"
    extra_cols = _analysis_extra_columns(df_step1, cfg)
    cols_step2 = _dedupe_keep_order(
        ["trade_date", "znz_code", cfg.return_col, future_col, cfg.market_value_column] + extra_cols + factors
    )
    with _analysis_stage_timer("process_factor_data", rows=int(len(df_step1)), factors=factor_count):
        df_step2 = process_factor_data(
            df_step1[cols_step2].copy(),
            factor_cols=factors,
            market_value_column=cfg.market_value_column,
            is_timeseries=cfg.is_timeseries,
            do_neutralize=cfg.do_neutralize,
            do_standardize=cfg.do_standardize,
        )

    step2_rows = int(len(df_step2))
    with _analysis_stage_timer("calculate_icir", rows=step2_rows, factors=factor_count):
        icir_out = calculate_icir(
            df_step2,
            factor_cols=factors,
            return_col=cfg.return_col,
            period=cfg.period,
            max_lag=int(cfg.max_lag) if bool(cfg.include_full_ic_lag_analysis) else None,
        )
    if len(icir_out) == 2:
        ic_df, summary_df = icir_out
        lag_analysis_results = []
    else:
        ic_df, summary_df, lag_analysis_results = icir_out
    direction_policy_df, phase_local_direction_df = build_direction_policy_tables(
        ic_df=ic_df,
        factors=factors,
        sample_split_config=cfg.sample_split_config,
    )
    train_locked_ic_signs = direction_sign_map(direction_policy_df)

    with _analysis_stage_timer("factor_layer_analysis", rows=step2_rows, factors=factor_count):
        layer_results = factor_layer_analysis(
            df_step2,
            factor_cols=factors,
            return_col=cfg.return_col,
            period=cfg.period,
            layers=cfg.layers,
            passthrough_cols=[cfg.can_buy_col, cfg.can_sell_col] if cfg.apply_tradability_constraints else None,
        )
    with _analysis_stage_timer("calculate_long_short_metrics", rows=step2_rows, factors=factor_count):
        long_short_metrics, layer_results_for_visualization = calculate_long_short_metrics(
            layer_results,
            period=cfg.period,
            direction_mode="by_ic_sign",
            ic_summary_df=summary_df,
            ic_signs_override=train_locked_ic_signs,
        )
    with _analysis_stage_timer("calculate_turnover_rate", rows=step2_rows, factors=factor_count):
        turnover_results = calculate_turnover_rate(layer_results, period=cfg.period)
        membership_turnover_summary_df = turnover_results_to_summary(turnover_results, factors=factors)
    with _analysis_stage_timer("calculate_best_layer_metrics", rows=step2_rows, factors=factor_count):
        best_layer_metrics_df = calculate_best_layer_metrics(
            layer_results,
            ic_summary_df=summary_df,
            period=cfg.period,
            ic_signs_override=train_locked_ic_signs,
        )
    with _analysis_stage_timer("calculate_long_only_portfolio_turnover", rows=step2_rows, factors=factor_count):
        long_only_turnover_results = calculate_long_only_portfolio_turnover(
            layer_results,
            ic_summary_df=summary_df,
            period=cfg.period,
            apply_tradability_constraints=cfg.apply_tradability_constraints,
            tradability_mode=cfg.tradability_mode,
            can_buy_col=cfg.can_buy_col,
            can_sell_col=cfg.can_sell_col,
            transaction_cost_config=cfg.transaction_cost_config,
            ic_signs_override=train_locked_ic_signs,
        )
        long_only_turnover_summary_df = summarize_long_only_turnover(long_only_turnover_results, factors=factors)
        margin_metrics_df = calculate_margin_metrics(long_only_turnover_results, factors=factors)
    with _analysis_stage_timer("calculate_long10_portfolio_returns", rows=step2_rows, factors=factor_count):
        long10_portfolio_returns = calculate_long10_portfolio_returns(
            layer_results,
            ic_summary_df=summary_df,
            top_n=cfg.long10_count,
            period=cfg.period,
            apply_tradability_constraints=cfg.apply_tradability_constraints,
            tradability_mode=cfg.tradability_mode,
            can_buy_col=cfg.can_buy_col,
            can_sell_col=cfg.can_sell_col,
            transaction_cost_config=cfg.transaction_cost_config,
            ic_signs_override=train_locked_ic_signs,
        )
        long10_portfolio_summary_df = summarize_long10_portfolio_returns(
            long10_portfolio_returns,
            factors=factors,
            period=cfg.period,
        )
    with _analysis_stage_timer("calculate_layer_portfolio_turnover", rows=step2_rows, factors=factor_count):
        layer_turnover_results = calculate_layer_portfolio_turnover(
            layer_results,
            period=cfg.period,
            transaction_cost_config=cfg.transaction_cost_config,
        )
    with _analysis_stage_timer("build_portfolio_pnl_table", rows=step2_rows, factors=factor_count):
        portfolio_pnl_df = build_portfolio_pnl_table(
            layer_results_for_visualization=layer_results_for_visualization,
            long_only_turnover_results=long_only_turnover_results,
            long10_portfolio_returns=long10_portfolio_returns,
            turnover_results=turnover_results,
            layer_turnover_results=layer_turnover_results,
            transaction_cost_config=cfg.transaction_cost_config,
        )
    net_score_inputs_df = _build_net_score_inputs(
        portfolio_pnl_df=portfolio_pnl_df,
        best_layer_metrics_df=best_layer_metrics_df,
        period=cfg.period,
    )
    benchmark_pnl_df = _build_benchmark_pnl_table(
        benchmark_returns=cfg.benchmark_returns,
        benchmark_enabled=bool(cfg.benchmark_enabled),
    )
    with _analysis_stage_timer("calculate_factor_coverage", rows=step2_rows, factors=factor_count):
        coverage = calculate_factor_coverage(df_step2, factors)
    with _analysis_stage_timer("calculate_ic_stability", rows=int(len(ic_df)), factors=factor_count):
        ic_stability_df = calculate_ic_stability(ic_df, factors)
    with _analysis_stage_timer("calculate_ic_time_breakdown", rows=int(len(ic_df)), factors=factor_count):
        ic_yearly_df = calculate_ic_time_breakdown(ic_df, factors, freq="Y")
        ic_monthly_df = calculate_ic_time_breakdown(ic_df, factors, freq="M")
    with _analysis_stage_timer("calculate_layer_monotonicity", rows=step2_rows, factors=factor_count):
        monotonicity = calculate_layer_monotonicity(layer_results)
    return_semantics = build_return_semantics_metadata(
        base_return_col=cfg.return_col,
        period=cfg.period,
        signal_delay=cfg.signal_delay,
    )
    with _analysis_stage_timer("build_phase_ic_decay_table", rows=step2_rows, factors=factor_count):
        analysis_ic_decay_df = (
            build_phase_ic_decay_table(
                df_step2,
                factors,
                return_col=cfg.return_col,
                period=cfg.period,
                max_lag=cfg.max_lag,
                sample_split_config=cfg.sample_split_config,
            )
            if cfg.include_phase_metrics
            else pd.DataFrame()
        )

    with _analysis_stage_timer("evaluate_factor_effectiveness", rows=step2_rows, factors=factor_count):
        eff_outputs = evaluate_factor_effectiveness(
            summary_df=summary_df,
            long_short_metrics=long_short_metrics,
            layer_results=layer_results,
            best_layer_metrics_df=best_layer_metrics_df,
            long_only_turnover_summary_df=long_only_turnover_summary_df,
            margin_metrics_df=margin_metrics_df,
            coverage_overall_df=coverage.get("overall"),
            coverage_by_date_df=coverage.get("by_date"),
            ic_stability_df=ic_stability_df,
            monotonicity_summary_df=monotonicity.get("summary"),
            turnover_results=turnover_results,
            ic_yearly_df=ic_yearly_df,
            ic_monthly_df=ic_monthly_df,
            ic_decay_df=analysis_ic_decay_df,
            period_comparison_df=None,
            apply_filtering=cfg.apply_filtering,
            config=cfg.effectiveness_config,
        )
    factor_effectiveness_table = eff_outputs["factor_effectiveness_table"]
    factor_effectiveness_table = _apply_score_basis_aliases(
        factor_effectiveness_table,
        net_score_inputs_df=net_score_inputs_df,
        cfg=cfg,
    )
    factor_effectiveness_table = _merge_optional_factor_metrics(
        factor_effectiveness_table,
        [direction_policy_df],
    )
    effective_factors_df = _filter_effective_from_table(
        factor_effectiveness_table, cfg=cfg, apply_filtering=cfg.apply_filtering
    )

    with _analysis_stage_timer("long_short_metrics_to_frame", rows=step2_rows, factors=factor_count):
        long_short_df = long_short_metrics_to_frame(long_short_metrics)
    with _analysis_stage_timer("double_sort_analysis", rows=step2_rows, factors=factor_count):
        double_sort_outputs = (
            double_sort_analysis(
                df_step2,
                factor_cols=factors,
                return_col=future_col,
                control_col=cfg.double_sort_control_col,
                fallback_control_col=cfg.double_sort_fallback_control_col,
                factor_bins=cfg.double_sort_factor_bins,
                control_bins=cfg.double_sort_control_bins,
                method=cfg.double_sort_method,
            )
            if cfg.include_double_sort
            else {
                "matrix_returns_df": pd.DataFrame(),
                "spread_returns_df": pd.DataFrame(),
                "summary_df": pd.DataFrame(),
            }
        )
    with _analysis_stage_timer("build_factor_metrics_table", rows=step2_rows, factors=factor_count):
        factor_metrics_df = build_factor_metrics_table(
            summary_df=summary_df,
            long_short_df=long_short_df,
            turnover_summary_df=membership_turnover_summary_df,
            monotonicity_summary_df=monotonicity.get("summary", pd.DataFrame()),
            best_layer_metrics_df=best_layer_metrics_df,
            long_only_turnover_summary_df=long_only_turnover_summary_df,
            long10_portfolio_summary_df=long10_portfolio_summary_df,
            margin_metrics_df=margin_metrics_df,
            effective_factors_df=effective_factors_df,
            factor_effectiveness_df=factor_effectiveness_table,
            return_semantics=return_semantics,
        )
        factor_metrics_df = _add_benchmark_relative_metrics(
            factor_metrics_df,
            portfolio_pnl_df=portfolio_pnl_df,
            benchmark_pnl_df=benchmark_pnl_df,
            period=cfg.period,
        )
    # Attach layer diagnostics (quantile fallback, insufficient stock dates)
    _layer_diag_rows = []
    for _factor, _lr_df in layer_results.items():
        _layer_diag_rows.append(
            {
                "factor": _factor,
                "layer_qcut_fallback_count": _lr_df.attrs.get("qcut_fallback_count", 0),
                "layer_fewer_groups_count": _lr_df.attrs.get("fewer_groups_count", 0),
                "layer_total_dates": _lr_df.attrs.get("total_dates", 0),
                "layer_insufficient_stock_dates": _lr_df.attrs.get("insufficient_stock_dates", 0),
            }
        )
    if (
        _layer_diag_rows
        and isinstance(factor_metrics_df, pd.DataFrame)
        and not factor_metrics_df.empty
        and "factor" in factor_metrics_df.columns
    ):
        factor_metrics_df = factor_metrics_df.merge(pd.DataFrame(_layer_diag_rows), on="factor", how="left")
    compute_phase_metrics = bool(cfg.include_phase_metrics or cfg.include_sample_split_analysis)
    if compute_phase_metrics:
        with _analysis_stage_timer("build_phase_metrics_table", rows=step2_rows, factors=factor_count):
            phase_metrics_df, phase_meta = build_phase_metrics_table(
                factors=factors,
                ic_df=ic_df,
                factor_metrics_df=factor_metrics_df,
                portfolio_pnl_df=portfolio_pnl_df,
                benchmark_pnl_df=benchmark_pnl_df,
                cfg=cfg,
            )
    else:
        phase_metrics_df, phase_meta = pd.DataFrame(), {}
    sample_split_metrics_df = (
        phase_metrics_to_legacy_sample_split_metrics(phase_metrics_df) if compute_phase_metrics else pd.DataFrame()
    )
    with _analysis_stage_timer("build_distribution_histogram_table", rows=step2_rows, factors=factor_count):
        analysis_distribution_histogram_df = (
            build_distribution_histogram_table(
                df_step2,
                factors,
                sample_split_config=cfg.sample_split_config,
            )
            if cfg.include_phase_metrics
            else pd.DataFrame()
        )
    with _analysis_stage_timer("_merge_optional_factor_metrics", rows=step2_rows, factors=factor_count):
        factor_metrics_df = _merge_optional_factor_metrics(
            factor_metrics_df,
            [
                double_sort_outputs.get("summary_df", pd.DataFrame()),
                sample_split_metrics_df,
                phase_metrics_df if cfg.include_phase_metrics else pd.DataFrame(),
            ],
        )

    get_logger().info(
        "[analysis_timing] stage=%s elapsed_seconds=%.2f rows=%s factors=%s",
        "run_factor_analysis_batch_light_total",
        perf_counter() - total_start,
        raw_rows,
        factor_count,
    )

    return {
        "config": cfg,
        "factor_cols": factors,
        "df_step1": df_step1,
        "df_step2": df_step2,
        "ic_df": ic_df,
        "summary_df": summary_df,
        "lag_analysis_results": lag_analysis_results,
        "layer_results": layer_results,
        "long_short_metrics": long_short_metrics,
        "layer_results_for_visualization": layer_results_for_visualization,
        "turnover_results": turnover_results,
        "membership_turnover_summary_df": membership_turnover_summary_df,
        "best_layer_metrics_df": best_layer_metrics_df,
        "long_only_turnover_results": long_only_turnover_results,
        "long_only_turnover_summary_df": long_only_turnover_summary_df,
        "long10_portfolio_returns": long10_portfolio_returns,
        "long10_portfolio_summary_df": long10_portfolio_summary_df,
        "layer_turnover_results": layer_turnover_results,
        "portfolio_pnl_df": portfolio_pnl_df,
        "benchmark_pnl_df": benchmark_pnl_df,
        "margin_metrics_df": margin_metrics_df,
        "return_semantics": return_semantics,
        "coverage": coverage,
        "ic_stability_df": ic_stability_df,
        "monotonicity": monotonicity,
        "ic_yearly_df": ic_yearly_df,
        "ic_monthly_df": ic_monthly_df,
        "robustness": None,
        "period_comparison_df": pd.DataFrame(),
        "factor_effectiveness_table": factor_effectiveness_table,
        "effective_factors_df": effective_factors_df,
        "long_short_df": long_short_df,
        "turnover_summary_df": membership_turnover_summary_df,
        "factor_metrics_df": factor_metrics_df,
        "double_sort_matrix_returns_df": double_sort_outputs.get("matrix_returns_df", pd.DataFrame()),
        "double_sort_spread_returns_df": double_sort_outputs.get("spread_returns_df", pd.DataFrame()),
        "double_sort_summary_df": double_sort_outputs.get("summary_df", pd.DataFrame()),
        "sample_split_metrics_df": sample_split_metrics_df,
        "phase_metrics_df": phase_metrics_df if cfg.include_phase_metrics else pd.DataFrame(),
        "phase_meta": phase_meta if cfg.include_phase_metrics else {},
        "analysis_distribution_histogram_df": analysis_distribution_histogram_df,
        "analysis_ic_decay_df": analysis_ic_decay_df,
        "analysis_factor_coverage_by_date_df": coverage.get("by_date", pd.DataFrame()),
        "direction_policy_df": direction_policy_df,
        "phase_local_direction_df": phase_local_direction_df,
        "net_score_inputs_df": net_score_inputs_df,
    }


def run_factor_analysis_batch_full(
    df_raw: pd.DataFrame,
    base_outputs: dict[str, Any],
    full_factor_cols: Sequence[str],
    config: BatchAnalysisConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BatchAnalysisConfig()
    full_factors = [str(c) for c in full_factor_cols if str(c)]
    if not full_factors:
        return {
            "ic_yearly_df": pd.DataFrame(),
            "ic_monthly_df": pd.DataFrame(),
            "robustness": None,
            "period_comparison_df": pd.DataFrame(),
        }

    ic_df = base_outputs["ic_df"]
    summary_df = base_outputs["summary_df"]
    long_short_metrics = base_outputs["long_short_metrics"]
    turnover_results = base_outputs["turnover_results"]
    best_layer_metrics_df = base_outputs.get("best_layer_metrics_df", pd.DataFrame())
    long_only_turnover_summary_df = base_outputs.get("long_only_turnover_summary_df", pd.DataFrame())
    long10_portfolio_summary_df = base_outputs.get("long10_portfolio_summary_df", pd.DataFrame())
    margin_metrics_df = base_outputs.get("margin_metrics_df", pd.DataFrame())
    monotonicity = base_outputs["monotonicity"]
    coverage = base_outputs["coverage"]
    ic_stability_df = base_outputs["ic_stability_df"]
    layer_results = base_outputs["layer_results"]

    base_yearly = base_outputs.get("ic_yearly_df", pd.DataFrame())
    if isinstance(base_yearly, pd.DataFrame) and not base_yearly.empty and "factor" in base_yearly.columns:
        ic_yearly_df = base_yearly[base_yearly["factor"].astype(str).isin(full_factors)].copy()
    else:
        ic_yearly_df = calculate_ic_time_breakdown(ic_df, full_factors, freq="Y")
    base_monthly = base_outputs.get("ic_monthly_df", pd.DataFrame())
    if isinstance(base_monthly, pd.DataFrame) and not base_monthly.empty and "factor" in base_monthly.columns:
        ic_monthly_df = base_monthly[base_monthly["factor"].astype(str).isin(full_factors)].copy()
    else:
        ic_monthly_df = calculate_ic_time_breakdown(ic_df, full_factors, freq="M")

    robustness = None
    period_comparison_df = pd.DataFrame()
    if cfg.include_robustness and cfg.robust_periods:
        precomputed_period_details = {
            int(cfg.period): {
                "summary_df": summary_df,
                "long_short_metrics": long_short_metrics,
                "turnover_results": turnover_results,
                "monotonicity_summary": monotonicity.get("summary"),
                "ic_df": ic_df,
                "layer_results": layer_results,
            }
        }
        robustness = analyze_holding_period_robustness(
            df=df_raw,
            factor_cols=full_factors,
            periods=list(cfg.robust_periods),
            return_col=cfg.return_col,
            layers=cfg.layers,
            market_value_column=cfg.market_value_column,
            is_timeseries=cfg.is_timeseries,
            do_neutralize=cfg.do_neutralize,
            do_standardize=cfg.do_standardize,
            store_detailed_results=cfg.robustness_store_detailed_results,
            run_gc_per_period=cfg.robustness_run_gc_per_period,
            already_computed_periods=[cfg.period],
            precomputed_period_details=precomputed_period_details,
        )
        period_comparison_df = robustness.get("period_comparison_df", robustness.get("comparison", pd.DataFrame()))

    full_summary_df = summary_df[summary_df["factor"].astype(str).isin(full_factors)].copy()
    full_long_short_metrics = {k: v for k, v in long_short_metrics.items() if str(k) in set(full_factors)}
    full_turnover_results = {k: v for k, v in turnover_results.items() if str(k) in set(full_factors)}
    full_mono = monotonicity.get("summary", pd.DataFrame())
    if isinstance(full_mono, pd.DataFrame) and not full_mono.empty and "factor" in full_mono.columns:
        full_mono = full_mono[full_mono["factor"].astype(str).isin(full_factors)].copy()

    eff_outputs_full = evaluate_factor_effectiveness(
        summary_df=full_summary_df,
        long_short_metrics=full_long_short_metrics,
        layer_results=layer_results,
        best_layer_metrics_df=best_layer_metrics_df,
        long_only_turnover_summary_df=long_only_turnover_summary_df,
        margin_metrics_df=margin_metrics_df,
        coverage_overall_df=coverage.get("overall"),
        coverage_by_date_df=coverage.get("by_date"),
        ic_stability_df=ic_stability_df,
        monotonicity_summary_df=full_mono,
        turnover_results=full_turnover_results,
        ic_yearly_df=ic_yearly_df,
        ic_monthly_df=ic_monthly_df,
        ic_decay_df=base_outputs.get("analysis_ic_decay_df", pd.DataFrame()),
        period_comparison_df=period_comparison_df,
        apply_filtering=cfg.apply_filtering,
        config=cfg.effectiveness_config,
    )

    merged_effectiveness = _merge_effectiveness_tables(
        base_outputs.get("factor_effectiveness_table", pd.DataFrame()),
        eff_outputs_full["factor_effectiveness_table"],
    )
    merged_effectiveness = _apply_score_basis_aliases(
        merged_effectiveness,
        net_score_inputs_df=base_outputs.get("net_score_inputs_df", pd.DataFrame()),
        cfg=cfg,
    )
    effective_factors_df = _filter_effective_from_table(
        merged_effectiveness, cfg=cfg, apply_filtering=cfg.apply_filtering
    )
    legacy_cols = _legacy_effective_columns()
    for col in legacy_cols:
        if col not in effective_factors_df.columns:
            effective_factors_df[col] = np.nan
    effective_factors_df = effective_factors_df[
        legacy_cols + [c for c in effective_factors_df.columns if c not in legacy_cols]
    ]

    factor_metrics_df = build_factor_metrics_table(
        summary_df=base_outputs["summary_df"],
        long_short_df=base_outputs["long_short_df"],
        turnover_summary_df=base_outputs["turnover_summary_df"],
        monotonicity_summary_df=base_outputs["monotonicity"].get("summary", pd.DataFrame()),
        best_layer_metrics_df=best_layer_metrics_df,
        long_only_turnover_summary_df=long_only_turnover_summary_df,
        long10_portfolio_summary_df=long10_portfolio_summary_df,
        margin_metrics_df=margin_metrics_df,
        effective_factors_df=effective_factors_df,
        factor_effectiveness_df=merged_effectiveness,
        return_semantics=base_outputs.get("return_semantics", {}),
    )

    return {
        "ic_yearly_df": ic_yearly_df,
        "ic_monthly_df": ic_monthly_df,
        "robustness": robustness,
        "period_comparison_df": period_comparison_df,
        "factor_effectiveness_table": merged_effectiveness,
        "effective_factors_df": effective_factors_df,
        "factor_metrics_df": factor_metrics_df,
    }


def long_short_metrics_to_frame(
    long_short_metrics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if not long_short_metrics:
        return pd.DataFrame(
            columns=[
                "factor",
                "long_short_total_return",
                "long_short_annualized_return",
                "long_short_volatility",
                "long_short_sharpe_ratio",
                "long_short_max_drawdown",
                "long_short_fitness_ratio",
            ]
        )

    rows: list[dict[str, Any]] = []
    for factor, metrics in long_short_metrics.items():
        rows.append(
            {
                "factor": str(factor),
                "long_short_total_return": metrics.get("total_return", np.nan),
                "long_short_annualized_return": metrics.get("annualized_return", np.nan),
                "long_short_volatility": metrics.get("volatility", np.nan),
                "long_short_sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
                "long_short_max_drawdown": metrics.get("max_drawdown", np.nan),
                "long_short_fitness_ratio": metrics.get("fitness_ratio", np.nan),
            }
        )
    return pd.DataFrame(rows)


def turnover_results_to_summary(
    turnover_results: dict[str, pd.DataFrame],
    factors: Sequence[str] | None = None,
) -> pd.DataFrame:
    factor_list = [str(x) for x in (factors or turnover_results.keys())]
    rows: list[dict[str, Any]] = []
    for factor in factor_list:
        tr = turnover_results.get(factor)
        if tr is None or len(tr) == 0:
            rows.append(
                {
                    "factor": factor,
                    "avg_min_layer_turnover": np.nan,
                    "avg_max_layer_turnover": np.nan,
                    "membership_turnover_worst_layer": np.nan,
                    "membership_turnover_best_layer": np.nan,
                }
            )
            continue
        min_avg = float(pd.to_numeric(tr["min_layer_turnover"], errors="coerce").mean())
        max_avg = float(pd.to_numeric(tr["max_layer_turnover"], errors="coerce").mean())
        rows.append(
            {
                "factor": factor,
                "avg_min_layer_turnover": min_avg,
                "avg_max_layer_turnover": max_avg,
                "membership_turnover_worst_layer": min_avg,
                "membership_turnover_best_layer": max_avg,
            }
        )
    return pd.DataFrame(rows)


def build_factor_metrics_table(
    summary_df: pd.DataFrame,
    long_short_df: pd.DataFrame,
    turnover_summary_df: pd.DataFrame,
    monotonicity_summary_df: pd.DataFrame,
    best_layer_metrics_df: pd.DataFrame | None = None,
    long_only_turnover_summary_df: pd.DataFrame | None = None,
    long10_portfolio_summary_df: pd.DataFrame | None = None,
    margin_metrics_df: pd.DataFrame | None = None,
    effective_factors_df: pd.DataFrame | None = None,
    factor_effectiveness_df: pd.DataFrame | None = None,
    return_semantics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    merged = summary_df.copy()
    for frame in [
        long_short_df,
        turnover_summary_df,
        monotonicity_summary_df,
        best_layer_metrics_df,
        long_only_turnover_summary_df,
        long10_portfolio_summary_df,
        margin_metrics_df,
    ]:
        if isinstance(frame, pd.DataFrame) and not frame.empty and "factor" in frame.columns:
            merged = pd.merge(merged, frame, on="factor", how="left")

    if (
        isinstance(factor_effectiveness_df, pd.DataFrame)
        and not factor_effectiveness_df.empty
        and "factor" in factor_effectiveness_df.columns
    ):
        enrich_cols = [
            c
            for c in [
                "factor",
                "stage_a_pass",
                "stage_b_pass",
                "passed_hard_filter",
                "score_predictive_power",
                "score_long_only_performance",
                "score_stability",
                "score_tradeability",
                "score_total",
                "effectiveness_score",
                "effectiveness_tier",
                "fail_reasons",
                "warning_reasons",
                "yearly_sign_consistency",
                "monthly_sign_consistency",
                "ic_decay_spearman",
                "robust_period_positive_ratio",
                "robust_ic_sign_consistency",
                "robust_ir_median",
                "score_predictive_power_gross",
                "score_predictive_power_net",
                "score_long_only_performance_gross",
                "score_long_only_performance_net",
                "score_stability_gross",
                "score_stability_net",
                "score_tradeability_gross",
                "score_tradeability_net",
                "score_total_gross",
                "score_total_net",
                "score_total_basis",
                "effectiveness_score_gross",
                "effectiveness_score_net",
                "effectiveness_tier_gross",
                "effectiveness_tier_net",
                "has_net_score",
                "best_layer_total_return_gross",
                "best_layer_total_return_net",
                "best_layer_annualized_return_gross",
                "best_layer_annualized_return_net",
                "best_layer_volatility_gross",
                "best_layer_volatility_net",
                "best_layer_sharpe_gross",
                "best_layer_sharpe_net",
                "best_layer_max_drawdown_gross",
                "best_layer_max_drawdown_net",
                "best_layer_fitness_ratio_gross",
                "best_layer_fitness_ratio_net",
                "best_minus_universe_annualized_return_gross",
                "best_minus_universe_annualized_return_net",
                "best_layer_positive_month_ratio_gross",
                "best_layer_positive_month_ratio_net",
                "best_layer_margin_gross",
                "best_layer_margin_net",
                "margin_long_only_gross",
                "margin_long_only_net",
                "direction_policy",
                "direction_source_phase",
                "direction_sign",
                "direction_ic_mean",
                "direction_obs",
                "best_layer_direction_train_locked",
            ]
            if c in factor_effectiveness_df.columns
        ]
        merged = pd.merge(merged, factor_effectiveness_df[enrich_cols], on="factor", how="left")

    for key, value in dict(return_semantics or {}).items():
        if key not in merged.columns:
            merged[key] = value

    effective_set = set()
    if (
        isinstance(effective_factors_df, pd.DataFrame)
        and not effective_factors_df.empty
        and "factor" in effective_factors_df.columns
    ):
        effective_set = set(effective_factors_df["factor"].astype(str).tolist())
    merged["is_effective"] = merged["factor"].astype(str).isin(effective_set)
    merged["alpha_name"] = merged["factor"].astype(str)
    return merged


def _build_net_score_inputs(
    *,
    portfolio_pnl_df: pd.DataFrame | None,
    best_layer_metrics_df: pd.DataFrame | None,
    period: int,
) -> pd.DataFrame:
    if portfolio_pnl_df is None or portfolio_pnl_df.empty or "factor" not in portfolio_pnl_df.columns:
        return pd.DataFrame()
    pnl = portfolio_pnl_df.copy()
    pnl["factor"] = pnl["factor"].astype(str)
    if "has_net_pnl" not in pnl.columns or "return_net" not in pnl.columns:
        return pd.DataFrame()
    best_map: dict[str, int] = {}
    if isinstance(best_layer_metrics_df, pd.DataFrame) and not best_layer_metrics_df.empty:
        if {"factor", "best_layer_label"}.issubset(best_layer_metrics_df.columns):
            for _, row in best_layer_metrics_df.iterrows():
                factor = str(row.get("factor") or "")
                try:
                    best_map[factor] = int(row.get("best_layer_label"))
                except Exception:
                    continue
    rows: list[dict[str, Any]] = []
    for factor, group in pnl.groupby("factor", sort=False):
        row: dict[str, Any] = {"factor": str(factor), "has_net_score": False}
        best_label = best_map.get(str(factor))
        if best_label is not None:
            layer = group[group["portfolio"].astype(str) == f"layer_{best_label}"].copy()
            row.update(_portfolio_score_columns(layer, prefix="best_layer", period=period))
            universe_ann = np.nan
            if (
                isinstance(best_layer_metrics_df, pd.DataFrame)
                and "universe_equal_weight_annualized_return" in best_layer_metrics_df.columns
            ):
                match = best_layer_metrics_df[best_layer_metrics_df["factor"].astype(str) == str(factor)]
                if not match.empty:
                    universe_ann = _to_float_or_nan(match.iloc[0].get("universe_equal_weight_annualized_return"))
            net_ann = _to_float_or_nan(row.get("best_layer_annualized_return_net"))
            gross_ann = _to_float_or_nan(row.get("best_layer_annualized_return_gross"))
            if np.isfinite(universe_ann):
                row["best_minus_universe_annualized_return_net"] = (
                    net_ann - universe_ann if np.isfinite(net_ann) else np.nan
                )
                row["best_minus_universe_annualized_return_gross"] = (
                    gross_ann - universe_ann if np.isfinite(gross_ann) else np.nan
                )
            row["best_layer_positive_month_ratio_net"] = _positive_month_ratio_local(
                layer, return_col="return_net", period=period, net_only=True
            )
            row["best_layer_positive_month_ratio_gross"] = _positive_month_ratio_local(
                layer, return_col="return_gross", period=period
            )
            row["best_layer_margin_net"] = _margin_for_portfolio(layer, return_col="return_net", net_only=True)
            row["best_layer_margin_gross"] = _margin_for_portfolio(layer, return_col="return_gross", net_only=False)

        long_only = group[group["portfolio"].astype(str) == "long_only"].copy()
        row["margin_long_only_net"] = _margin_for_portfolio(long_only, return_col="return_net", net_only=True)
        row["margin_long_only_gross"] = _margin_for_portfolio(long_only, return_col="return_gross", net_only=False)
        if _has_net_returns(long_only):
            row["has_net_score"] = True
        elif best_label is not None and _has_net_returns(
            group[group["portfolio"].astype(str) == f"layer_{best_label}"]
        ):
            row["has_net_score"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def _portfolio_score_columns(frame: pd.DataFrame, *, prefix: str, period: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for basis, return_col in [("gross", "return_gross"), ("net", "return_net")]:
        if basis == "net" and not _has_net_returns(frame):
            continue
        returns = pd.to_numeric(
            frame.get(return_col, frame.get("return", pd.Series(dtype=float))),
            errors="coerce",
        )
        returns = returns.dropna()
        if returns.empty:
            continue
        metrics = calculate_risk_metrics(returns, period=period)
        out[f"{prefix}_total_return_{basis}"] = metrics.get("total_return", np.nan)
        out[f"{prefix}_annualized_return_{basis}"] = metrics.get("annualized_return", np.nan)
        out[f"{prefix}_volatility_{basis}"] = metrics.get("volatility", np.nan)
        out[f"{prefix}_sharpe_{basis}"] = metrics.get("sharpe_ratio", np.nan)
        out[f"{prefix}_max_drawdown_{basis}"] = metrics.get("max_drawdown", np.nan)
        out[f"{prefix}_fitness_ratio_{basis}"] = metrics.get("fitness_ratio", np.nan)
    return out


def _apply_score_basis_aliases(
    table: pd.DataFrame,
    *,
    net_score_inputs_df: pd.DataFrame | None,
    cfg: BatchAnalysisConfig,
) -> pd.DataFrame:
    if table is None or table.empty or "factor" not in table.columns:
        return pd.DataFrame() if table is None else table
    out = table.copy()
    out["factor"] = out["factor"].astype(str)
    if (
        isinstance(net_score_inputs_df, pd.DataFrame)
        and not net_score_inputs_df.empty
        and "factor" in net_score_inputs_df.columns
    ):
        net = net_score_inputs_df.copy()
        net["factor"] = net["factor"].astype(str)
        out = pd.merge(out, net, on="factor", how="left", suffixes=("", "_net_input"))
    for base in [
        "best_layer_total_return",
        "best_layer_annualized_return",
        "best_layer_volatility",
        "best_layer_sharpe",
        "best_layer_max_drawdown",
        "best_layer_fitness_ratio",
        "best_minus_universe_annualized_return",
        "best_minus_benchmark_annualized_return",
        "best_layer_positive_month_ratio",
        "best_layer_margin",
        "margin_long_only",
        "portfolio_return_long_only_sum",
    ]:
        gross_col = f"{base}_gross"
        if base in out.columns and gross_col not in out.columns:
            out[gross_col] = out[base]
    if "has_net_score" not in out.columns:
        out["has_net_score"] = False
    cfg_eff = (
        cfg.effectiveness_config
        if isinstance(cfg.effectiveness_config, FactorEffectivenessConfig)
        else FactorEffectivenessConfig()
    )
    score_cols = [
        "score_predictive_power",
        "score_long_only_performance",
        "score_stability",
        "score_tradeability",
        "score_total",
    ]
    records: list[dict[str, Any]] = []
    for _, source in out.iterrows():
        row = dict(source)
        gross = compute_effectiveness_score_for_basis(row, basis="gross", config=cfg_eff)
        has_net = bool(row.get("has_net_score")) and bool(cfg.transaction_cost_config.enabled)
        net = compute_effectiveness_score_for_basis(row, basis="net", config=cfg_eff) if has_net else {}
        selected = net if has_net else gross
        basis = "net" if has_net else "gross"
        for col in score_cols:
            row[f"{col}_gross"] = gross.get(col, np.nan)
            row[f"{col}_net"] = net.get(col, np.nan) if has_net else np.nan
            row[col] = selected.get(col, row.get(col, np.nan))
        row["score_total_basis"] = basis
        row["effectiveness_score_gross"] = gross.get("score_total", np.nan)
        row["effectiveness_score_net"] = net.get("score_total", np.nan) if has_net else np.nan
        row["effectiveness_score"] = selected.get("score_total", row.get("effectiveness_score", np.nan))
        row["effectiveness_tier_gross"] = _score_to_tier_local(row["effectiveness_score_gross"], cfg_eff)
        row["effectiveness_tier_net"] = _score_to_tier_local(row["effectiveness_score_net"], cfg_eff) if has_net else ""
        row["effectiveness_tier"] = _score_to_tier_local(row["effectiveness_score"], cfg_eff)
        records.append(row)
    return pd.DataFrame(records)


def _filter_effective_from_table(
    table: pd.DataFrame,
    *,
    cfg: BatchAnalysisConfig,
    apply_filtering: bool,
) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    work = table.copy()
    if not apply_filtering:
        return work.reset_index(drop=True)
    cfg_eff = (
        cfg.effectiveness_config
        if isinstance(cfg.effectiveness_config, FactorEffectivenessConfig)
        else FactorEffectivenessConfig()
    )
    keep = work.get("stage_b_pass", pd.Series([False] * len(work), index=work.index)).astype(bool) & (
        pd.to_numeric(work.get("effectiveness_score", np.nan), errors="coerce") >= float(cfg_eff.effective_min_score)
    )
    return work.loc[keep].reset_index(drop=True)


def _has_net_returns(frame: pd.DataFrame) -> bool:
    if frame is None or frame.empty or "has_net_pnl" not in frame.columns or "return_net" not in frame.columns:
        return False
    mask = frame["has_net_pnl"].astype(bool)
    return bool(pd.to_numeric(frame.loc[mask, "return_net"], errors="coerce").notna().any())


def _margin_for_portfolio(frame: pd.DataFrame, *, return_col: str, net_only: bool) -> float:
    if frame is None or frame.empty or "turnover" not in frame.columns:
        return np.nan
    work = frame.copy()
    if net_only:
        if "has_net_pnl" not in work.columns:
            return np.nan
        work = work[work["has_net_pnl"].astype(bool)]
    if work.empty or return_col not in work.columns:
        return np.nan
    traded = float(pd.to_numeric(work["turnover"], errors="coerce").sum(skipna=True))
    if traded <= 1e-8:
        return np.nan
    returns = pd.to_numeric(work[return_col], errors="coerce")
    return float(returns.sum(skipna=True) / traded)


def _positive_month_ratio_local(frame: pd.DataFrame, *, return_col: str, period: int, net_only: bool = False) -> float:
    if frame is None or frame.empty or "trade_date" not in frame.columns or return_col not in frame.columns:
        return np.nan
    work = frame.copy()
    if net_only:
        if "has_net_pnl" not in work.columns:
            return np.nan
        work = work[work["has_net_pnl"].astype(bool)]
    if work.empty:
        return np.nan
    dates = pd.to_datetime(work["trade_date"], errors="coerce")
    returns = pd.to_numeric(work[return_col], errors="coerce")
    tmp = pd.DataFrame({"month": dates.dt.to_period("M"), "return": returns}).dropna()
    if tmp.empty:
        return np.nan
    monthly = tmp.groupby("month", sort=True)["return"].apply(lambda s: float((1.0 + s).prod() - 1.0))
    return float((monthly > 0).mean()) if len(monthly) else np.nan


def _score_to_tier_local(score: object, cfg: FactorEffectivenessConfig) -> str:
    value = _to_float_or_nan(score)
    if not np.isfinite(value):
        return "REJECT"
    if value >= cfg.tier_cutoffs.get("S", 80.0):
        return "S"
    if value >= cfg.tier_cutoffs.get("A", 70.0):
        return "A"
    if value >= cfg.tier_cutoffs.get("B", 60.0):
        return "B"
    if value >= cfg.tier_cutoffs.get("C", 50.0):
        return "C"
    return "REJECT"


def build_structured_report_table(
    factor_metrics_df: pd.DataFrame,
    coverage_overall_df: pd.DataFrame | None = None,
    ic_stability_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_columns = [
        "factor",
        "total_obs",
        "non_missing_obs",
        "coverage_rate",
        "missing_rate",
        "ic_mean",
        "ic_std",
        "ir",
        "positive_ic_ratio",
        "t_stat",
        "p_value",
        "long_short_total_return",
        "long_short_annualized_return",
        "long_short_volatility",
        "long_short_sharpe_ratio",
        "long_short_max_drawdown",
        "long_short_fitness_ratio",
        "avg_min_layer_turnover",
        "avg_max_layer_turnover",
        "membership_turnover_worst_layer",
        "membership_turnover_best_layer",
        "best_layer_label",
        "best_layer_direction",
        "best_layer_total_return",
        "best_layer_annualized_return",
        "best_layer_volatility",
        "best_layer_sharpe",
        "best_layer_max_drawdown",
        "best_layer_fitness_ratio",
        "universe_equal_weight_annualized_return",
        "best_minus_universe_annualized_return",
        "benchmark_annualized_return",
        "long_short_excess_annualized_return_vs_benchmark",
        "long_only_excess_annualized_return_vs_benchmark",
        "best_minus_benchmark_annualized_return",
        "best_layer_positive_month_ratio",
        "turnover_long_only_mean",
        "turnover_long_only_median",
        "turnover_long_only_p90",
        "margin_long_only",
        "margin_long_only_bp",
        "margin_long_only_valid",
        "best_layer_margin",
        "margin_long_short",
        "margin_long_short_bp",
        "margin_long_short_valid",
        "monotonicity_mean",
        "monotonicity_std",
        "monotonicity_positive_ratio",
        "yearly_sign_consistency",
        "monthly_sign_consistency",
        "ic_decay_spearman",
        "robust_period_positive_ratio",
        "robust_ic_sign_consistency",
        "robust_ir_median",
        "stage_a_pass",
        "stage_b_pass",
        "passed_hard_filter",
        "score_predictive_power",
        "score_long_only_performance",
        "score_stability",
        "score_tradeability",
        "score_total",
        "effectiveness_score",
        "effectiveness_tier",
        "fail_reasons",
        "warning_reasons",
        "is_effective",
        "ic_skew",
        "ic_kurtosis",
        "obs_count",
        "signal_delay",
        "base_return_col",
        "future_return_col",
        "analysis_period",
        "analysis_exposure_date_rule",
        "holding_window_from_analysis_exposure",
        "effective_raw_signal_return_window",
        "equivalent_exec_return_formula",
        "ret_exec_cc_main_col",
    ]

    if factor_metrics_df is None or factor_metrics_df.empty:
        return pd.DataFrame(columns=target_columns)

    out = factor_metrics_df.copy()
    out["factor"] = out["factor"].astype(str)

    if (
        isinstance(coverage_overall_df, pd.DataFrame)
        and not coverage_overall_df.empty
        and "factor" in coverage_overall_df.columns
    ):
        coverage_cols = [
            c
            for c in [
                "factor",
                "total_obs",
                "non_missing_obs",
                "coverage_rate",
                "missing_rate",
            ]
            if c in coverage_overall_df.columns
        ]
        cov = coverage_overall_df[coverage_cols].copy()
        cov["factor"] = cov["factor"].astype(str)
        out = pd.merge(out, cov, on="factor", how="left")

    if isinstance(ic_stability_df, pd.DataFrame) and not ic_stability_df.empty and "factor" in ic_stability_df.columns:
        stability_cols = [c for c in ["factor", "ic_skew", "ic_kurtosis", "obs_count"] if c in ic_stability_df.columns]
        stab = ic_stability_df[stability_cols].copy()
        stab["factor"] = stab["factor"].astype(str)
        out = pd.merge(out, stab, on="factor", how="left")

    for col in target_columns:
        if col not in out.columns:
            out[col] = np.nan

    out = out[target_columns].copy()
    out = out.sort_values("factor", kind="mergesort").reset_index(drop=True)
    return out


def _validate_required_columns(df_raw: pd.DataFrame, factors: Sequence[str], cfg: BatchAnalysisConfig) -> None:
    required_cols = [
        "trade_date",
        "znz_code",
        cfg.return_col,
        cfg.market_value_column,
    ] + list(factors)
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _analysis_extra_columns(df: pd.DataFrame, cfg: BatchAnalysisConfig) -> list[str]:
    cols: list[str] = []
    if cfg.include_double_sort:
        for col in [cfg.double_sort_control_col, cfg.double_sort_fallback_control_col]:
            if str(col) in df.columns and str(col) not in cols:
                cols.append(str(col))
    if cfg.apply_tradability_constraints:
        for col in [cfg.can_buy_col, cfg.can_sell_col]:
            if str(col) in df.columns and str(col) not in cols:
                cols.append(str(col))
    return cols


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def build_phase_metrics_table(
    *,
    factors: Sequence[str],
    ic_df: pd.DataFrame,
    factor_metrics_df: pd.DataFrame,
    portfolio_pnl_df: pd.DataFrame | None,
    cfg: BatchAnalysisConfig,
    benchmark_pnl_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    factor_list = [str(x) for x in factors if str(x)]
    if not factor_list:
        return pd.DataFrame(), {}
    max_date = _max_trade_date([ic_df, portfolio_pnl_df])
    windows = build_phase_windows(cfg.sample_split_config, max_date=max_date, include_test=True)
    phase_meta = {
        "windows": [window.to_dict() for window in windows],
        "available_phases": [window.key for window in windows],
        "feedback_phase": str(cfg.feedback_phase or "train"),
        "test_default_visible": False,
        "phase_metric_min_obs": int(max(1, cfg.phase_metric_min_obs)),
    }
    full_by_factor = _factor_row_map(factor_metrics_df)
    pnl = portfolio_pnl_df.copy() if isinstance(portfolio_pnl_df, pd.DataFrame) else pd.DataFrame()
    if not pnl.empty and "trade_date" in pnl.columns:
        pnl["trade_date"] = pd.to_datetime(pnl["trade_date"], errors="coerce")
    benchmark_pnl = benchmark_pnl_df.copy() if isinstance(benchmark_pnl_df, pd.DataFrame) else pd.DataFrame()
    if not benchmark_pnl.empty and "trade_date" in benchmark_pnl.columns:
        benchmark_pnl["trade_date"] = pd.to_datetime(benchmark_pnl["trade_date"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for factor in factor_list:
        row: dict[str, Any] = {
            "factor": factor,
            "feedback_phase": str(cfg.feedback_phase or "train"),
        }
        for window in windows:
            prefix = str(window.key)
            ic_values = _phase_ic_values(ic_df, factor=factor, start=window.start, end=window.end)
            row.update(
                _prefix_dict(
                    prefix,
                    _summarize_ic_values(ic_values, min_obs=cfg.phase_metric_min_obs),
                )
            )
            factor_pnl = (
                pnl[pnl.get("factor", pd.Series(dtype=str)).astype(str) == factor].copy()
                if not pnl.empty and "factor" in pnl.columns
                else pd.DataFrame()
            )
            phase_pnl = _slice_by_date(factor_pnl, start=window.start, end=window.end)
            phase_benchmark_pnl = _slice_by_date(benchmark_pnl, start=window.start, end=window.end)
            row.update(
                _prefix_dict(
                    prefix,
                    _summarize_portfolio_phase_metrics_by_basis(
                        phase_pnl,
                        benchmark_pnl=phase_benchmark_pnl,
                        cfg=cfg,
                    ),
                )
            )
            score_gross = _compute_phase_score(row, prefix=prefix, full_row=full_by_factor.get(factor), basis="gross")
            score_net = _compute_phase_score(row, prefix=prefix, full_row=full_by_factor.get(factor), basis="net")
            phase_basis = (
                "net"
                if bool(cfg.transaction_cost_config.enabled) and np.isfinite(_to_float_or_nan(score_net))
                else "gross"
            )
            row[f"{prefix}_score_total_gross"] = score_gross
            row[f"{prefix}_score_total_net"] = score_net
            row[f"{prefix}_score_total_basis"] = phase_basis
            row[f"{prefix}_score_total"] = score_net if phase_basis == "net" else score_gross
        row["feedback_score"] = _first_finite(
            [
                row.get(f"{row['feedback_phase']}_score_total"),
                row.get("train_score_total"),
                row.get("train_score"),
                full_by_factor.get(factor, {}).get("score_total", np.nan),
                full_by_factor.get(factor, {}).get("scoreboard_score", np.nan),
            ]
        )
        row["feedback_score_gross"] = _first_finite(
            [
                row.get(f"{row['feedback_phase']}_score_total_gross"),
                row.get("train_score_total_gross"),
                full_by_factor.get(factor, {}).get("score_total_gross", np.nan),
            ]
        )
        row["feedback_score_net"] = _first_finite(
            [
                row.get(f"{row['feedback_phase']}_score_total_net"),
                row.get("train_score_total_net"),
                full_by_factor.get(factor, {}).get("score_total_net", np.nan),
            ]
        )
        row["feedback_score_basis"] = (
            "net"
            if bool(cfg.transaction_cost_config.enabled)
            and np.isfinite(_to_float_or_nan(row.get("feedback_score_net")))
            else "gross"
        )
        if row["feedback_score_basis"] == "net":
            row["feedback_score"] = row.get("feedback_score_net")
        rows.append(row)
    return pd.DataFrame(rows), phase_meta


def phase_metrics_to_legacy_sample_split_metrics(
    phase_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    if phase_metrics_df is None or phase_metrics_df.empty or "factor" not in phase_metrics_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, source in phase_metrics_df.iterrows():
        row: dict[str, Any] = {"factor": str(source.get("factor"))}
        for phase, legacy in [
            ("train", "train"),
            ("val", "validation"),
            ("test", "oos"),
        ]:
            row[f"{legacy}_obs"] = source.get(f"{phase}_obs", np.nan)
            row[f"{legacy}_ic_mean_mean"] = source.get(f"{phase}_ic_mean", np.nan)
            row[f"{legacy}_score_total_mean"] = source.get(f"{phase}_score_total", np.nan)
            row[f"{legacy}_score"] = source.get(f"{phase}_score_total", np.nan)
        row["validation_decay_ratio"] = _ratio_or_nan(
            _to_float_or_nan(row.get("validation_score")),
            _to_float_or_nan(row.get("train_score")),
        )
        row["oos_decay_ratio"] = _ratio_or_nan(
            _to_float_or_nan(row.get("oos_score")),
            _to_float_or_nan(row.get("train_score")),
        )
        warnings: list[str] = []
        train_score = _to_float_or_nan(row.get("train_score"))
        validation_score = _to_float_or_nan(row.get("validation_score"))
        if not np.isfinite(train_score):
            warnings.append("missing_train_score")
        if not np.isfinite(validation_score):
            warnings.append("missing_validation_score")
        if np.isfinite(train_score) and np.isfinite(validation_score) and train_score * validation_score < 0:
            warnings.append("validation_score_sign_flip")
        row["split_pass"] = len(warnings) == 0
        row["split_warning_reasons"] = "; ".join(warnings)
        rows.append(row)
    return pd.DataFrame(rows)


def _phase_ic_values(ic_df: pd.DataFrame, factor: str, start: str, end: str | None) -> pd.Series:
    if ic_df is None or ic_df.empty or "trade_date" not in ic_df.columns:
        return pd.Series(dtype=float)
    col = f"{factor}_ic"
    if col not in ic_df.columns:
        return pd.Series(dtype=float)
    work = _slice_by_date(ic_df[["trade_date", col]].copy(), start=start, end=end)
    return pd.to_numeric(work[col], errors="coerce").dropna()


def _summarize_ic_values(values: pd.Series, min_obs: int) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    obs = int(len(clean))
    if obs < max(1, int(min_obs)):
        return {
            "obs": obs,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ir": np.nan,
            "positive_ic_ratio": np.nan,
        }
    ic_mean = float(clean.mean())
    ic_std = float(clean.std(ddof=1)) if obs > 1 else np.nan
    ir = float(ic_mean / ic_std) if np.isfinite(ic_std) and abs(ic_std) > 1e-12 else np.nan
    return {
        "obs": obs,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ir,
        "positive_ic_ratio": float((clean > 0).mean()),
    }


def _summarize_portfolio_phase_metrics(
    phase_pnl: pd.DataFrame,
    *,
    benchmark_pnl: pd.DataFrame | None,
    cfg: BatchAnalysisConfig,
    return_col: str = "return",
    net_only: bool = False,
) -> dict[str, Any]:
    if phase_pnl is None or phase_pnl.empty or "portfolio" not in phase_pnl.columns:
        return {}
    source_pnl = phase_pnl.copy()
    if net_only:
        if "has_net_pnl" not in source_pnl.columns:
            return {}
        source_pnl = source_pnl[source_pnl["has_net_pnl"].astype(bool)].copy()
    if source_pnl.empty or return_col not in source_pnl.columns:
        return {}
    out: dict[str, Any] = {}
    benchmark_annual = _annualized_return_for_frame(benchmark_pnl, period=cfg.period)
    if benchmark_annual is not None:
        out["benchmark_annualized_return"] = benchmark_annual
    for portfolio, output_prefix in [
        ("long_short", "long_short"),
        ("long_only", "long_only"),
    ]:
        part = source_pnl[source_pnl["portfolio"].astype(str) == portfolio].copy()
        returns = pd.to_numeric(part.get(return_col, pd.Series(dtype=float)), errors="coerce").dropna()
        if not returns.empty:
            metrics = calculate_risk_metrics(returns, period=cfg.period)
            out[f"{output_prefix}_total_return"] = metrics.get("total_return", np.nan)
            out[f"{output_prefix}_annualized_return"] = metrics.get("annualized_return", np.nan)
            annual = _to_float_or_nan(metrics.get("annualized_return", np.nan))
            out[f"{output_prefix}_excess_annualized_return_vs_benchmark"] = (
                annual - benchmark_annual if benchmark_annual is not None and np.isfinite(annual) else np.nan
            )
            out[f"{output_prefix}_volatility"] = metrics.get("volatility", np.nan)
            out[f"{output_prefix}_sharpe_ratio"] = metrics.get("sharpe_ratio", np.nan)
            out[f"{output_prefix}_max_drawdown"] = metrics.get("max_drawdown", np.nan)
            out[f"{output_prefix}_fitness_ratio"] = metrics.get("fitness_ratio", np.nan)
            if portfolio == "long_only":
                out["best_minus_benchmark_annualized_return"] = out[
                    f"{output_prefix}_excess_annualized_return_vs_benchmark"
                ]
        if portfolio == "long_short" and not part.empty:
            turnover = pd.to_numeric(part.get("turnover", pd.Series(dtype=float)), errors="coerce")
            out["turnover_long_short_mean"] = float(turnover.mean(skipna=True)) if turnover.notna().any() else np.nan
            traded = float(turnover.sum(skipna=True)) if turnover.notna().any() else 0.0
            returns_all = pd.to_numeric(part.get(return_col, pd.Series(dtype=float)), errors="coerce")
            margin = float(returns_all.sum(skipna=True) / traded) if traded > 1e-8 else np.nan
            out["margin_long_short"] = margin
            out["margin_long_short_bp"] = margin * 10000.0 if np.isfinite(margin) else np.nan
        if portfolio == "long_only" and not part.empty:
            turnover = pd.to_numeric(part.get("turnover", pd.Series(dtype=float)), errors="coerce")
            out["turnover_long_only_mean"] = float(turnover.mean(skipna=True)) if turnover.notna().any() else np.nan
            traded = float(turnover.sum(skipna=True)) if turnover.notna().any() else 0.0
            returns_all = pd.to_numeric(part.get(return_col, pd.Series(dtype=float)), errors="coerce")
            margin = float(returns_all.sum(skipna=True) / traded) if traded > 1e-8 else np.nan
            out["margin_long_only"] = margin
            out["margin_long_only_bp"] = margin * 10000.0 if np.isfinite(margin) else np.nan
    return out


def _summarize_portfolio_phase_metrics_by_basis(
    phase_pnl: pd.DataFrame,
    *,
    benchmark_pnl: pd.DataFrame | None,
    cfg: BatchAnalysisConfig,
) -> dict[str, Any]:
    gross = _summarize_portfolio_phase_metrics(
        phase_pnl,
        benchmark_pnl=benchmark_pnl,
        cfg=cfg,
        return_col="return_gross"
        if isinstance(phase_pnl, pd.DataFrame) and "return_gross" in phase_pnl.columns
        else "return",
    )
    net = _summarize_portfolio_phase_metrics(
        phase_pnl,
        benchmark_pnl=benchmark_pnl,
        cfg=cfg,
        return_col="return_net",
        net_only=True,
    )
    selected_basis = "net" if bool(cfg.transaction_cost_config.enabled) and bool(net) else "gross"
    out: dict[str, Any] = {}
    for key, value in gross.items():
        out[f"{key}_gross"] = value
    for key, value in net.items():
        out[f"{key}_net"] = value
    selected = net if selected_basis == "net" else gross
    if selected_basis == "net":
        selected = {**gross, **net}
    out.update(selected)
    out["score_basis"] = selected_basis
    return out


def _build_benchmark_pnl_table(
    *,
    benchmark_returns: Sequence[dict[str, Any]],
    benchmark_enabled: bool,
) -> pd.DataFrame:
    if not benchmark_enabled or not benchmark_returns:
        return pd.DataFrame()
    bench = pd.DataFrame(list(benchmark_returns))
    if bench.empty or "trade_date" not in bench.columns or "return" not in bench.columns:
        return pd.DataFrame()
    bench = bench[["trade_date", "return"]].copy()
    bench["trade_date"] = pd.to_datetime(bench["trade_date"], errors="coerce")
    bench["return"] = pd.to_numeric(bench["return"], errors="coerce")
    bench = bench.dropna(subset=["trade_date", "return"]).sort_values("trade_date", kind="mergesort")
    if bench.empty:
        return pd.DataFrame()
    bench["cum_return"] = (1.0 + bench["return"]).cumprod() - 1.0
    bench["portfolio"] = "benchmark"
    bench["holding_count"] = np.nan
    bench["turnover"] = np.nan
    bench["blocked_buy_ratio"] = np.nan
    bench["blocked_sell_ratio"] = np.nan
    bench["tradability_return_drag"] = np.nan
    return bench.reset_index(drop=True)


def _add_benchmark_relative_metrics(
    factor_metrics_df: pd.DataFrame,
    *,
    portfolio_pnl_df: pd.DataFrame | None,
    benchmark_pnl_df: pd.DataFrame | None,
    period: int,
) -> pd.DataFrame:
    if factor_metrics_df is None or factor_metrics_df.empty:
        return factor_metrics_df
    out = factor_metrics_df.copy()
    benchmark_annual = _annualized_return_for_frame(benchmark_pnl_df, period=period)
    if benchmark_annual is None:
        for col in [
            "benchmark_annualized_return",
            "long_short_excess_annualized_return_vs_benchmark",
            "long_only_excess_annualized_return_vs_benchmark",
            "best_minus_benchmark_annualized_return",
        ]:
            if col not in out.columns:
                out[col] = np.nan
        return out
    out["benchmark_annualized_return"] = benchmark_annual
    if "long_short_annualized_return" in out.columns:
        out["long_short_excess_annualized_return_vs_benchmark"] = (
            pd.to_numeric(out["long_short_annualized_return"], errors="coerce") - benchmark_annual
        )
    if "best_layer_annualized_return" in out.columns:
        out["best_minus_benchmark_annualized_return"] = (
            pd.to_numeric(out["best_layer_annualized_return"], errors="coerce") - benchmark_annual
        )
    if (
        portfolio_pnl_df is not None
        and not portfolio_pnl_df.empty
        and {"factor", "portfolio", "return"}.issubset(portfolio_pnl_df.columns)
    ):
        long_only_by_factor: dict[str, float] = {}
        for factor, group in portfolio_pnl_df[portfolio_pnl_df["portfolio"].astype(str) == "long_only"].groupby(
            "factor", sort=False
        ):
            annual = _annualized_return_for_frame(group, period=period)
            if annual is not None:
                long_only_by_factor[str(factor)] = annual - benchmark_annual
        if long_only_by_factor:
            out["long_only_excess_annualized_return_vs_benchmark"] = out["factor"].astype(str).map(long_only_by_factor)
    return out


def _annualized_return_for_frame(frame: pd.DataFrame | None, *, period: int) -> float | None:
    if frame is None or frame.empty or "return" not in frame.columns:
        return None
    returns = pd.to_numeric(frame.get("return", pd.Series(dtype=float)), errors="coerce").dropna()
    if returns.empty:
        return None
    value = _to_float_or_nan(calculate_risk_metrics(returns, period=period).get("annualized_return"))
    return float(value) if np.isfinite(value) else None


def _compute_phase_score(
    row: dict[str, Any],
    prefix: str,
    full_row: dict[str, Any] | None,
    basis: str = "gross",
) -> float:
    if not np.isfinite(_to_float_or_nan(row.get(f"{prefix}_ic_mean"))):
        return np.nan
    full = dict(full_row or {})
    suffix = "net" if str(basis or "").lower() == "net" else "gross"
    if suffix == "net":
        has_phase_net = any(str(k).startswith(f"{prefix}_") and str(k).endswith("_net") for k in row)
        has_full_net = bool(full.get("has_net_score"))
        if not has_phase_net and not has_full_net:
            return np.nan

    def phase_metric(name: str, fallback: str | None = None) -> Any:
        values = [
            row.get(f"{prefix}_{name}_{suffix}"),
            full.get(f"{fallback or name}_{suffix}"),
        ]
        if suffix != "net":
            values.extend([row.get(f"{prefix}_{name}"), full.get(fallback or name)])
        return _first_finite(values)

    score_row = {
        "ic_mean": row.get(f"{prefix}_ic_mean"),
        "ir": row.get(f"{prefix}_ir"),
        "positive_ic_ratio": row.get(f"{prefix}_positive_ic_ratio"),
        "long_short_total_return": phase_metric("long_short_total_return"),
        "long_short_annualized_return": phase_metric("long_short_annualized_return"),
        "long_short_sharpe_ratio": phase_metric("long_short_sharpe_ratio"),
        "long_short_max_drawdown": phase_metric("long_short_max_drawdown"),
        "long_short_fitness_ratio": phase_metric("long_short_fitness_ratio"),
        "best_layer_total_return": phase_metric("long_only_total_return", "best_layer_total_return"),
        "best_layer_annualized_return": phase_metric("long_only_annualized_return", "best_layer_annualized_return"),
        "best_layer_sharpe": phase_metric("long_only_sharpe_ratio", "best_layer_sharpe"),
        "best_layer_max_drawdown": phase_metric("long_only_max_drawdown", "best_layer_max_drawdown"),
        "best_layer_fitness_ratio": phase_metric("long_only_fitness_ratio", "best_layer_fitness_ratio"),
        "best_minus_universe_annualized_return": _first_finite(
            [
                row.get(f"{prefix}_best_minus_benchmark_annualized_return_{suffix}"),
                row.get(f"{prefix}_best_minus_benchmark_annualized_return"),
                full.get(f"best_minus_benchmark_annualized_return_{suffix}"),
                full.get(f"best_minus_universe_annualized_return_{suffix}"),
                full.get("best_minus_benchmark_annualized_return"),
                full.get("best_minus_universe_annualized_return"),
            ]
        ),
        "turnover_long_only_mean": phase_metric("turnover_long_only_mean", "turnover_long_only_mean"),
        "margin_long_only": phase_metric("margin_long_only", "margin_long_only"),
        "sign_adjusted_monotonicity": full.get("sign_adjusted_monotonicity", full.get("monotonicity_mean")),
        "yearly_sign_consistency": full.get("yearly_sign_consistency"),
        "monthly_sign_consistency": full.get("monthly_sign_consistency"),
    }
    return float(compute_effectiveness_score_parts(score_row).get("score_total", np.nan))


def _slice_by_date(frame: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    if frame is None or frame.empty or "trade_date" not in frame.columns:
        return pd.DataFrame() if frame is None else frame.iloc[0:0].copy()
    work = frame.copy()
    dates = pd.to_datetime(work["trade_date"], errors="coerce")
    mask = dates >= pd.Timestamp(start)
    if end:
        mask &= dates <= pd.Timestamp(end)
    return work[mask].copy()


def _prefix_dict(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _factor_row_map(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if frame is None or frame.empty or "factor" not in frame.columns:
        return {}
    return {str(row.get("factor")): row for row in frame.to_dict(orient="records")}


def _max_trade_date(frames: Sequence[pd.DataFrame | None]) -> str | None:
    max_ts: pd.Timestamp | None = None
    for frame in frames:
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty or "trade_date" not in frame.columns:
            continue
        values = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
        if values.empty:
            continue
        candidate = pd.Timestamp(values.max())
        max_ts = candidate if max_ts is None or candidate > max_ts else max_ts
    return max_ts.strftime("%Y-%m-%d") if max_ts is not None else None


def _first_finite(values: Sequence[Any]) -> float:
    for value in values:
        number = _to_float_or_nan(value)
        if np.isfinite(number):
            return number
    return np.nan


def _to_float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _ratio_or_nan(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _sample_split_metrics_from_ic(
    ic_df: pd.DataFrame, factors: Sequence[str], cfg: BatchAnalysisConfig
) -> pd.DataFrame:
    if ic_df is None or ic_df.empty or "trade_date" not in ic_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for factor in factors:
        col = f"{factor}_ic"
        if col not in ic_df.columns:
            continue
        tmp = ic_df[["trade_date", col]].copy()
        tmp["factor"] = str(factor)
        tmp["ic_mean"] = pd.to_numeric(tmp[col], errors="coerce")
        tmp["score_total"] = tmp["ic_mean"].abs() * 100.0
        rows.extend(tmp[["trade_date", "factor", "ic_mean", "score_total"]].to_dict("records"))
    if not rows:
        return pd.DataFrame()
    split_df = assign_sample_split(pd.DataFrame(rows), date_col="trade_date", config=cfg.sample_split_config)
    return summarize_split_metrics(split_df, factor_col="factor", metric_cols=["score_total", "ic_mean"])


def _merge_optional_factor_metrics(factor_metrics_df: pd.DataFrame, frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if factor_metrics_df is None or factor_metrics_df.empty or "factor" not in factor_metrics_df.columns:
        return pd.DataFrame() if factor_metrics_df is None else factor_metrics_df
    out = factor_metrics_df.copy()
    out["factor"] = out["factor"].astype(str)
    for frame in frames:
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty or "factor" not in frame.columns:
            continue
        extra = frame.copy()
        extra["factor"] = extra["factor"].astype(str)
        cols = ["factor"] + [c for c in extra.columns if c != "factor" and c not in out.columns]
        if len(cols) <= 1:
            continue
        out = pd.merge(out, extra[cols], on="factor", how="left")
    return out


def _merge_effectiveness_tables(base_df: pd.DataFrame, update_df: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return update_df.copy() if isinstance(update_df, pd.DataFrame) else pd.DataFrame()
    if update_df is None or update_df.empty:
        return base_df.copy()
    if "factor" not in base_df.columns or "factor" not in update_df.columns:
        return base_df.copy()

    merged = base_df.copy()
    merged["factor"] = merged["factor"].astype(str)
    upd = update_df.copy()
    upd["factor"] = upd["factor"].astype(str)
    merged = merged.set_index("factor")
    upd = upd.set_index("factor")
    for col in upd.columns:
        merged[col] = merged[col] if col in merged.columns else np.nan
        merged.loc[upd.index, col] = upd[col]
    return merged.reset_index()


def _legacy_effective_columns() -> list[str]:
    return [
        "factor",
        "ic_mean",
        "ic_std",
        "ir",
        "positive_ic_ratio",
        "t_stat",
        "p_value",
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe_ratio",
        "max_drawdown",
        "fitness_ratio",
    ]


# ---------------------------------------------------------------------------
# Recall Validation for light_then_full mode
# ---------------------------------------------------------------------------


def validate_light_filter_recall(
    df_raw: pd.DataFrame,
    all_factors: list[str],
    light_survivors: list[str],
    config: BatchAnalysisConfig,
    validation_config: RecallValidationConfig | None = None,
) -> dict[str, Any]:
    """验证轻量评估的召回率：从被过滤因子中随机抽样，跑完整评估。

    返回:
        enabled, rejected_count, sample_size, effective_in_sample,
        false_negative_rate, estimated_recall
    """
    import random as _random

    cfg = validation_config or RecallValidationConfig()
    if not cfg.enabled:
        return {"enabled": False}

    rejected = [f for f in all_factors if f not in set(light_survivors)]
    if not rejected:
        return {"enabled": True, "rejected_count": 0, "recall": 1.0}

    sample_size = min(
        cfg.max_sample_size,
        max(cfg.min_sample_size, int(len(rejected) * cfg.sample_ratio)),
    )
    sample_size = min(sample_size, len(rejected))
    sampled = _random.sample(rejected, sample_size)

    full_results = run_factor_analysis_batch_full(
        df_raw=df_raw,
        base_outputs={},
        full_factor_cols=sampled,
        config=config,
    )

    effective_in_sample = 0
    eff_df = full_results.get("effective_factors_df")
    if isinstance(eff_df, pd.DataFrame) and not eff_df.empty:
        effective_in_sample = len(eff_df)

    false_negative_rate = effective_in_sample / sample_size if sample_size > 0 else 0.0

    return {
        "enabled": True,
        "rejected_count": len(rejected),
        "sample_size": sample_size,
        "sampled_factors": sampled,
        "effective_in_sample": effective_in_sample,
        "false_negative_rate": false_negative_rate,
        "estimated_recall": 1.0 - false_negative_rate,
    }
