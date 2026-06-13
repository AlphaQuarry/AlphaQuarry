from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .utils import cumulative_returns


@dataclass(frozen=True)
class FactorEffectivenessConfig:
    coverage_hard_reject_min: float = 0.60
    coverage_stage_b_min: float = 0.60
    coverage_by_date_p10_stage_b_min: float = 0.50
    obs_hard_reject_min: int = 250
    obs_stage_b_min: int = 250

    ic_abs_stage_a_min: float = 0.015
    ic_abs_stage_b_min: float = 0.020
    ir_abs_stage_a_min: float = 0.25
    ir_abs_stage_b_min: float = 0.30
    p_value_stage_a_max: float = 0.10
    p_value_stage_b_max: float = 0.10
    positive_ic_ratio_stage_a_pos_min: float = 0.55
    positive_ic_ratio_stage_a_neg_max: float = 0.45
    positive_ic_ratio_stage_b_pos_min: float = 0.55
    positive_ic_ratio_stage_b_neg_max: float = 0.45

    sign_adjusted_monotonicity_stage_a_min: float = 0.10
    sign_adjusted_monotonicity_stage_b_min: float = 0.15
    monotonicity_ratio_pos_min: float = 0.54
    monotonicity_ratio_neg_max: float = 0.46

    sharpe_stage_a_min: float = 0.40
    sharpe_stage_b_min: float = 0.50
    fitness_stage_a_min: float = 0.40
    fitness_stage_b_min: float = 0.50
    max_drawdown_stage_a_max: float = 0.55
    max_drawdown_stage_b_max: float = 0.50

    turnover_stage_a_max: float = 0.80
    turnover_stage_b_max: float = 0.75
    stale_turnover_warning_max: float = 0.01

    yearly_sign_consistency_min: float = 0.50
    monthly_sign_consistency_min: float = 0.55
    robust_positive_ratio_min: float = 0.55
    robust_ic_sign_consistency_min: float = 0.55
    robust_ir_median_abs_min: float = 0.30

    skew_warning_abs: float = 1.5
    kurt_warning_min: float = 8.0
    obs_warning_min: int = 80

    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "predictive_power": 30.0,
            "long_only_performance": 45.0,
            "time_stability": 5.0,
            "tradeability": 20.0,
        }
    )
    effective_min_score: float = 50.0
    best_layer_annualized_return_min: float = 0.05
    best_minus_universe_annualized_return_min: float = 0.03
    best_layer_sharpe_min: float = 0.50
    best_layer_max_drawdown_max: float = 0.50
    best_layer_margin_min: float = 0.0
    tier_cutoffs: dict[str, float] = field(
        default_factory=lambda: {
            "S": 80.0,
            "A": 70.0,
            "B": 60.0,
            "C": 50.0,
        }
    )

    require_time_stability_when_available: bool = False
    require_robustness_when_available: bool = False
    ic_decay_spearman_full_score_abs: float = 0.40
    ic_decay_spearman_score_phase: str = "train"


def evaluate_factor_effectiveness(
    summary_df: pd.DataFrame,
    long_short_metrics: dict[str, dict[str, Any]],
    layer_results: dict[str, pd.DataFrame] | None = None,
    best_layer_metrics_df: pd.DataFrame | None = None,
    long_only_turnover_summary_df: pd.DataFrame | None = None,
    margin_metrics_df: pd.DataFrame | None = None,
    coverage_overall_df: pd.DataFrame | None = None,
    coverage_by_date_df: pd.DataFrame | None = None,
    ic_stability_df: pd.DataFrame | None = None,
    monotonicity_summary_df: pd.DataFrame | None = None,
    turnover_results: dict[str, pd.DataFrame] | None = None,
    ic_yearly_df: pd.DataFrame | None = None,
    ic_monthly_df: pd.DataFrame | None = None,
    ic_decay_df: pd.DataFrame | None = None,
    period_comparison_df: pd.DataFrame | None = None,
    apply_filtering: bool = True,
    config: FactorEffectivenessConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Comprehensive factor effectiveness evaluation with staged quality gates.

    Evaluates factors across multiple dimensions:
    - Predictive power (IC, IR, t-stat, p-value)
    - Long-only performance (Sharpe, max drawdown, fitness)
    - Time stability (IC sign consistency, monotonicity)
    - Tradeability (coverage, turnover, margins)

    Applies two-stage quality gates:
    - Stage A: Relaxed thresholds for initial screening
    - Stage B: Strict thresholds for final selection

    Args:
        summary_df: DataFrame with IC statistics per factor.
        long_short_metrics: Dictionary of long-short portfolio metrics per factor.
        layer_results: Dictionary of layer analysis results per factor.
        best_layer_metrics_df: DataFrame with best layer performance metrics.
        long_only_turnover_summary_df: DataFrame with long-only turnover metrics.
        margin_metrics_df: DataFrame with margin/bid-ask metrics.
        coverage_overall_df: DataFrame with overall coverage statistics.
        coverage_by_date_df: DataFrame with daily coverage statistics.
        ic_stability_df: DataFrame with IC stability metrics.
        monotonicity_summary_df: DataFrame with layer monotonicity metrics.
        turnover_results: Dictionary of turnover analysis results per factor.
        ic_yearly_df: DataFrame with yearly IC sign consistency.
        ic_monthly_df: DataFrame with monthly IC sign consistency.
        ic_decay_df: DataFrame with IC decay analysis.
        period_comparison_df: DataFrame with period robustness comparison.
        apply_filtering: Whether to filter out ineffective factors. Defaults to True.
        config: FactorEffectivenessConfig with threshold settings.

    Returns:
        Dictionary with two DataFrames:
        - 'factor_effectiveness_table': Full evaluation results for all factors.
        - 'effective_factors_df': Filtered effective factors (if apply_filtering=True).

    Example:
        >>> result = evaluate_factor_effectiveness(summary_df, long_short_metrics)
        >>> effective = result['effective_factors_df']
        >>> print(effective[['factor', 'effectiveness_score', 'effectiveness_tier']])
    """
    cfg = config or FactorEffectivenessConfig()
    required_summary_cols = {
        "factor",
        "ic_mean",
        "ic_std",
        "ir",
        "positive_ic_ratio",
        "t_stat",
        "p_value",
    }
    if summary_df is None or summary_df.empty or not required_summary_cols.issubset(summary_df.columns):
        empty = pd.DataFrame(columns=_legacy_effective_columns())
        return {
            "factor_effectiveness_table": empty.copy(),
            "effective_factors_df": empty,
        }

    summary_df = summary_df.copy()
    summary_df["factor"] = summary_df["factor"].astype(str)

    if monotonicity_summary_df is None and layer_results:
        try:
            from .diagnostics import calculate_layer_monotonicity

            monotonicity_summary_df = calculate_layer_monotonicity(layer_results).get("summary")
        except Exception:
            monotonicity_summary_df = None

    factors = list(dict.fromkeys(summary_df["factor"].astype(str).tolist()))
    for factor in long_short_metrics.keys():
        name = str(factor)
        if name not in factors:
            factors.append(name)

    cov_map = _to_factor_map(coverage_overall_df)
    cov_p10_map = _coverage_by_date_p10_map(coverage_by_date_df)
    stab_map = _to_factor_map(ic_stability_df)
    mono_map = _to_factor_map(monotonicity_summary_df)
    turnover_map = _turnover_summary_map(turnover_results or {})
    best_layer_map = _to_factor_map(best_layer_metrics_df)
    long_only_turnover_map = _to_factor_map(long_only_turnover_summary_df)
    margin_map = _to_factor_map(margin_metrics_df)
    yearly_consistency_map = _ic_sign_consistency_map(ic_yearly_df, summary_df=summary_df)
    monthly_consistency_map = _ic_sign_consistency_map(ic_monthly_df, summary_df=summary_df)
    decay_map = _ic_decay_spearman_map(ic_decay_df, preferred_phase=cfg.ic_decay_spearman_score_phase)
    robust_map = _robustness_summary_map(period_comparison_df, summary_df=summary_df)

    rows: list[dict[str, Any]] = []
    for factor in factors:
        summary_match = summary_df[summary_df["factor"] == factor]
        if summary_match.empty:
            continue
        ic_row = summary_match.iloc[0]
        ls_row = long_short_metrics.get(factor, {})

        row: dict[str, Any] = {
            "factor": factor,
            "ic_mean": _to_float(ic_row.get("ic_mean")),
            "ic_std": _to_float(ic_row.get("ic_std")),
            "ir": _to_float(ic_row.get("ir")),
            "positive_ic_ratio": _to_float(ic_row.get("positive_ic_ratio")),
            "t_stat": _to_float(ic_row.get("t_stat")),
            "p_value": _to_float(ic_row.get("p_value")),
            "total_return": _to_float(ls_row.get("total_return")),
            "annualized_return": _to_float(ls_row.get("annualized_return")),
            "volatility": _to_float(ls_row.get("volatility")),
            "sharpe_ratio": _to_float(ls_row.get("sharpe_ratio")),
            "max_drawdown": _to_float(ls_row.get("max_drawdown")),
            "fitness_ratio": _to_float(ls_row.get("fitness_ratio")),
        }

        cov = cov_map.get(factor, {})
        row["coverage_rate"] = _to_float(cov.get("coverage_rate"))
        row["missing_rate"] = _to_float(cov.get("missing_rate"))
        row["total_obs"] = _to_float(cov.get("total_obs"))
        row["non_missing_obs"] = _to_float(cov.get("non_missing_obs"))
        row["coverage_rate_by_date_p10"] = _to_float(cov_p10_map.get(factor))

        stab = stab_map.get(factor, {})
        row["ic_skew"] = _to_float(stab.get("ic_skew"))
        row["ic_kurtosis"] = _to_float(stab.get("ic_kurtosis"))
        row["obs_count"] = _to_float(stab.get("obs_count"))

        mono = mono_map.get(factor, {})
        row["monotonicity_mean"] = _to_float(mono.get("monotonicity_mean"))
        row["monotonicity_std"] = _to_float(mono.get("monotonicity_std"))
        row["monotonicity_positive_ratio"] = _to_float(mono.get("monotonicity_positive_ratio"))

        tr = turnover_map.get(factor, {})
        row["avg_min_layer_turnover"] = _to_float(tr.get("avg_min_layer_turnover"))
        row["avg_max_layer_turnover"] = _to_float(tr.get("avg_max_layer_turnover"))
        row["membership_turnover_worst_layer"] = _to_float(
            tr.get("membership_turnover_worst_layer", tr.get("avg_min_layer_turnover"))
        )
        row["membership_turnover_best_layer"] = _to_float(
            tr.get("membership_turnover_best_layer", tr.get("avg_max_layer_turnover"))
        )

        best = best_layer_map.get(factor, {})
        for col in [
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
            "best_layer_positive_month_ratio",
        ]:
            row[col] = best.get(col, np.nan)

        lo_to = long_only_turnover_map.get(factor, {})
        for col in [
            "turnover_long_only_mean",
            "turnover_long_only_median",
            "turnover_long_only_p90",
            "portfolio_return_long_only_sum",
        ]:
            row[col] = _to_float(lo_to.get(col))

        margin = margin_map.get(factor, {})
        for col in [
            "margin_long_only",
            "margin_long_only_bp",
            "best_layer_margin",
            "margin_long_short",
            "margin_long_short_bp",
        ]:
            row[col] = _to_float(margin.get(col))
        row["margin_long_only_valid"] = bool(margin.get("margin_long_only_valid", _is_finite(row["margin_long_only"])))
        row["margin_long_short_valid"] = bool(margin.get("margin_long_short_valid", False))

        row["yearly_sign_consistency"] = _to_float(yearly_consistency_map.get(factor))
        row["monthly_sign_consistency"] = _to_float(monthly_consistency_map.get(factor))
        row["ic_decay_spearman"] = _to_float(decay_map.get(factor))

        robust = robust_map.get(factor, {})
        row["robust_period_positive_ratio"] = _to_float(robust.get("robust_period_positive_ratio"))
        row["robust_ic_sign_consistency"] = _to_float(robust.get("robust_ic_sign_consistency"))
        row["robust_ir_median"] = _to_float(robust.get("robust_ir_median"))

        sign = _sign(row["ic_mean"])
        row["sign_adjusted_monotonicity"] = (
            sign * row["monotonicity_mean"] if sign != 0 and _is_finite(row["monotonicity_mean"]) else np.nan
        )

        stage_a_pass, stage_b_pass = _stage_passes(row, cfg)
        row["stage_a_pass"] = bool(stage_a_pass)
        row["stage_b_pass"] = bool(stage_b_pass)
        row["passed_hard_filter"] = bool(stage_b_pass)
        score_parts = compute_effectiveness_score_parts(row, cfg)
        row.update(score_parts)
        row["score_total"] = float(score_parts.get("score_total", 0.0))
        row["effectiveness_score"] = row["score_total"]
        row["effectiveness_tier"] = _score_to_tier(row["effectiveness_score"], cfg)

        fail_reasons = build_fail_reasons(row, cfg)
        warning_reasons = build_warning_reasons(row, cfg)
        row["fail_reasons"] = "; ".join(fail_reasons)
        row["warning_reasons"] = "; ".join(warning_reasons)
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        empty = pd.DataFrame(columns=_legacy_effective_columns())
        return {
            "factor_effectiveness_table": empty.copy(),
            "effective_factors_df": empty,
        }

    if apply_filtering:
        keep_mask = table["stage_b_pass"].astype(bool) & (
            pd.to_numeric(table["effectiveness_score"], errors="coerce") >= float(cfg.effective_min_score)
        )
    else:
        keep_mask = pd.Series([True] * len(table), index=table.index)

    effective = table.loc[keep_mask].copy()
    legacy_cols = _legacy_effective_columns()
    for col in legacy_cols:
        if col not in effective.columns:
            effective[col] = np.nan
    effective = effective[legacy_cols + [c for c in effective.columns if c not in legacy_cols]]
    return {
        "factor_effectiveness_table": table.reset_index(drop=True),
        "effective_factors_df": effective.reset_index(drop=True),
    }


def filter_effective_factors(
    summary_df: pd.DataFrame,
    long_short_metrics: dict,
    layer_results: dict,
    apply_filtering: bool = True,
    layer_significance_mode: str = "legacy_q_10",
    coverage_overall_df: pd.DataFrame | None = None,
    coverage_by_date_df: pd.DataFrame | None = None,
    ic_stability_df: pd.DataFrame | None = None,
    monotonicity_summary_df: pd.DataFrame | None = None,
    turnover_results: dict[str, pd.DataFrame] | None = None,
    ic_yearly_df: pd.DataFrame | None = None,
    ic_monthly_df: pd.DataFrame | None = None,
    period_comparison_df: pd.DataFrame | None = None,
    effectiveness_config: FactorEffectivenessConfig | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.

    Notes:
    - Keeps original function name and return type.
    - Internally upgrades to staged effectiveness evaluation with score/tier.
    - `layer_significance_mode` is preserved for API compatibility.
    """
    _ = layer_significance_mode
    result = evaluate_factor_effectiveness(
        summary_df=summary_df,
        long_short_metrics=long_short_metrics,
        layer_results=layer_results,
        coverage_overall_df=coverage_overall_df,
        coverage_by_date_df=coverage_by_date_df,
        ic_stability_df=ic_stability_df,
        monotonicity_summary_df=monotonicity_summary_df,
        turnover_results=turnover_results,
        ic_yearly_df=ic_yearly_df,
        ic_monthly_df=ic_monthly_df,
        ic_decay_df=None,
        period_comparison_df=period_comparison_df,
        apply_filtering=apply_filtering,
        config=effectiveness_config,
    )
    return result["effective_factors_df"]


def build_fail_reasons(row: dict[str, Any], config: FactorEffectivenessConfig | None = None) -> list[str]:
    cfg = config or FactorEffectivenessConfig()
    reasons: list[str] = []

    coverage = _to_float(row.get("coverage_rate"))
    obs_count = _to_float(row.get("obs_count"))
    cov_p10 = _to_float(row.get("coverage_rate_by_date_p10"))

    if _is_finite(coverage) and coverage < cfg.coverage_hard_reject_min:
        reasons.append(f"coverage_rate<{cfg.coverage_hard_reject_min:.2f}")
    if _is_finite(obs_count) and obs_count < cfg.obs_hard_reject_min:
        reasons.append(f"obs_count<{int(cfg.obs_hard_reject_min)}")
    if _is_finite(coverage) and coverage < cfg.coverage_stage_b_min:
        reasons.append(f"coverage_rate<{cfg.coverage_stage_b_min:.2f}")
    if _is_finite(cov_p10) and cov_p10 < cfg.coverage_by_date_p10_stage_b_min:
        reasons.append(f"coverage_rate_by_date_p10<{cfg.coverage_by_date_p10_stage_b_min:.2f}")
    if _is_finite(obs_count) and obs_count < cfg.obs_stage_b_min:
        reasons.append(f"obs_count<{int(cfg.obs_stage_b_min)}")

    ic_mean = _to_float(row.get("ic_mean"))
    ir = _to_float(row.get("ir"))
    p_value = _to_float(row.get("p_value"))
    if _is_finite(ic_mean) and abs(ic_mean) < cfg.ic_abs_stage_b_min:
        reasons.append(f"|ic_mean|<{cfg.ic_abs_stage_b_min:.3f}")
    if _is_finite(ir) and abs(ir) < cfg.ir_abs_stage_b_min:
        reasons.append(f"|ir|<{cfg.ir_abs_stage_b_min:.2f}")
    if _is_finite(p_value) and p_value > cfg.p_value_stage_b_max:
        reasons.append(f"p_value>{cfg.p_value_stage_b_max:.2f}")

    best_ann = _to_float(row.get("best_layer_annualized_return"))
    best_excess = _to_float(row.get("best_minus_universe_annualized_return"))
    best_sharpe = _to_float(row.get("best_layer_sharpe"))
    best_dd = _to_float(row.get("best_layer_max_drawdown"))
    turnover = _to_float(row.get("turnover_long_only_mean"))
    margin = _to_float(row.get("best_layer_margin", row.get("margin_long_only")))
    if _is_finite(best_ann) and best_ann <= cfg.best_layer_annualized_return_min:
        reasons.append(f"best_layer_annualized_return<={cfg.best_layer_annualized_return_min:.2f}")
    if _is_finite(best_excess) and best_excess <= cfg.best_minus_universe_annualized_return_min:
        reasons.append(f"best_minus_universe_annualized_return<={cfg.best_minus_universe_annualized_return_min:.2f}")
    if _is_finite(best_sharpe) and best_sharpe < cfg.best_layer_sharpe_min:
        reasons.append(f"best_layer_sharpe<{cfg.best_layer_sharpe_min:.2f}")
    if _is_finite(best_dd) and best_dd > cfg.best_layer_max_drawdown_max:
        reasons.append(f"best_layer_max_drawdown>{cfg.best_layer_max_drawdown_max:.2f}")
    if _is_finite(turnover) and turnover > cfg.turnover_stage_b_max:
        reasons.append(f"turnover_long_only_mean>{cfg.turnover_stage_b_max:.2f}")
    if _is_finite(margin) and margin <= cfg.best_layer_margin_min:
        reasons.append(f"best_layer_margin<={cfg.best_layer_margin_min:.4f}")

    yearly_cons = _to_float(row.get("yearly_sign_consistency"))
    monthly_cons = _to_float(row.get("monthly_sign_consistency"))
    if _is_finite(yearly_cons) and yearly_cons < cfg.yearly_sign_consistency_min:
        reasons.append(f"yearly_sign_consistency<{cfg.yearly_sign_consistency_min:.2f}")
    if _is_finite(monthly_cons) and monthly_cons < cfg.monthly_sign_consistency_min:
        reasons.append(f"monthly_sign_consistency<{cfg.monthly_sign_consistency_min:.2f}")

    robust_pos = _to_float(row.get("robust_period_positive_ratio"))
    robust_sign = _to_float(row.get("robust_ic_sign_consistency"))
    robust_ir_median = _to_float(row.get("robust_ir_median"))
    if _is_finite(robust_pos) and robust_pos < cfg.robust_positive_ratio_min:
        reasons.append(f"robust_period_positive_ratio<{cfg.robust_positive_ratio_min:.2f}")
    if _is_finite(robust_sign) and robust_sign < cfg.robust_ic_sign_consistency_min:
        reasons.append(f"robust_ic_sign_consistency<{cfg.robust_ic_sign_consistency_min:.2f}")
    if _is_finite(robust_ir_median) and abs(robust_ir_median) <= cfg.robust_ir_median_abs_min:
        reasons.append(f"|robust_ir_median|<={cfg.robust_ir_median_abs_min:.2f}")
    return reasons


def build_warning_reasons(row: dict[str, Any], config: FactorEffectivenessConfig | None = None) -> list[str]:
    cfg = config or FactorEffectivenessConfig()
    warnings: list[str] = []

    skew = _to_float(row.get("ic_skew"))
    kurt = _to_float(row.get("ic_kurtosis"))
    obs_count = _to_float(row.get("obs_count"))
    if _is_finite(skew) and abs(skew) > cfg.skew_warning_abs:
        warnings.append(f"|ic_skew|>{cfg.skew_warning_abs:.1f}")
    if _is_finite(kurt) and kurt > cfg.kurt_warning_min:
        warnings.append(f"ic_kurtosis>{cfg.kurt_warning_min:.1f}")
    if _is_finite(obs_count) and obs_count < cfg.obs_warning_min:
        warnings.append(f"obs_count<{int(cfg.obs_warning_min)}")

    turnover = _to_float(row.get("turnover_long_only_mean"))
    if _is_finite(turnover) and turnover > cfg.turnover_stage_b_max and turnover <= cfg.turnover_stage_a_max:
        warnings.append("turnover_long_only_high")
    avg_min_to = _to_float(row.get("membership_turnover_worst_layer", row.get("avg_min_layer_turnover")))
    avg_max_to = _to_float(row.get("membership_turnover_best_layer", row.get("avg_max_layer_turnover")))
    if (
        _is_finite(avg_min_to)
        and _is_finite(avg_max_to)
        and avg_min_to < cfg.stale_turnover_warning_max
        and avg_max_to < cfg.stale_turnover_warning_max
    ):
        warnings.append("stale_signal_warning")
    return warnings


def compute_effectiveness_score(row: dict[str, Any], config: FactorEffectivenessConfig | None = None) -> float:
    return float(compute_effectiveness_score_parts(row, config).get("score_total", 0.0))


def compute_effectiveness_score_for_basis(
    row: dict[str, Any],
    basis: str = "gross",
    config: FactorEffectivenessConfig | None = None,
) -> dict[str, float]:
    """Compute score using suffixed gross/net inputs when they are present.

    The core scorer stays on the historical unsuffixed schema. This wrapper is a
    compatibility adapter for tables that carry both diagnostic gross columns and
    fee-aware net columns.
    """
    selected = _select_score_inputs_by_basis(row, basis=basis)
    return compute_effectiveness_score_parts(selected, config)


def _select_score_inputs_by_basis(row: dict[str, Any], basis: str = "gross") -> dict[str, Any]:
    suffix = str(basis or "gross").strip().lower()
    if suffix not in {"gross", "net"}:
        suffix = "gross"
    out = dict(row or {})
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
        key = f"{base}_{suffix}"
        if key in out and not pd.isna(out.get(key)):
            out[base] = out.get(key)
    return out


def compute_effectiveness_score_parts(
    row: dict[str, Any], config: FactorEffectivenessConfig | None = None
) -> dict[str, float]:
    cfg = config or FactorEffectivenessConfig()
    w = cfg.score_weights

    predictive_power = _weighted_avg(
        [
            (
                _linear_score(abs(_to_float(row.get("ic_mean"))), cfg.ic_abs_stage_b_min, 0.08),
                1.0,
            ),
            (
                _linear_score(abs(_to_float(row.get("ir"))), cfg.ir_abs_stage_b_min, 0.80),
                1.0,
            ),
            (
                _linear_score(_to_float(row.get("sign_adjusted_monotonicity")), 0.00, 0.40),
                1.0,
            ),
            (
                _linear_score_optional(
                    abs(_to_float(row.get("ic_decay_spearman"))),
                    0.00,
                    cfg.ic_decay_spearman_full_score_abs,
                ),
                1.0,
            ),
        ]
    )
    long_only_performance = _weighted_avg(
        [
            (
                _linear_score(_to_float(row.get("best_layer_annualized_return")), 0.05, 0.50),
                1.0,
            ),
            (_linear_score(_to_float(row.get("best_layer_sharpe")), 0.50, 2.50), 1.0),
            (
                _reverse_linear_score(_to_float(row.get("best_layer_max_drawdown")), 0.05, 0.50),
                1.0,
            ),
            (
                _linear_score(
                    _to_float(row.get("best_layer_margin", row.get("margin_long_only"))),
                    0.001,
                    0.030,
                ),
                1.0,
            ),
            (
                _linear_score(
                    _to_float(row.get("best_minus_universe_annualized_return")),
                    0.03,
                    0.48,
                ),
                1.0,
            ),
            (
                _linear_score(_to_float(row.get("best_layer_positive_month_ratio")), 0.48, 0.75),
                1.0,
            ),
        ]
    )
    time_stability = _weighted_avg(
        [
            (
                _linear_score(
                    _to_float(row.get("yearly_sign_consistency")),
                    cfg.yearly_sign_consistency_min,
                    0.90,
                ),
                0.30,
            ),
            (
                _linear_score(
                    _to_float(row.get("monthly_sign_consistency")),
                    cfg.monthly_sign_consistency_min,
                    0.90,
                ),
                0.70,
            ),
        ]
    )
    tradeability = _weighted_avg(
        [
            (
                _reverse_linear_score(_to_float(row.get("turnover_long_only_mean")), 0.15, 0.75),
                0.40,
            ),
            (_linear_score(_to_float(row.get("margin_long_only")), 0.001, 0.020), 0.60),
        ]
    )

    parts = {
        "predictive_power": predictive_power,
        "long_only_performance": long_only_performance,
        "time_stability": time_stability,
        "tradeability": tradeability,
    }

    score = 0.0
    for key, weight in w.items():
        score += float(weight) * float(parts.get(key, 0.0))
    return {
        "score_predictive_power": float(100.0 * predictive_power),
        "score_long_only_performance": float(100.0 * long_only_performance),
        "score_stability": float(100.0 * time_stability),
        "score_tradeability": float(100.0 * tradeability),
        "score_total": float(np.clip(score, 0.0, 100.0)),
    }


def _stage_passes(row: dict[str, Any], cfg: FactorEffectivenessConfig) -> tuple[bool, bool]:
    coverage = _to_float(row.get("coverage_rate"))
    cov_p10 = _to_float(row.get("coverage_rate_by_date_p10"))
    obs_count = _to_float(row.get("obs_count"))
    ic_mean = _to_float(row.get("ic_mean"))
    ir = _to_float(row.get("ir"))
    p_value = _to_float(row.get("p_value"))
    best_ann = _to_float(row.get("best_layer_annualized_return"))
    best_excess = _to_float(row.get("best_minus_universe_annualized_return"))
    best_sharpe = _to_float(row.get("best_layer_sharpe"))
    best_dd = _to_float(row.get("best_layer_max_drawdown"))
    turnover = _to_float(row.get("turnover_long_only_mean"))
    margin = _to_float(row.get("best_layer_margin", row.get("margin_long_only")))
    yearly_cons = _to_float(row.get("yearly_sign_consistency"))
    monthly_cons = _to_float(row.get("monthly_sign_consistency"))
    robust_pos = _to_float(row.get("robust_period_positive_ratio"))
    robust_sign = _to_float(row.get("robust_ic_sign_consistency"))
    robust_ir_median = _to_float(row.get("robust_ir_median"))

    stage_a = True
    stage_a &= not (_is_finite(coverage) and coverage < cfg.coverage_hard_reject_min)
    stage_a &= not (_is_finite(obs_count) and obs_count < cfg.obs_hard_reject_min)
    stage_a &= _is_finite(abs(ic_mean)) and abs(ic_mean) >= cfg.ic_abs_stage_a_min
    stage_a &= _is_finite(abs(ir)) and abs(ir) >= cfg.ir_abs_stage_a_min
    stage_a &= _is_finite(p_value) and p_value <= cfg.p_value_stage_a_max
    stage_a &= _is_finite(best_ann) and best_ann > 0
    stage_a &= _is_finite(best_sharpe) and best_sharpe >= cfg.best_layer_sharpe_min
    stage_a &= _is_finite(best_dd) and best_dd <= cfg.best_layer_max_drawdown_max
    stage_a &= not (_is_finite(turnover) and turnover > cfg.turnover_stage_a_max)

    stage_b = bool(stage_a)
    stage_b &= _is_finite(coverage) and coverage >= cfg.coverage_stage_b_min
    if _is_finite(cov_p10):
        stage_b &= cov_p10 >= cfg.coverage_by_date_p10_stage_b_min
    stage_b &= _is_finite(obs_count) and obs_count >= cfg.obs_stage_b_min
    stage_b &= _is_finite(abs(ic_mean)) and abs(ic_mean) >= cfg.ic_abs_stage_b_min
    stage_b &= _is_finite(abs(ir)) and abs(ir) >= cfg.ir_abs_stage_b_min
    stage_b &= _is_finite(p_value) and p_value <= cfg.p_value_stage_b_max
    stage_b &= _is_finite(best_ann) and best_ann > cfg.best_layer_annualized_return_min
    stage_b &= _is_finite(best_excess) and best_excess > cfg.best_minus_universe_annualized_return_min
    stage_b &= _is_finite(best_sharpe) and best_sharpe >= cfg.best_layer_sharpe_min
    stage_b &= _is_finite(best_dd) and best_dd <= cfg.best_layer_max_drawdown_max
    stage_b &= not (_is_finite(turnover) and turnover > cfg.turnover_stage_b_max)
    stage_b &= _is_finite(margin) and margin > cfg.best_layer_margin_min

    has_time = _is_finite(yearly_cons) or _is_finite(monthly_cons)
    if has_time or cfg.require_time_stability_when_available:
        if _is_finite(yearly_cons):
            stage_b &= yearly_cons >= cfg.yearly_sign_consistency_min
        if _is_finite(monthly_cons):
            stage_b &= monthly_cons >= cfg.monthly_sign_consistency_min

    has_robust = _is_finite(robust_pos) or _is_finite(robust_sign) or _is_finite(robust_ir_median)
    if has_robust or cfg.require_robustness_when_available:
        if _is_finite(robust_pos):
            stage_b &= robust_pos >= cfg.robust_positive_ratio_min
        if _is_finite(robust_sign):
            stage_b &= robust_sign >= cfg.robust_ic_sign_consistency_min
        if _is_finite(robust_ir_median):
            stage_b &= abs(robust_ir_median) > cfg.robust_ir_median_abs_min

    return bool(stage_a), bool(stage_b)


def _score_to_tier(score: float, cfg: FactorEffectivenessConfig) -> str:
    if not _is_finite(score):
        return "REJECT"
    if score >= cfg.tier_cutoffs.get("S", 85.0):
        return "S"
    if score >= cfg.tier_cutoffs.get("A", 75.0):
        return "A"
    if score >= cfg.tier_cutoffs.get("B", 65.0):
        return "B"
    if score >= cfg.tier_cutoffs.get("C", 55.0):
        return "C"
    return "REJECT"


def _is_sign_consistency_pass(ic_mean: float, pos_ratio: float, stage: str, cfg: FactorEffectivenessConfig) -> bool:
    if not _is_finite(ic_mean) or not _is_finite(pos_ratio):
        return False
    if stage == "b":
        pos_min = cfg.positive_ic_ratio_stage_b_pos_min
        neg_max = cfg.positive_ic_ratio_stage_b_neg_max
    else:
        pos_min = cfg.positive_ic_ratio_stage_a_pos_min
        neg_max = cfg.positive_ic_ratio_stage_a_neg_max
    if ic_mean > 0:
        return pos_ratio >= pos_min
    if ic_mean < 0:
        return pos_ratio <= neg_max
    return False


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


def _to_factor_map(df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "factor" not in df.columns:
        return {}
    work = df.copy()
    work["factor"] = work["factor"].astype(str)
    return work.set_index("factor").to_dict("index")


def _coverage_by_date_p10_map(
    coverage_by_date_df: pd.DataFrame | None,
) -> dict[str, float]:
    if coverage_by_date_df is None or not isinstance(coverage_by_date_df, pd.DataFrame) or coverage_by_date_df.empty:
        return {}
    required = {"factor", "coverage_rate"}
    if not required.issubset(coverage_by_date_df.columns):
        return {}
    work = coverage_by_date_df.copy()
    work["factor"] = work["factor"].astype(str)
    out: dict[str, float] = {}
    for factor, g in work.groupby("factor", sort=False):
        s = pd.to_numeric(g["coverage_rate"], errors="coerce").dropna()
        out[factor] = float(s.quantile(0.1)) if not s.empty else np.nan
    return out


def _turnover_summary_map(
    turnover_results: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for factor, tr in turnover_results.items():
        if tr is None or not isinstance(tr, pd.DataFrame) or tr.empty:
            continue
        out[str(factor)] = {
            "avg_min_layer_turnover": float(pd.to_numeric(tr["min_layer_turnover"], errors="coerce").mean()),
            "avg_max_layer_turnover": float(pd.to_numeric(tr["max_layer_turnover"], errors="coerce").mean()),
        }
    return out


def _ic_sign_consistency_map(
    breakdown_df: pd.DataFrame | None,
    summary_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    if breakdown_df is None or not isinstance(breakdown_df, pd.DataFrame) or breakdown_df.empty:
        return {}
    if "factor" not in breakdown_df.columns or "ic_mean" not in breakdown_df.columns:
        return {}

    overall_sign: dict[str, int] = {}
    if (
        summary_df is not None
        and isinstance(summary_df, pd.DataFrame)
        and not summary_df.empty
        and "factor" in summary_df.columns
    ):
        tmp = summary_df[["factor", "ic_mean"]].copy()
        tmp["factor"] = tmp["factor"].astype(str)
        for _, row in tmp.iterrows():
            overall_sign[str(row["factor"])] = _sign(_to_float(row["ic_mean"]))

    work = breakdown_df.copy()
    work["factor"] = work["factor"].astype(str)
    out: dict[str, float] = {}
    for factor, g in work.groupby("factor", sort=False):
        s = pd.to_numeric(g["ic_mean"], errors="coerce").dropna()
        if s.empty:
            out[factor] = np.nan
            continue
        sign_ref = overall_sign.get(factor, _sign(float(s.mean())))
        if sign_ref == 0:
            out[factor] = np.nan
            continue
        signs = np.sign(s.values.astype(float))
        valid = signs != 0
        if not np.any(valid):
            out[factor] = np.nan
            continue
        out[factor] = float((signs[valid] == sign_ref).mean())
    return out


def _ic_decay_spearman_map(
    ic_decay_df: pd.DataFrame | None,
    preferred_phase: str = "train",
) -> dict[str, float]:
    if ic_decay_df is None or not isinstance(ic_decay_df, pd.DataFrame) or ic_decay_df.empty:
        return {}
    required = {"factor", "phase", "ic_decay_rank_corr"}
    if not required.issubset(ic_decay_df.columns):
        return {}

    work = ic_decay_df[list(required)].copy()
    work["factor"] = work["factor"].astype(str)
    work["phase"] = work["phase"].astype(str)
    work["ic_decay_rank_corr"] = pd.to_numeric(work["ic_decay_rank_corr"], errors="coerce")
    work = work.dropna(subset=["ic_decay_rank_corr"])
    if work.empty:
        return {}

    out: dict[str, float] = {}
    preferred_key = str(preferred_phase or "train")
    for factor, group in work.groupby("factor", sort=False):
        preferred = group[group["phase"] == preferred_key]
        source = preferred if not preferred.empty else group
        vals = source["ic_decay_rank_corr"].dropna().drop_duplicates()
        out[str(factor)] = float(vals.mean()) if not vals.empty else np.nan
    return out


def _robustness_summary_map(
    period_comparison_df: pd.DataFrame | None,
    summary_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    if period_comparison_df is None or not isinstance(period_comparison_df, pd.DataFrame) or period_comparison_df.empty:
        return {}
    required = {"factor", "period", "ic_mean", "ir", "long_short_total_return"}
    if not required.issubset(period_comparison_df.columns):
        return {}

    overall_sign: dict[str, int] = {}
    if (
        summary_df is not None
        and isinstance(summary_df, pd.DataFrame)
        and not summary_df.empty
        and "factor" in summary_df.columns
    ):
        tmp = summary_df[["factor", "ic_mean"]].copy()
        tmp["factor"] = tmp["factor"].astype(str)
        for _, row in tmp.iterrows():
            overall_sign[str(row["factor"])] = _sign(_to_float(row["ic_mean"]))

    work = period_comparison_df.copy()
    work["factor"] = work["factor"].astype(str)
    out: dict[str, dict[str, float]] = {}
    for factor, g in work.groupby("factor", sort=False):
        ret_s = pd.to_numeric(g["long_short_total_return"], errors="coerce")
        ic_s = pd.to_numeric(g["ic_mean"], errors="coerce")
        ir_s = pd.to_numeric(g["ir"], errors="coerce")

        ret_valid = ret_s.dropna()
        ic_valid = ic_s.dropna()
        sign_ref = overall_sign.get(factor, _sign(float(ic_valid.mean())) if not ic_valid.empty else 0)
        if sign_ref == 0:
            ic_cons = np.nan
        else:
            ic_sign = np.sign(ic_valid.values.astype(float))
            valid = ic_sign != 0
            ic_cons = float((ic_sign[valid] == sign_ref).mean()) if np.any(valid) else np.nan

        out[factor] = {
            "robust_period_positive_ratio": float((ret_valid > 0).mean()) if not ret_valid.empty else np.nan,
            "robust_ic_sign_consistency": ic_cons,
            "robust_ir_median": float(ir_s.dropna().median()) if ir_s.notna().any() else np.nan,
        }
    return out


def _sign(value: float) -> int:
    if not _is_finite(value):
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    if np.isfinite(out):
        return out
    return np.nan


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _clip01(value: float) -> float:
    if not _is_finite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _linear_score(value: float, low: float, high: float) -> float:
    if not _is_finite(value):
        return 0.0
    if high <= low:
        return 1.0 if value >= low else 0.0
    return _clip01((value - low) / (high - low))


def _linear_score_optional(value: float, low: float, high: float) -> float:
    if not _is_finite(value):
        return np.nan
    return _linear_score(value, low, high)


def _reverse_linear_score(value: float, low_good: float, high_bad: float) -> float:
    if not _is_finite(value):
        return 0.0
    if high_bad <= low_good:
        return 1.0 if value <= low_good else 0.0
    return _clip01((high_bad - value) / (high_bad - low_good))


def _avg(values: Sequence[float]) -> float:
    valid = [float(v) for v in values if _is_finite(v)]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def _weighted_avg(values: Sequence[tuple[float, float]]) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for value, weight in values:
        if not _is_finite(value) or not _is_finite(weight) or weight <= 0:
            continue
        weighted_sum += float(value) * float(weight)
        weight_sum += float(weight)
    if weight_sum <= 0:
        return 0.0
    return float(weighted_sum / weight_sum)


def _binary_positive(value: float) -> float:
    if not _is_finite(value):
        return 0.0
    return 1.0 if value > 0 else 0.0


def _sign_consistency_score(ic_mean: float, pos_ratio: float) -> float:
    if not _is_finite(ic_mean) or not _is_finite(pos_ratio):
        return 0.0
    if ic_mean > 0:
        return _linear_score(pos_ratio, 0.50, 0.70)
    if ic_mean < 0:
        return _linear_score(1.0 - pos_ratio, 0.50, 0.70)
    return 0.0


def _mono_ratio_score(ic_mean: float, mono_ratio: float, cfg: FactorEffectivenessConfig) -> float:
    if not _is_finite(ic_mean) or not _is_finite(mono_ratio):
        return 0.0
    if ic_mean > 0:
        return _linear_score(mono_ratio, cfg.monotonicity_ratio_pos_min - 0.10, 0.80)
    if ic_mean < 0:
        return _linear_score(1.0 - mono_ratio, (1.0 - cfg.monotonicity_ratio_neg_max) - 0.10, 0.80)
    return 0.0


def calculate_factor_correlation(
    df: pd.DataFrame,
    factor_cols: list[str],
    method: str = "factor_values",
    layer_results: dict | None = None,
    long_short_returns_dict: dict | None = None,
    pairwise_complete_obs: bool = True,
) -> pd.DataFrame:
    """
    Calculate factor correlation by factor values or long-short return series.

    `pairwise_complete_obs=True` (default) avoids listwise deletion across all
    factors and uses pairwise-valid observations for each correlation pair.
    """
    if method == "factor_values":
        factor_data = df[factor_cols].copy()
        return factor_data.corr(method="pearson")

    if method != "long_short_returns":
        raise ValueError("method must be 'factor_values' or 'long_short_returns'")

    if long_short_returns_dict is not None:
        filtered = {
            factor: long_short_returns_dict[factor] for factor in factor_cols if factor in long_short_returns_dict
        }
        if not filtered:
            return pd.DataFrame(index=factor_cols, columns=factor_cols)
        long_short_df = pd.DataFrame(filtered)
        if not pairwise_complete_obs:
            long_short_df = long_short_df.dropna()
        return long_short_df.corr(method="pearson")

    if layer_results is None:
        raise ValueError("When method='long_short_returns', provide layer_results or long_short_returns_dict")

    ls_dict = {}
    for factor in factor_cols:
        if factor not in layer_results:
            continue
        data = layer_results[factor]
        return_col = data.columns[-1]
        daily_layer_returns = data.groupby(["trade_date", "layer"])[return_col].mean().reset_index()
        daily_layer_returns_wide = daily_layer_returns.pivot(index="trade_date", columns="layer", values=return_col)
        numeric_columns = [col for col in daily_layer_returns_wide.columns if isinstance(col, (int, np.integer))]
        if not numeric_columns:
            continue

        min_layer = min(numeric_columns)
        max_layer = max(numeric_columns)
        min_layer_cumulative = cumulative_returns(daily_layer_returns_wide[min_layer])
        max_layer_cumulative = cumulative_returns(daily_layer_returns_wide[max_layer])
        if min_layer_cumulative.iloc[-1] > max_layer_cumulative.iloc[-1]:
            long_short_returns = daily_layer_returns_wide[min_layer] - daily_layer_returns_wide[max_layer]
        else:
            long_short_returns = daily_layer_returns_wide[max_layer] - daily_layer_returns_wide[min_layer]

        ls_dict[factor] = long_short_returns

    if not ls_dict:
        return pd.DataFrame(index=factor_cols, columns=factor_cols)
    ls_df = pd.DataFrame(ls_dict)
    if not pairwise_complete_obs:
        ls_df = ls_df.dropna()
    return ls_df.corr(method="pearson")


def filter_factors_by_correlation_advanced(
    correlation_matrix: pd.DataFrame,
    effective_factors_df: pd.DataFrame,
    threshold: float = 0.7,
    use_absolute_corr: bool = False,
) -> pd.DataFrame:
    """
    Greedy independent-set style filtering using Sharpe ranking.

    use_absolute_corr:
    - False (legacy): conflicts when corr > threshold.
    - True: conflicts when abs(corr) > threshold.
    """
    factors_in_corr = set(correlation_matrix.index)
    effective_factors_in_corr = effective_factors_df[effective_factors_df["factor"].isin(factors_in_corr)].copy()
    if len(effective_factors_in_corr) == 0:
        return effective_factors_df

    factor_to_sharpe = dict(
        zip(
            effective_factors_in_corr["factor"],
            effective_factors_in_corr["sharpe_ratio"],
        )
    )

    conflicts = {}
    for factor in correlation_matrix.columns:
        if factor not in factor_to_sharpe:
            continue
        conflicts[factor] = []
        for other_factor in correlation_matrix.columns:
            if other_factor not in factor_to_sharpe or factor == other_factor:
                continue
            corr_value = correlation_matrix.loc[factor, other_factor]
            conflict_value = abs(corr_value) if use_absolute_corr else corr_value
            if conflict_value > threshold:
                conflicts[factor].append(other_factor)

    sorted_factors = sorted(factor_to_sharpe.keys(), key=lambda f: factor_to_sharpe[f], reverse=True)

    selected_factors = []
    for factor in sorted_factors:
        conflict = False
        for selected in selected_factors:
            if factor in conflicts[selected]:
                conflict = True
                break
        if not conflict:
            selected_factors.append(factor)

    filtered_factors_df = effective_factors_in_corr[effective_factors_in_corr["factor"].isin(selected_factors)]
    return filtered_factors_df.reset_index(drop=True)
