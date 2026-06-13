from __future__ import annotations

import warnings

import pandas as pd

from .combination import (
    backtest_factor_strategy,
    calculate_factor_weights,
    combine_factors_with_weights,
)
from .config import FactorResearchConfig
from .double_sort import (
    DoubleSortConfig,
    assign_quantile_groups,
    double_sort_analysis,
    newey_west_stats,
    newey_west_tstat,
)
from .diagnostics import (
    analyze_holding_period_robustness,
    calculate_factor_coverage,
    calculate_ic_stability,
    calculate_ic_time_breakdown,
    calculate_layer_monotonicity,
    summarize_ic_sign_consistency,
    summarize_period_robustness,
)
from .ewma import (
    backtest_ewma_strategy,
    calculate_ewma_factors,
    calculate_factor_returns,
    predict_stock_returns,
    run_ewma_factor_strategy,
)
from .metrics import (
    prepare_visualization_data_for_all_factors,
    restructure_factor_analysis_data,
)
from .preprocess import (
    add_execution_return_audit_columns,
    build_return_semantics_metadata,
    process_factor_data,
    process_future_return,
)
from .reporting import (
    build_factor_summary_report,
    build_structured_report,
    export_report,
)
from .screening import (
    FactorEffectivenessConfig,
    calculate_factor_correlation,
    compute_effectiveness_score_for_basis,
    evaluate_factor_effectiveness,
    filter_effective_factors,
    filter_factors_by_correlation_advanced,
)
from .sample_splits import (
    PhaseWindow,
    SampleSplitConfig,
    assign_phase,
    assign_sample_split,
    build_phase_windows,
    summarize_split_metrics,
)
from .single_factor import (
    TransactionCostConfig,
    build_portfolio_pnl_table,
    calculate_layer_portfolio_turnover,
    calculate_icir,
    calculate_best_layer_metrics,
    calculate_long10_portfolio_returns,
    calculate_margin_metrics,
    calculate_long_short_metrics,
    calculate_long_only_portfolio_turnover,
    calculate_turnover_rate,
    factor_layer_analysis,
    process_factors_individually,
    summarize_long10_portfolio_returns,
    summarize_long_only_turnover,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

try:
    from .plotting import (
        visualize_backtest_results,
        visualize_ewma_backtest_results,
        visualize_factor_distribution,
        visualize_factor_correlation,
        visualize_ic_analysis,
        visualize_ic_yearly_bar,
        visualize_layer_terminal_values,
        visualize_layer_analysis,
        visualize_period_comparison,
        visualize_turnover_analysis,
    )
except Exception as _plotting_import_error:

    def _plotting_unavailable(*args, **kwargs):
        raise ImportError("Plotting dependencies are not installed") from _plotting_import_error  # noqa: F821

    visualize_backtest_results = _plotting_unavailable
    visualize_ewma_backtest_results = _plotting_unavailable
    visualize_factor_distribution = _plotting_unavailable
    visualize_factor_correlation = _plotting_unavailable
    visualize_ic_analysis = _plotting_unavailable
    visualize_ic_yearly_bar = _plotting_unavailable
    visualize_layer_terminal_values = _plotting_unavailable
    visualize_layer_analysis = _plotting_unavailable
    visualize_period_comparison = _plotting_unavailable
    visualize_turnover_analysis = _plotting_unavailable

__all__ = [
    "FactorResearchConfig",
    "DoubleSortConfig",
    "assign_quantile_groups",
    "newey_west_tstat",
    "newey_west_stats",
    "double_sort_analysis",
    "SampleSplitConfig",
    "PhaseWindow",
    "assign_phase",
    "assign_sample_split",
    "build_phase_windows",
    "summarize_split_metrics",
    "process_factor_data",
    "process_future_return",
    "build_return_semantics_metadata",
    "add_execution_return_audit_columns",
    "calculate_factor_coverage",
    "calculate_ic_time_breakdown",
    "calculate_ic_stability",
    "calculate_layer_monotonicity",
    "summarize_ic_sign_consistency",
    "summarize_period_robustness",
    "analyze_holding_period_robustness",
    "build_factor_summary_report",
    "build_structured_report",
    "export_report",
    "calculate_icir",
    "TransactionCostConfig",
    "visualize_ic_analysis",
    "visualize_factor_distribution",
    "visualize_ic_yearly_bar",
    "visualize_layer_terminal_values",
    "visualize_period_comparison",
    "factor_layer_analysis",
    "build_portfolio_pnl_table",
    "calculate_layer_portfolio_turnover",
    "calculate_best_layer_metrics",
    "calculate_long10_portfolio_returns",
    "calculate_long_short_metrics",
    "calculate_long_only_portfolio_turnover",
    "summarize_long10_portfolio_returns",
    "summarize_long_only_turnover",
    "calculate_margin_metrics",
    "visualize_layer_analysis",
    "calculate_turnover_rate",
    "visualize_turnover_analysis",
    "process_factors_individually",
    "FactorEffectivenessConfig",
    "evaluate_factor_effectiveness",
    "filter_effective_factors",
    "restructure_factor_analysis_data",
    "prepare_visualization_data_for_all_factors",
    "calculate_factor_correlation",
    "compute_effectiveness_score_for_basis",
    "visualize_factor_correlation",
    "filter_factors_by_correlation_advanced",
    "calculate_factor_weights",
    "combine_factors_with_weights",
    "backtest_factor_strategy",
    "visualize_backtest_results",
    "calculate_factor_returns",
    "calculate_ewma_factors",
    "predict_stock_returns",
    "backtest_ewma_strategy",
    "visualize_ewma_backtest_results",
    "run_ewma_factor_strategy",
]
