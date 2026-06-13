from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

from .double_sort import newey_west_stats
from .preprocess import process_factor_data
from .utils import (
    apply_period_sampling,
    assign_quantile_labels,
    calculate_risk_metrics,
    cumulative_returns,
    ensure_and_sort_panel,
    ensure_columns,
    infer_return_column_from_layer_frame,
    resolve_future_return_column,
    get_logger,
)


@dataclass(frozen=True)
class TransactionCostConfig:
    enabled: bool = True
    mode: str = "flat"
    model_name: str = "cn_a_linear_v1"
    commission_bps_per_side: float = 2.0
    slippage_bps_per_side: float = 3.0
    stamp_tax_bps_sell: float = 5.0
    transfer_fee_bps_per_side: float = 0.1
    exchange_fee_bps_per_side: float = 0.341
    regulatory_fee_bps_per_side: float = 0.2
    include_commission: bool = True
    include_slippage: bool = True
    include_stamp_tax: bool = True
    include_transfer_fee: bool = True
    include_exchange_fee: bool = True
    include_regulatory_fee: bool = True
    charge_initial_position: bool = False
    apply_to_long_only: bool = True
    apply_to_long10: bool = True
    apply_to_layers: bool = True
    apply_to_long_short: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _grouped_rank_corr_by_codes(
    x_values: np.ndarray,
    y_values: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Compute per-group correlation with integer group codes using vectorized bincount."""
    out = np.full(int(n_groups), np.nan, dtype=np.float64)
    if n_groups <= 0:
        return out

    valid = np.isfinite(x_values) & np.isfinite(y_values) & (group_codes >= 0)
    if not np.any(valid):
        return out

    g = group_codes[valid]
    x = x_values[valid]
    y = y_values[valid]

    n = np.bincount(g, minlength=n_groups).astype(np.float64)
    sum_x = np.bincount(g, weights=x, minlength=n_groups)
    sum_y = np.bincount(g, weights=y, minlength=n_groups)
    sum_xx = np.bincount(g, weights=x * x, minlength=n_groups)
    sum_yy = np.bincount(g, weights=y * y, minlength=n_groups)
    sum_xy = np.bincount(g, weights=x * y, minlength=n_groups)

    numerator = n * sum_xy - sum_x * sum_y
    denom_sq = (n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)
    denom_sq = np.where(denom_sq < 0.0, 0.0, denom_sq)
    denominator = np.sqrt(denom_sq)

    valid_group = (n >= 2.0) & np.isfinite(denominator) & (denominator > 0.0)
    out[valid_group] = numerator[valid_group] / denominator[valid_group]
    out[~np.isfinite(out)] = np.nan
    return out


def calculate_icir(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_col: str = "pct_chg",
    period: int = 1,
    max_lag: int | None = None,
):
    """Calculate daily Information Coefficient (IC) series and IC/IR summary.

    Computes Spearman rank correlation between factor values and forward returns
    for each trading day, then summarizes with IC mean, IC standard deviation,
    and Information Ratio (IR = IC_mean / IC_std).

    Args:
        df: Long-format DataFrame with columns 'trade_date', 'znz_code',
            factor columns, and return column.
        factor_cols: List of factor column names to analyze.
        return_col: Name of the return column. Defaults to 'pct_chg'.
        period: Forward return period in trading days. Defaults to 1.
        max_lag: Maximum lag for IC decay analysis. If None, skip lag analysis.

    Returns:
        Tuple of (ic_df, summary_df) or (ic_df, summary_df, lag_analysis_results):
        - ic_df: DataFrame with daily IC values for each factor.
        - summary_df: DataFrame with IC statistics (mean, std, IR, t-stat, p-value).
        - lag_analysis_results: List of dicts with IC decay analysis (if max_lag set).

    Example:
        >>> ic_df, summary = calculate_icir(df, ['momentum_20d', 'volatility_20d'])
        >>> print(summary[['factor', 'ic_mean', 'ir']])
    """
    ensure_columns(df, ["trade_date", "znz_code"], caller="calculate_icir")
    df = ensure_and_sort_panel(df, caller="calculate_icir")

    active_return_col = resolve_future_return_column(df, return_col, period, caller="calculate_icir")
    ensure_columns(df, [active_return_col], caller="calculate_icir")
    df = apply_period_sampling(df, period=period, drop_tail_for_future=True)
    df = df.dropna(subset=[active_return_col]).copy()
    if df.empty:
        get_logger().warning("[calculate_icir] empty frame after sampling/dropna on return column")
    else:
        min_cs = df.groupby("trade_date").size().min()
        if min_cs < 2:
            get_logger().warning("[calculate_icir] some dates have fewer than 2 stocks; IC may be invalid")

    date_groups = df.groupby("trade_date", sort=False)
    ranked_return = date_groups[active_return_col].rank()
    ranked_return_values = np.asarray(pd.to_numeric(ranked_return, errors="coerce"), dtype=np.float64)
    date_codes, date_index = pd.factorize(df["trade_date"], sort=False)
    n_dates = int(len(date_index))

    active_factors = [factor for factor in factor_cols if factor in df.columns]
    ranked_factors = date_groups[active_factors].rank() if active_factors else pd.DataFrame(index=df.index)

    ic_series_map: dict[str, pd.Series] = {}
    for factor in active_factors:
        ranked_factor_values = np.asarray(pd.to_numeric(ranked_factors[factor], errors="coerce"), dtype=np.float64)
        daily_ic_values = _grouped_rank_corr_by_codes(
            x_values=ranked_factor_values,
            y_values=ranked_return_values,
            group_codes=date_codes,
            n_groups=n_dates,
        )
        ic_series_map[f"{factor}_ic"] = pd.Series(daily_ic_values, index=date_index)

    if ic_series_map:
        ic_df = pd.DataFrame(ic_series_map)
        ic_df.index.name = "trade_date"
        ic_df = ic_df.reset_index().sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    else:
        ic_df = pd.DataFrame(columns=["trade_date"])

    summary_results = []
    for factor in factor_cols:
        ic_col = f"{factor}_ic"
        if ic_col not in ic_df.columns:
            continue
        ic_series = ic_df[ic_col].dropna()
        if len(ic_series) == 0:
            continue
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std != 0 else 0
        n = len(ic_series)
        nw = newey_west_stats(ic_series)
        t_stat = nw.get("nw_t", 0.0) if np.isfinite(nw.get("nw_t", np.nan)) else 0.0
        p_value = nw.get("p_value", 1.0) if np.isfinite(nw.get("p_value", np.nan)) else 1.0
        summary_results.append(
            {
                "factor": factor,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ir": ir,
                "ic_valid_count": n,
                "positive_ic_ratio": (ic_series > 0).sum() / len(ic_series),
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

    summary_df = pd.DataFrame(summary_results)

    if max_lag is None:
        return ic_df, summary_df

    lag_analysis_results = []
    grouped_by_code_return = df.groupby("znz_code", sort=False)[active_return_col]
    lag_ranked_return: dict[int, np.ndarray] = {0: ranked_return_values}
    for lag in range(1, max_lag + 1):
        lag_series = grouped_by_code_return.shift(-lag)
        lag_ranked = lag_series.groupby(df["trade_date"], sort=False).rank()
        lag_ranked_return[lag] = np.asarray(pd.to_numeric(lag_ranked, errors="coerce"), dtype=np.float64)

    for factor in active_factors:
        ranked_factor_values = np.asarray(pd.to_numeric(ranked_factors[factor], errors="coerce"), dtype=np.float64)
        lag_ic_values = []
        for lag in range(max_lag + 1):
            daily_corr_values = _grouped_rank_corr_by_codes(
                x_values=ranked_factor_values,
                y_values=lag_ranked_return[lag],
                group_codes=date_codes,
                n_groups=n_dates,
            )
            mean_ic = float(np.nanmean(daily_corr_values)) if np.isfinite(daily_corr_values).any() else np.nan
            lag_ic_values.append(mean_ic)

        # Spearman rank-correlation between lag index and lag IC mean values
        # to measure whether IC decays monotonically as lag increases.
        decay_pairs = [(lag, v) for lag, v in enumerate(lag_ic_values) if v is not None and np.isfinite(v)]
        ic_decay_rank_corr = np.nan
        if len(decay_pairs) >= 2:
            lag_idx = [lag for lag, _ in decay_pairs]
            lag_vals = [v for _, v in decay_pairs]
            if len(set(lag_idx)) >= 2 and len(set(lag_vals)) >= 2:
                rank_corr = stats.spearmanr(lag_idx, lag_vals, nan_policy="omit").correlation
                if rank_corr is not None and np.isfinite(rank_corr):
                    ic_decay_rank_corr = float(rank_corr)

        if lag_ic_values and not np.isnan(lag_ic_values[0]) and lag_ic_values[0] != 0:
            half_life = None
            for i in range(1, len(lag_ic_values)):
                if not np.isnan(lag_ic_values[i]) and abs(lag_ic_values[i]) < abs(lag_ic_values[0] / 2):
                    half_life = i
                    break
            lag_analysis_results.append(
                {
                    "factor": factor,
                    "lag_ic_values": lag_ic_values,
                    "half_life": half_life,
                    "ic_decay_rank_corr": ic_decay_rank_corr,
                }
            )
        else:
            lag_analysis_results.append(
                {
                    "factor": factor,
                    "lag_ic_values": lag_ic_values,
                    "half_life": None,
                    "ic_decay_rank_corr": ic_decay_rank_corr,
                }
            )

    return ic_df, summary_df, lag_analysis_results


def factor_layer_analysis(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_col: str = "pct_chg",
    period: int = 1,
    layers: int = 5,
    passthrough_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run cross-sectional layer/quantile analysis for multiple factors.

    Groups stocks into quantile layers based on factor values and computes
    layer returns, holdings, and cumulative performance.

    Args:
        df: Long-format DataFrame with 'trade_date', 'znz_code', factor columns,
            and return column.
        factor_cols: List of factor column names to analyze.
        return_col: Name of the return column. Defaults to 'pct_chg'.
        period: Forward return period in trading days. Defaults to 1.
        layers: Number of quantile layers. Defaults to 5.
        passthrough_cols: Additional columns to include in layer holdings.

    Returns:
        Dictionary mapping factor names to DataFrames with layer analysis results.
        Each DataFrame contains columns: trade_date, layer, layer_return, holdings.

    Example:
        >>> results = factor_layer_analysis(df, ['momentum_20d'], layers=5)
        >>> momentum_layers = results['momentum_20d']
        >>> print(momentum_layers.groupby('layer')['layer_return'].mean())
    """
    ensure_columns(df, ["znz_code", "trade_date", return_col], caller="factor_layer_analysis")
    active_return_col = resolve_future_return_column(df, return_col, period, caller="factor_layer_analysis")
    df = apply_period_sampling(df, period=period, drop_tail_for_future=True)

    layer_results: dict[str, pd.DataFrame] = {}

    for factor in factor_cols:
        if factor not in df.columns:
            continue

        passthrough = [str(c) for c in (passthrough_cols or []) if str(c) in df.columns]
        clean_cols = ["znz_code", factor, active_return_col, "trade_date"] + passthrough
        clean_df = df[clean_cols].dropna(subset=["znz_code", factor, active_return_col, "trade_date"])
        if len(clean_df) == 0:
            continue

        clean_df = clean_df.sort_values(["trade_date", factor])
        group_sizes = clean_df.groupby("trade_date").size()
        total_dates = int(group_sizes.shape[0]) if len(group_sizes) else 0
        quantile_warn_stats: dict[str, float | int] = {}
        if len(group_sizes) and group_sizes.min() < layers:
            get_logger().warning(
                "[factor_layer_analysis] factor=%s has dates with too few stocks (min=%s < layers=%s)",
                factor,
                int(group_sizes.min()),
                layers,
            )

        clean_df["layer"] = clean_df.groupby("trade_date")[factor].transform(
            lambda s: assign_quantile_labels(
                s,
                layers,
                labels_name="layer",
                warn_context=None,
                warn_stats=quantile_warn_stats,
            )
        )
        qcut_fallback_count = int(quantile_warn_stats.get("qcut_fallback_count", 0))
        if qcut_fallback_count > 0:
            get_logger().warning(
                "[factor_layer_analysis] factor=%s qcut fallback happened on %s/%s dates (duplicated/constant values likely); per-date warnings suppressed",
                factor,
                qcut_fallback_count,
                total_dates,
            )
        fewer_groups_count = int(quantile_warn_stats.get("fewer_groups_count", 0))
        if fewer_groups_count > 0:
            min_groups = int(float(quantile_warn_stats.get("fewer_groups_min", layers)))
            get_logger().warning(
                "[factor_layer_analysis] factor=%s produced fewer groups than requested on %s/%s dates (worst=%s < %s); per-date warnings suppressed",
                factor,
                fewer_groups_count,
                total_dates,
                min_groups,
                layers,
            )
        clean_df = clean_df.dropna(subset=["layer"])
        if len(clean_df) == 0:
            continue

        clean_df["layer"] = clean_df["layer"].astype(int)
        layer_results[factor] = clean_df[
            ["znz_code", "trade_date", "layer", factor] + passthrough + [active_return_col]
        ].copy()
        layer_results[factor].attrs["qcut_fallback_count"] = qcut_fallback_count
        layer_results[factor].attrs["fewer_groups_count"] = fewer_groups_count
        layer_results[factor].attrs["total_dates"] = total_dates
        layer_results[factor].attrs["insufficient_stock_dates"] = (
            int((group_sizes < layers).sum()) if len(group_sizes) else 0
        )

    return layer_results


def calculate_long_short_metrics(
    layer_results: dict[str, pd.DataFrame],
    period: int = 1,
    include_long_short_visualization: bool = True,
    direction_mode: str = "by_ic_sign",
    ic_summary_df: pd.DataFrame | None = None,
    ic_signs_override: dict[str, float] | None = None,
    sharpe_penalty_divisor: float = 2.0,
    gross_exposure: float = 2.0,
):
    """
    Calculate long-short performance metrics from layer results.

    direction_mode:
    - "by_ic_sign" (default): use sign of `ic_mean` from `ic_summary_df` or `ic_signs_override`.
    - "auto_by_final_return" (deprecated): decide long-short direction by terminal return of top/bottom layer. Has look-ahead bias.
    - "top_minus_bottom": fixed top layer minus bottom layer.

    gross_exposure:
    - If long and short legs are each 100% notional, gross exposure is 2.0.
    - Reported strategy returns are spread returns divided by gross exposure.
    """
    long_short_metrics: dict[str, dict] = {}
    layer_results_for_visualization: dict[str, pd.DataFrame] = {}

    for factor, data in layer_results.items():
        return_col = infer_return_column_from_layer_frame(data)
        daily_layer_returns = data.groupby(["trade_date", "layer"])[return_col].mean().reset_index()
        daily_layer_returns_wide = daily_layer_returns.pivot(index="trade_date", columns="layer", values=return_col)

        numeric_columns = [col for col in daily_layer_returns_wide.columns if isinstance(col, (int, np.integer))]
        if not numeric_columns:
            continue

        min_layer = min(numeric_columns)
        max_layer = max(numeric_columns)

        min_layer_cumulative = cumulative_returns(daily_layer_returns_wide[min_layer])
        max_layer_cumulative = cumulative_returns(daily_layer_returns_wide[max_layer])

        if direction_mode == "top_minus_bottom":
            long_short_returns = daily_layer_returns_wide[max_layer] - daily_layer_returns_wide[min_layer]
        elif direction_mode == "by_ic_sign":
            ic_sign = dict(ic_signs_override or {}).get(str(factor))
            if (
                ic_sign is None
                and ic_summary_df is not None
                and not ic_summary_df.empty
                and "factor" in ic_summary_df.columns
            ):
                row = ic_summary_df[ic_summary_df["factor"] == factor]
                if not row.empty:
                    ic_sign = row.iloc[0]["ic_mean"]
            if ic_sign is None or pd.isna(ic_sign):
                get_logger().warning(
                    "[calculate_long_short_metrics] by_ic_sign requested but missing ic_mean for factor=%s; fallback to auto_by_final_return",
                    factor,
                )
                direction = "auto"
            else:
                direction = "top" if ic_sign >= 0 else "bottom"
            if direction == "top":
                long_short_returns = daily_layer_returns_wide[max_layer] - daily_layer_returns_wide[min_layer]
            elif direction == "bottom":
                long_short_returns = daily_layer_returns_wide[min_layer] - daily_layer_returns_wide[max_layer]
            else:
                long_short_returns = (
                    daily_layer_returns_wide[min_layer] - daily_layer_returns_wide[max_layer]
                    if min_layer_cumulative.iloc[-1] > max_layer_cumulative.iloc[-1]
                    else daily_layer_returns_wide[max_layer] - daily_layer_returns_wide[min_layer]
                )
        else:  # auto_by_final_return (deprecated)
            get_logger().warning(
                "[calculate_long_short_metrics] direction_mode='auto_by_final_return' is deprecated "
                "due to look-ahead bias. Use 'by_ic_sign' with ic_summary_df instead."
            )
            long_short_returns = (
                daily_layer_returns_wide[min_layer] - daily_layer_returns_wide[max_layer]
                if min_layer_cumulative.iloc[-1] > max_layer_cumulative.iloc[-1]
                else daily_layer_returns_wide[max_layer] - daily_layer_returns_wide[min_layer]
            )

        effective_gross = float(gross_exposure) if gross_exposure and gross_exposure > 0 else 2.0
        strategy_returns = long_short_returns / effective_gross

        long_short_metrics[factor] = calculate_risk_metrics(
            strategy_returns,
            period=period,
            sharpe_penalty_divisor=sharpe_penalty_divisor,
        )

        if include_long_short_visualization:
            long_short_df = strategy_returns.reset_index()
            long_short_df["layer"] = "long_short"
            long_short_df = long_short_df.rename(columns={0: return_col})
            layer_results_for_visualization[factor] = pd.concat([daily_layer_returns, long_short_df], ignore_index=True)
        else:
            layer_results_for_visualization[factor] = daily_layer_returns

    return long_short_metrics, layer_results_for_visualization


def calculate_turnover_rate(layer_results: dict[str, pd.DataFrame], period: int = 1) -> dict[str, pd.DataFrame]:
    """Calculate turnover for top and bottom layer holdings."""
    turnover_results: dict[str, pd.DataFrame] = {}

    for factor, data in layer_results.items():
        holdings_by_date: dict[object, tuple[set, set]] = {}
        for date, g in data.groupby("trade_date", sort=True):
            min_layer = g["layer"].min()
            max_layer = g["layer"].max()
            min_stocks = set(g[g["layer"] == min_layer]["znz_code"])
            max_stocks = set(g[g["layer"] == max_layer]["znz_code"])
            holdings_by_date[date] = (min_stocks, max_stocks)

        dates = sorted(holdings_by_date.keys())
        comparison_step = 1
        if len(dates) <= comparison_step:
            continue

        turnover_rates = []
        for i in range(comparison_step, len(dates)):
            current_date = dates[i]
            previous_date = dates[i - comparison_step]

            current_min_stocks, current_max_stocks = holdings_by_date[current_date]
            previous_min_stocks, previous_max_stocks = holdings_by_date[previous_date]

            min_turnover = (
                len(previous_min_stocks - current_min_stocks) / len(previous_min_stocks) if previous_min_stocks else 0
            )
            max_turnover = (
                len(previous_max_stocks - current_max_stocks) / len(previous_max_stocks) if previous_max_stocks else 0
            )

            turnover_rates.append(
                {
                    "trade_date": current_date,
                    "min_layer_turnover": min_turnover,
                    "max_layer_turnover": max_turnover,
                    "min_layer_count": len(previous_min_stocks),
                    "max_layer_count": len(previous_max_stocks),
                }
            )

        if turnover_rates:
            turnover_results[factor] = pd.DataFrame(turnover_rates)

    return turnover_results


def calculate_best_layer_metrics(
    layer_results: dict[str, pd.DataFrame],
    ic_summary_df: pd.DataFrame | None = None,
    period: int = 1,
    ic_signs_override: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Calculate long-only metrics for the layer selected by IC sign."""
    ic_signs = _ic_sign_map(ic_summary_df)
    ic_signs.update({str(k): float(v) for k, v in dict(ic_signs_override or {}).items()})
    rows: list[dict[str, object]] = []
    for factor, data in layer_results.items():
        if data is None or data.empty:
            continue
        return_col = infer_return_column_from_layer_frame(data)
        numeric_layers = sorted([x for x in data["layer"].dropna().unique() if isinstance(x, (int, np.integer))])
        if not numeric_layers:
            continue
        min_layer = int(min(numeric_layers))
        max_layer = int(max(numeric_layers))
        ic_sign = ic_signs.get(str(factor), 1.0)
        best_layer = max_layer if ic_sign >= 0 else min_layer

        layer_returns = data.groupby(["trade_date", "layer"], sort=True)[return_col].mean().reset_index()
        wide = layer_returns.pivot(index="trade_date", columns="layer", values=return_col).sort_index()
        best_returns = pd.to_numeric(wide.get(best_layer, pd.Series(dtype=float)), errors="coerce")
        universe_returns = pd.to_numeric(data.groupby("trade_date", sort=True)[return_col].mean(), errors="coerce")
        aligned = pd.concat([best_returns.rename("best"), universe_returns.rename("universe")], axis=1).dropna(
            how="all"
        )
        best_metrics = (
            calculate_risk_metrics(aligned["best"], period=period)
            if "best" in aligned
            else calculate_risk_metrics(pd.Series(dtype=float), period=period)
        )
        universe_metrics = (
            calculate_risk_metrics(aligned["universe"], period=period)
            if "universe" in aligned
            else calculate_risk_metrics(pd.Series(dtype=float), period=period)
        )
        excess = aligned["best"].fillna(0.0) - aligned["universe"].fillna(0.0)
        excess_metrics = calculate_risk_metrics(excess, period=period)
        month_ratio = _positive_month_ratio(aligned["best"], period=period)

        rows.append(
            {
                "factor": str(factor),
                "best_layer_label": int(best_layer),
                "best_layer_direction": "top" if best_layer == max_layer else "bottom",
                "best_layer_total_return": best_metrics.get("total_return", np.nan),
                "best_layer_annualized_return": best_metrics.get("annualized_return", np.nan),
                "best_layer_volatility": best_metrics.get("volatility", np.nan),
                "best_layer_sharpe": best_metrics.get("sharpe_ratio", np.nan),
                "best_layer_max_drawdown": best_metrics.get("max_drawdown", np.nan),
                "best_layer_fitness_ratio": best_metrics.get("fitness_ratio", np.nan),
                "universe_equal_weight_annualized_return": universe_metrics.get("annualized_return", np.nan),
                "best_minus_universe_annualized_return": excess_metrics.get("annualized_return", np.nan),
                "best_layer_positive_month_ratio": month_ratio,
            }
        )
    return pd.DataFrame(rows)


def calculate_long_only_portfolio_turnover(
    layer_results: dict[str, pd.DataFrame],
    ic_summary_df: pd.DataFrame | None = None,
    period: int = 1,
    apply_tradability_constraints: bool = False,
    tradability_mode: str = "entry_exit",
    can_buy_col: str = "can_buy",
    can_sell_col: str = "can_sell",
    transaction_cost_config: TransactionCostConfig | None = None,
    ic_signs_override: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calculate equal-weight best-layer long-only portfolio turnover.

    Turnover is 0.5 * sum(abs(target_weight - drifted_previous_weight)).
    The 0.5 converts two-sided absolute weight change into one-way traded
    notional as a fraction of portfolio gross exposure.
    """
    cost_cfg = transaction_cost_config or TransactionCostConfig()
    has_net = bool(cost_cfg.enabled and cost_cfg.apply_to_long_only)
    ic_signs = _ic_sign_map(ic_summary_df)
    ic_signs.update({str(k): float(v) for k, v in dict(ic_signs_override or {}).items()})
    out: dict[str, pd.DataFrame] = {}
    for factor, data in layer_results.items():
        if data is None or data.empty:
            continue
        return_col = infer_return_column_from_layer_frame(data)
        numeric_layers = sorted([x for x in data["layer"].dropna().unique() if isinstance(x, (int, np.integer))])
        if not numeric_layers:
            continue
        best_layer = int(max(numeric_layers) if ic_signs.get(str(factor), 1.0) >= 0 else min(numeric_layers))
        data_work, full_frames_by_date = _group_frames_by_trade_date(data)
        work = data_work[data_work["layer"] == best_layer].copy()
        if work.empty:
            continue
        rows: list[dict[str, object]] = []
        prev_weights: dict[str, float] | None = None
        prev_returns: dict[str, float] = {}
        for date, g in work.groupby("trade_date", sort=True):
            full_date_frame = full_frames_by_date.get(date, data_work.iloc[0:0])
            candidate_codes = g["znz_code"].astype(str).tolist()
            prev_code_set = set(prev_weights or {})
            if apply_tradability_constraints:
                buy_allowed = _trade_allowed_map(g, can_buy_col, default=True)
                desired_codes = [
                    code for code in candidate_codes if code in prev_code_set or bool(buy_allowed.get(code, True))
                ]
                blocked_buy_count = len(
                    [
                        code
                        for code in candidate_codes
                        if code not in prev_code_set and not bool(buy_allowed.get(code, True))
                    ]
                )
            else:
                desired_codes = list(candidate_codes)
                blocked_buy_count = 0
            if (
                prev_weights is not None
                and apply_tradability_constraints
                and str(tradability_mode or "entry_exit") == "entry_exit"
            ):
                sell_allowed = _trade_allowed_map(full_date_frame, can_sell_col, default=True)
                blocked_sell_codes = [
                    code
                    for code in prev_weights
                    if code not in set(candidate_codes) and not bool(sell_allowed.get(code, True))
                ]
            else:
                blocked_sell_codes = []
            codes = list(dict.fromkeys(desired_codes + blocked_sell_codes))
            n = len(codes)
            if n == 0:
                continue
            target = {code: 1.0 / n for code in codes}
            return_map = _return_map_for_codes(full_date_frame, return_col)
            portfolio_return = float(
                sum(float(target.get(code, 0.0)) * float(return_map.get(code, 0.0)) for code in codes)
            )
            raw_n = len(candidate_codes)
            raw_portfolio_return = (
                float(sum(float(return_map.get(code, 0.0)) for code in candidate_codes) / raw_n)
                if raw_n > 0
                else np.nan
            )
            blocked_buy_ratio = float(blocked_buy_count / raw_n) if raw_n > 0 else 0.0
            prev_n = len(prev_weights or {})
            blocked_sell_ratio = float(len(blocked_sell_codes) / prev_n) if prev_n > 0 else 0.0
            tradability_return_drag = (
                float(raw_portfolio_return - portfolio_return)
                if apply_tradability_constraints and np.isfinite(raw_portfolio_return)
                else 0.0
            )
            turnover_parts = _weight_turnover_components(
                target_weights=target,
                prev_weights=prev_weights,
                prev_returns=prev_returns,
                charge_initial_position=bool(cost_cfg.charge_initial_position),
            )
            turnover = float(turnover_parts["turnover"])
            buy_turnover = float(turnover_parts["buy_turnover"])
            sell_turnover = float(turnover_parts["sell_turnover"])
            transaction_cost = (
                _transaction_cost_for_date(
                    buy_turnover=buy_turnover,
                    sell_turnover=sell_turnover,
                    config=cost_cfg,
                )
                if has_net
                else np.nan
            )
            portfolio_return_net = portfolio_return - transaction_cost if has_net else np.nan
            rows.append(
                {
                    "trade_date": date,
                    "factor": str(factor),
                    "best_layer_label": best_layer,
                    "turnover_long_only": float(turnover),
                    "buy_turnover_long_only": buy_turnover,
                    "sell_turnover_long_only": sell_turnover,
                    "portfolio_return_long_only": portfolio_return,
                    "transaction_cost_long_only": transaction_cost,
                    "portfolio_return_long_only_net": portfolio_return_net,
                    "raw_portfolio_return_long_only": raw_portfolio_return,
                    "tradability_return_drag": tradability_return_drag,
                    "holding_count": int(n),
                    "blocked_buy_count": int(blocked_buy_count),
                    "blocked_sell_count": int(len(blocked_sell_codes)),
                    "blocked_buy_ratio": blocked_buy_ratio,
                    "blocked_sell_ratio": blocked_sell_ratio,
                    "tradability_mode": str(tradability_mode or ""),
                }
            )
            prev_weights = target
            prev_returns = {code: float(return_map.get(code, 0.0)) for code in codes}
        if rows:
            out[str(factor)] = pd.DataFrame(rows)
    return out


def _trade_allowed_map(frame: pd.DataFrame, column: str, default: bool = True) -> dict[str, bool]:
    if frame is None or frame.empty or column not in frame.columns:
        return {str(code): bool(default) for code in frame.get("znz_code", pd.Series(dtype=str)).astype(str).tolist()}
    out: dict[str, bool] = {}
    for code, value in zip(frame["znz_code"].astype(str), frame[column]):
        if pd.isna(value):
            out[str(code)] = bool(default)
        else:
            out[str(code)] = str(value).strip().lower() not in {
                "0",
                "false",
                "no",
                "nan",
            }
    return out


def _return_map_for_codes(frame: pd.DataFrame, return_col: str) -> dict[str, float]:
    if frame is None or frame.empty or return_col not in frame.columns:
        return {}
    out: dict[str, float] = {}
    for code, value in zip(frame["znz_code"].astype(str), pd.to_numeric(frame[return_col], errors="coerce")):
        out[str(code)] = float(value) if pd.notna(value) else 0.0
    return out


def _weight_turnover_components(
    *,
    target_weights: dict[str, float],
    prev_weights: dict[str, float] | None,
    prev_returns: dict[str, float] | None,
    charge_initial_position: bool = False,
) -> dict[str, float]:
    if prev_weights is None:
        if not charge_initial_position:
            return {"turnover": 0.0, "buy_turnover": 0.0, "sell_turnover": 0.0}
        buy = float(sum(max(float(weight), 0.0) for weight in target_weights.values()))
        sell = float(sum(max(-float(weight), 0.0) for weight in target_weights.values()))
        return {
            "turnover": 0.5 * (buy + sell),
            "buy_turnover": buy,
            "sell_turnover": sell,
        }

    drifted = {
        code: float(weight) * (1.0 + float((prev_returns or {}).get(code, 0.0)))
        for code, weight in prev_weights.items()
    }
    gross = float(sum(abs(value) for value in drifted.values()))
    if gross > 0:
        drifted = {code: value / gross for code, value in drifted.items()}
    all_codes = set(target_weights) | set(drifted)
    buy = 0.0
    sell = 0.0
    for code in all_codes:
        diff = float(target_weights.get(code, 0.0)) - float(drifted.get(code, 0.0))
        if diff > 0:
            buy += diff
        elif diff < 0:
            sell += -diff
    return {
        "turnover": 0.5 * (buy + sell),
        "buy_turnover": float(buy),
        "sell_turnover": float(sell),
    }


def _cost_rates_for_config(config: TransactionCostConfig | None) -> tuple[float, float]:
    cfg = config or TransactionCostConfig()
    buy_bps = 0.0
    sell_bps = 0.0
    if cfg.include_commission:
        buy_bps += float(cfg.commission_bps_per_side)
        sell_bps += float(cfg.commission_bps_per_side)
    if cfg.include_slippage:
        buy_bps += float(cfg.slippage_bps_per_side)
        sell_bps += float(cfg.slippage_bps_per_side)
    if cfg.include_transfer_fee:
        buy_bps += float(cfg.transfer_fee_bps_per_side)
        sell_bps += float(cfg.transfer_fee_bps_per_side)
    if cfg.include_exchange_fee:
        buy_bps += float(cfg.exchange_fee_bps_per_side)
        sell_bps += float(cfg.exchange_fee_bps_per_side)
    if cfg.include_regulatory_fee:
        buy_bps += float(cfg.regulatory_fee_bps_per_side)
        sell_bps += float(cfg.regulatory_fee_bps_per_side)
    if cfg.include_stamp_tax:
        sell_bps += float(cfg.stamp_tax_bps_sell)
    return buy_bps / 10000.0, sell_bps / 10000.0


def _transaction_cost_for_date(
    *,
    buy_turnover: float,
    sell_turnover: float,
    config: TransactionCostConfig | None,
) -> float:
    cfg = config or TransactionCostConfig()
    if not cfg.enabled:
        return 0.0
    buy_rate, sell_rate = _cost_rates_for_config(cfg)
    return float(float(buy_turnover) * buy_rate + float(sell_turnover) * sell_rate)


def _group_frames_by_trade_date(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.DataFrame]]:
    """Normalize trade dates once and cache per-date frames for hot portfolio loops."""
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    grouped = {date: group for date, group in work.groupby("trade_date", sort=False)}
    return work, grouped


def summarize_long_only_turnover(
    turnover_results: dict[str, pd.DataFrame], factors: list[str] | None = None
) -> pd.DataFrame:
    factor_list = [str(x) for x in (factors or turnover_results.keys())]
    rows: list[dict[str, object]] = []
    for factor in factor_list:
        frame = turnover_results.get(factor)
        if frame is None or frame.empty:
            rows.append(
                {
                    "factor": factor,
                    "turnover_long_only_mean": np.nan,
                    "turnover_long_only_median": np.nan,
                    "turnover_long_only_p90": np.nan,
                    "portfolio_return_long_only_sum": np.nan,
                    "blocked_buy_ratio": np.nan,
                    "blocked_sell_ratio": np.nan,
                    "tradability_return_drag": np.nan,
                }
            )
            continue
        turnover = pd.to_numeric(frame["turnover_long_only"], errors="coerce")
        returns = pd.to_numeric(frame["portfolio_return_long_only"], errors="coerce")
        blocked_buy_ratio = pd.to_numeric(frame.get("blocked_buy_ratio", pd.Series(dtype=float)), errors="coerce")
        blocked_sell_ratio = pd.to_numeric(frame.get("blocked_sell_ratio", pd.Series(dtype=float)), errors="coerce")
        return_drag = pd.to_numeric(
            frame.get("tradability_return_drag", pd.Series(dtype=float)),
            errors="coerce",
        )
        rows.append(
            {
                "factor": factor,
                "turnover_long_only_mean": float(turnover.mean()),
                "turnover_long_only_median": float(turnover.median()),
                "turnover_long_only_p90": float(turnover.quantile(0.90)),
                "portfolio_return_long_only_sum": float(returns.sum(skipna=True)),
                "blocked_buy_ratio": float(blocked_buy_ratio.mean(skipna=True))
                if blocked_buy_ratio.notna().any()
                else np.nan,
                "blocked_sell_ratio": float(blocked_sell_ratio.mean(skipna=True))
                if blocked_sell_ratio.notna().any()
                else np.nan,
                "tradability_return_drag": float(return_drag.sum(skipna=True)) if return_drag.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calculate_long10_portfolio_returns(
    layer_results: dict[str, pd.DataFrame],
    ic_summary_df: pd.DataFrame | None = None,
    top_n: int = 10,
    period: int = 1,
    apply_tradability_constraints: bool = False,
    tradability_mode: str = "entry_exit",
    can_buy_col: str = "can_buy",
    can_sell_col: str = "can_sell",
    transaction_cost_config: TransactionCostConfig | None = None,
    ic_signs_override: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate an equal-weight long-only top-N portfolio following IC sign."""
    del period
    cost_cfg = transaction_cost_config or TransactionCostConfig()
    has_net = bool(cost_cfg.enabled and cost_cfg.apply_to_long10)
    count = max(1, int(top_n))
    ic_signs = _ic_sign_map(ic_summary_df)
    ic_signs.update({str(k): float(v) for k, v in dict(ic_signs_override or {}).items()})
    out: dict[str, pd.DataFrame] = {}
    for factor, data in layer_results.items():
        factor_col = str(factor)
        if data is None or data.empty or factor_col not in data.columns:
            continue
        return_col = infer_return_column_from_layer_frame(data)
        work, full_frames_by_date = _group_frames_by_trade_date(data)
        rows: list[dict[str, object]] = []
        prev_weights: dict[str, float] | None = None
        prev_returns: dict[str, float] = {}
        ic_sign = ic_signs.get(factor_col, 1.0)
        ascending_signal = bool(ic_sign < 0)
        direction = "bottom" if ascending_signal else "top"

        for date, g in work.groupby("trade_date", sort=True):
            full_date_frame = full_frames_by_date.get(date, work.iloc[0:0])
            ranked = g.dropna(subset=[factor_col]).copy()
            if ranked.empty:
                continue
            ranked["znz_code"] = ranked["znz_code"].astype(str)
            ranked = ranked.sort_values(
                [factor_col, "znz_code"],
                ascending=[ascending_signal, True],
                kind="mergesort",
            )
            candidates = ranked.head(count).copy()
            candidate_codes = candidates["znz_code"].astype(str).tolist()
            prev_code_set = set(prev_weights or {})

            if apply_tradability_constraints:
                buy_allowed = _trade_allowed_map(candidates, can_buy_col, default=True)
                desired_codes = [
                    code for code in candidate_codes if code in prev_code_set or bool(buy_allowed.get(code, True))
                ]
                blocked_buy_count = len(
                    [
                        code
                        for code in candidate_codes
                        if code not in prev_code_set and not bool(buy_allowed.get(code, True))
                    ]
                )
            else:
                desired_codes = list(candidate_codes)
                blocked_buy_count = 0

            if (
                prev_weights is not None
                and apply_tradability_constraints
                and str(tradability_mode or "entry_exit") == "entry_exit"
            ):
                sell_allowed = _trade_allowed_map(full_date_frame, can_sell_col, default=True)
                blocked_sell_codes = [
                    code
                    for code in prev_weights
                    if code not in set(candidate_codes) and not bool(sell_allowed.get(code, True))
                ]
            else:
                blocked_sell_codes = []

            codes = list(dict.fromkeys(desired_codes + blocked_sell_codes))
            n = len(codes)
            if n == 0:
                continue
            target = {code: 1.0 / n for code in codes}
            return_map = _return_map_for_codes(full_date_frame, return_col)
            portfolio_return = float(
                sum(float(target.get(code, 0.0)) * float(return_map.get(code, 0.0)) for code in codes)
            )
            raw_n = len(candidate_codes)
            raw_portfolio_return = (
                float(sum(float(return_map.get(code, 0.0)) for code in candidate_codes) / raw_n)
                if raw_n > 0
                else np.nan
            )
            blocked_buy_ratio = float(blocked_buy_count / raw_n) if raw_n > 0 else 0.0
            prev_n = len(prev_weights or {})
            blocked_sell_ratio = float(len(blocked_sell_codes) / prev_n) if prev_n > 0 else 0.0
            tradability_return_drag = (
                float(raw_portfolio_return - portfolio_return)
                if apply_tradability_constraints and np.isfinite(raw_portfolio_return)
                else 0.0
            )
            turnover_parts = _weight_turnover_components(
                target_weights=target,
                prev_weights=prev_weights,
                prev_returns=prev_returns,
                charge_initial_position=bool(cost_cfg.charge_initial_position),
            )
            turnover = float(turnover_parts["turnover"])
            buy_turnover = float(turnover_parts["buy_turnover"])
            sell_turnover = float(turnover_parts["sell_turnover"])
            transaction_cost = (
                _transaction_cost_for_date(
                    buy_turnover=buy_turnover,
                    sell_turnover=sell_turnover,
                    config=cost_cfg,
                )
                if has_net
                else np.nan
            )
            portfolio_return_net = portfolio_return - transaction_cost if has_net else np.nan

            rows.append(
                {
                    "trade_date": date,
                    "factor": factor_col,
                    "long10_direction": direction,
                    "long10_count": int(count),
                    "turnover_long10": float(turnover),
                    "buy_turnover_long10": buy_turnover,
                    "sell_turnover_long10": sell_turnover,
                    "portfolio_return_long10": portfolio_return,
                    "transaction_cost_long10": transaction_cost,
                    "portfolio_return_long10_net": portfolio_return_net,
                    "raw_portfolio_return_long10": raw_portfolio_return,
                    "tradability_return_drag_long10": tradability_return_drag,
                    "holding_count_long10": int(n),
                    "blocked_buy_count_long10": int(blocked_buy_count),
                    "blocked_sell_count_long10": int(len(blocked_sell_codes)),
                    "blocked_buy_ratio_long10": blocked_buy_ratio,
                    "blocked_sell_ratio_long10": blocked_sell_ratio,
                    "tradability_mode": str(tradability_mode or ""),
                }
            )
            prev_weights = target
            prev_returns = {code: float(return_map.get(code, 0.0)) for code in codes}
        if rows:
            out[factor_col] = pd.DataFrame(rows)
    return out


def summarize_long10_portfolio_returns(
    long10_results: dict[str, pd.DataFrame],
    factors: list[str] | None = None,
    period: int = 1,
) -> pd.DataFrame:
    factor_list = [str(x) for x in (factors or long10_results.keys())]
    rows: list[dict[str, object]] = []
    for factor in factor_list:
        frame = long10_results.get(factor)
        if frame is None or frame.empty:
            rows.append(
                {
                    "factor": factor,
                    "long10_total_return": np.nan,
                    "long10_annualized_return": np.nan,
                    "long10_volatility": np.nan,
                    "long10_sharpe_ratio": np.nan,
                    "long10_max_drawdown": np.nan,
                    "long10_fitness_ratio": np.nan,
                    "portfolio_return_long10_sum": np.nan,
                    "turnover_long10_mean": np.nan,
                    "holding_count_long10_mean": np.nan,
                    "blocked_buy_ratio_long10": np.nan,
                    "blocked_sell_ratio_long10": np.nan,
                    "tradability_return_drag_long10": np.nan,
                }
            )
            continue
        returns = pd.to_numeric(frame["portfolio_return_long10"], errors="coerce")
        turnover = pd.to_numeric(frame.get("turnover_long10", pd.Series(dtype=float)), errors="coerce")
        holding = pd.to_numeric(frame.get("holding_count_long10", pd.Series(dtype=float)), errors="coerce")
        blocked_buy_ratio = pd.to_numeric(
            frame.get("blocked_buy_ratio_long10", pd.Series(dtype=float)),
            errors="coerce",
        )
        blocked_sell_ratio = pd.to_numeric(
            frame.get("blocked_sell_ratio_long10", pd.Series(dtype=float)),
            errors="coerce",
        )
        return_drag = pd.to_numeric(
            frame.get("tradability_return_drag_long10", pd.Series(dtype=float)),
            errors="coerce",
        )
        metrics = calculate_risk_metrics(returns, period=period)
        rows.append(
            {
                "factor": factor,
                "long10_total_return": metrics.get("total_return", np.nan),
                "long10_annualized_return": metrics.get("annualized_return", np.nan),
                "long10_volatility": metrics.get("volatility", np.nan),
                "long10_sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
                "long10_max_drawdown": metrics.get("max_drawdown", np.nan),
                "long10_fitness_ratio": metrics.get("fitness_ratio", np.nan),
                "portfolio_return_long10_sum": float(returns.sum(skipna=True)),
                "turnover_long10_mean": float(turnover.mean(skipna=True)) if turnover.notna().any() else np.nan,
                "holding_count_long10_mean": float(holding.mean(skipna=True)) if holding.notna().any() else np.nan,
                "blocked_buy_ratio_long10": float(blocked_buy_ratio.mean(skipna=True))
                if blocked_buy_ratio.notna().any()
                else np.nan,
                "blocked_sell_ratio_long10": float(blocked_sell_ratio.mean(skipna=True))
                if blocked_sell_ratio.notna().any()
                else np.nan,
                "tradability_return_drag_long10": float(return_drag.sum(skipna=True))
                if return_drag.notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calculate_layer_portfolio_turnover(
    layer_results: dict[str, pd.DataFrame],
    period: int = 1,
    transaction_cost_config: TransactionCostConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate weight-level turnover and optional net returns for every numeric layer portfolio."""
    del period
    cost_cfg = transaction_cost_config or TransactionCostConfig()
    has_net = bool(cost_cfg.enabled and cost_cfg.apply_to_layers)
    out: dict[str, pd.DataFrame] = {}
    for factor, data in layer_results.items():
        if data is None or data.empty:
            continue
        return_col = infer_return_column_from_layer_frame(data)
        work, _ = _group_frames_by_trade_date(data)
        rows: list[dict[str, object]] = []
        numeric_layers = sorted([x for x in work["layer"].dropna().unique() if isinstance(x, (int, np.integer))])
        for layer in numeric_layers:
            layer_frame = work[work["layer"] == layer].copy()
            if layer_frame.empty:
                continue
            prev_weights: dict[str, float] | None = None
            prev_returns: dict[str, float] = {}
            for date, g in layer_frame.groupby("trade_date", sort=True):
                codes = g["znz_code"].astype(str).tolist()
                n = len(codes)
                if n == 0:
                    continue
                target = {code: 1.0 / n for code in codes}
                return_map = _return_map_for_codes(g, return_col)
                portfolio_return = float(
                    sum(float(target.get(code, 0.0)) * float(return_map.get(code, 0.0)) for code in codes)
                )
                turnover_parts = _weight_turnover_components(
                    target_weights=target,
                    prev_weights=prev_weights,
                    prev_returns=prev_returns,
                    charge_initial_position=bool(cost_cfg.charge_initial_position),
                )
                buy_turnover = float(turnover_parts["buy_turnover"])
                sell_turnover = float(turnover_parts["sell_turnover"])
                transaction_cost = (
                    _transaction_cost_for_date(
                        buy_turnover=buy_turnover,
                        sell_turnover=sell_turnover,
                        config=cost_cfg,
                    )
                    if has_net
                    else np.nan
                )
                rows.append(
                    {
                        "trade_date": date,
                        "factor": str(factor),
                        "layer": int(layer),
                        "portfolio": f"layer_{int(layer)}",
                        "holding_count_layer": int(n),
                        "portfolio_return_layer": portfolio_return,
                        "turnover_layer": float(turnover_parts["turnover"]),
                        "buy_turnover_layer": buy_turnover,
                        "sell_turnover_layer": sell_turnover,
                        "transaction_cost_layer": transaction_cost,
                        "portfolio_return_layer_net": portfolio_return - transaction_cost if has_net else np.nan,
                    }
                )
                prev_weights = target
                prev_returns = {code: float(return_map.get(code, 0.0)) for code in codes}
        if rows:
            out[str(factor)] = pd.DataFrame(rows)
    return out


def build_portfolio_pnl_table(
    layer_results_for_visualization: dict[str, pd.DataFrame],
    long_only_turnover_results: dict[str, pd.DataFrame] | None = None,
    long10_portfolio_returns: dict[str, pd.DataFrame] | None = None,
    turnover_results: dict[str, pd.DataFrame] | None = None,
    layer_turnover_results: dict[str, pd.DataFrame] | None = None,
    transaction_cost_config: TransactionCostConfig | None = None,
) -> pd.DataFrame:
    columns = [
        "factor",
        "trade_date",
        "portfolio",
        "return",
        "cum_return",
        "return_gross",
        "cum_return_gross",
        "transaction_cost",
        "return_net",
        "cum_return_net",
        "has_net_pnl",
        "cost_model",
        "holding_count",
        "turnover",
        "buy_turnover",
        "sell_turnover",
        "blocked_buy_ratio",
        "blocked_sell_ratio",
        "tradability_return_drag",
    ]
    rows: list[dict[str, object]] = []
    cost_cfg = transaction_cost_config or TransactionCostConfig()
    cost_model = str(cost_cfg.model_name or "") if cost_cfg.enabled else None

    def append_row(
        *,
        factor: object,
        trade_date: object,
        portfolio: str,
        return_gross: object,
        holding_count: object = np.nan,
        turnover: object = np.nan,
        buy_turnover: object = np.nan,
        sell_turnover: object = np.nan,
        transaction_cost: object = np.nan,
        return_net: object = np.nan,
        has_net_pnl: bool = False,
        blocked_buy_ratio: object = np.nan,
        blocked_sell_ratio: object = np.nan,
        tradability_return_drag: object = np.nan,
    ) -> None:
        rows.append(
            {
                "factor": str(factor),
                "trade_date": trade_date,
                "portfolio": portfolio,
                "return": return_gross,
                "return_gross": return_gross,
                "transaction_cost": transaction_cost,
                "return_net": return_net if has_net_pnl else np.nan,
                "has_net_pnl": bool(has_net_pnl),
                "cost_model": cost_model if has_net_pnl else None,
                "holding_count": holding_count,
                "turnover": turnover,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "blocked_buy_ratio": blocked_buy_ratio,
                "blocked_sell_ratio": blocked_sell_ratio,
                "tradability_return_drag": tradability_return_drag,
            }
        )

    for factor, frame in (layer_results_for_visualization or {}).items():
        if frame is None or frame.empty:
            continue
        return_col = infer_return_column_from_layer_frame(frame)
        has_layer_turnover = (
            layer_turnover_results is not None
            and layer_turnover_results.get(factor) is not None
            and not layer_turnover_results.get(factor).empty
        )
        long_short_turnover_by_date: dict[object, float] = {}
        turnover_frame = (turnover_results or {}).get(factor)
        if turnover_frame is not None and not turnover_frame.empty:
            required_turnover_cols = {
                "trade_date",
                "min_layer_turnover",
                "max_layer_turnover",
            }
            if required_turnover_cols.issubset(set(turnover_frame.columns)):
                turnover_work = turnover_frame.copy()
                min_turnover = pd.to_numeric(turnover_work["min_layer_turnover"], errors="coerce")
                max_turnover = pd.to_numeric(turnover_work["max_layer_turnover"], errors="coerce")
                turnover_work["long_short_turnover"] = (min_turnover + max_turnover) / 2.0
                long_short_turnover_by_date = dict(
                    zip(
                        turnover_work["trade_date"],
                        turnover_work["long_short_turnover"],
                    )
                )
        for _, row in frame.iterrows():
            layer = row.get("layer")
            portfolio = "long_short" if str(layer) == "long_short" else f"layer_{layer}"
            if has_layer_turnover and portfolio != "long_short":
                continue
            append_row(
                factor=factor,
                trade_date=row.get("trade_date"),
                portfolio=portfolio,
                return_gross=row.get(return_col),
                turnover=long_short_turnover_by_date.get(row.get("trade_date"), np.nan)
                if portfolio == "long_short"
                else np.nan,
            )

    for factor, frame in (layer_turnover_results or {}).items():
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            return_net = row.get("portfolio_return_layer_net")
            has_net = pd.notna(return_net)
            append_row(
                factor=factor,
                trade_date=row.get("trade_date"),
                portfolio=str(row.get("portfolio") or f"layer_{row.get('layer')}"),
                return_gross=row.get("portfolio_return_layer"),
                holding_count=row.get("holding_count_layer"),
                turnover=row.get("turnover_layer"),
                buy_turnover=row.get("buy_turnover_layer"),
                sell_turnover=row.get("sell_turnover_layer"),
                transaction_cost=row.get("transaction_cost_layer"),
                return_net=return_net,
                has_net_pnl=bool(has_net),
            )

    for factor, frame in (long_only_turnover_results or {}).items():
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            return_net = row.get("portfolio_return_long_only_net")
            append_row(
                factor=factor,
                trade_date=row.get("trade_date"),
                portfolio="long_only",
                return_gross=row.get("portfolio_return_long_only"),
                holding_count=row.get("holding_count"),
                turnover=row.get("turnover_long_only"),
                buy_turnover=row.get("buy_turnover_long_only", row.get("turnover_long_only")),
                sell_turnover=row.get("sell_turnover_long_only", row.get("turnover_long_only")),
                transaction_cost=row.get("transaction_cost_long_only"),
                return_net=return_net,
                has_net_pnl=bool(pd.notna(return_net)),
                blocked_buy_ratio=row.get("blocked_buy_ratio"),
                blocked_sell_ratio=row.get("blocked_sell_ratio"),
                tradability_return_drag=row.get("tradability_return_drag"),
            )

    for factor, frame in (long10_portfolio_returns or {}).items():
        if frame is None or frame.empty:
            continue
        for _, row in frame.iterrows():
            return_net = row.get("portfolio_return_long10_net")
            append_row(
                factor=factor,
                trade_date=row.get("trade_date"),
                portfolio="long_10",
                return_gross=row.get("portfolio_return_long10"),
                holding_count=row.get("holding_count_long10"),
                turnover=row.get("turnover_long10"),
                buy_turnover=row.get("buy_turnover_long10", row.get("turnover_long10")),
                sell_turnover=row.get("sell_turnover_long10", row.get("turnover_long10")),
                transaction_cost=row.get("transaction_cost_long10"),
                return_net=return_net,
                has_net_pnl=bool(pd.notna(return_net)),
                blocked_buy_ratio=row.get("blocked_buy_ratio_long10"),
                blocked_sell_ratio=row.get("blocked_sell_ratio_long10"),
                tradability_return_drag=row.get("tradability_return_drag_long10"),
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out["factor"] = out["factor"].astype(str)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["return_gross"] = pd.to_numeric(out["return_gross"], errors="coerce")
    out["return"] = out["return_gross"]
    out["return_net"] = pd.to_numeric(out.get("return_net", pd.Series(dtype=float)), errors="coerce")
    out["has_net_pnl"] = out.get("has_net_pnl", False).fillna(False).astype(bool) & out["return_net"].notna()
    out = out.sort_values(["factor", "portfolio", "trade_date"], kind="mergesort").reset_index(drop=True)
    out["cum_return_gross"] = out.groupby(["factor", "portfolio"], sort=False)["return_gross"].transform(
        lambda s: (1.0 + pd.to_numeric(s, errors="coerce").fillna(0.0)).cumprod() - 1.0
    )
    out["cum_return"] = out["cum_return_gross"]
    out["cum_return_net"] = out.groupby(["factor", "portfolio"], sort=False)["return_net"].transform(
        lambda s: (1.0 + pd.to_numeric(s, errors="coerce").fillna(0.0)).cumprod() - 1.0
    )
    out.loc[~out["has_net_pnl"], "return_net"] = np.nan
    out.loc[~out["has_net_pnl"], "cum_return_net"] = np.nan
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns].copy()


def calculate_margin_metrics(
    long_only_turnover_results: dict[str, pd.DataFrame],
    factors: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate return per one-way turnover for long-only portfolios."""
    factor_list = [str(x) for x in (factors or long_only_turnover_results.keys())]
    rows: list[dict[str, object]] = []
    for factor in factor_list:
        frame = long_only_turnover_results.get(factor)
        if frame is None or frame.empty:
            rows.append(_empty_margin_row(factor))
            continue
        returns = pd.to_numeric(frame["portfolio_return_long_only"], errors="coerce")
        traded = pd.to_numeric(frame["turnover_long_only"], errors="coerce")
        denom = float(traded.sum(skipna=True))
        margin = float(returns.sum(skipna=True) / denom) if denom > 1e-8 else np.nan
        rows.append(
            {
                "factor": factor,
                "margin_long_only": margin,
                "margin_long_only_bp": margin * 10000.0 if np.isfinite(margin) else np.nan,
                "margin_long_only_valid": bool(np.isfinite(margin)),
                "best_layer_margin": margin,
                "margin_long_short": np.nan,
                "margin_long_short_bp": np.nan,
                "margin_long_short_valid": False,
            }
        )
    return pd.DataFrame(rows)


def _empty_margin_row(factor: str) -> dict[str, object]:
    return {
        "factor": str(factor),
        "margin_long_only": np.nan,
        "margin_long_only_bp": np.nan,
        "margin_long_only_valid": False,
        "best_layer_margin": np.nan,
        "margin_long_short": np.nan,
        "margin_long_short_bp": np.nan,
        "margin_long_short_valid": False,
    }


def _ic_sign_map(ic_summary_df: pd.DataFrame | None) -> dict[str, float]:
    if (
        ic_summary_df is None
        or ic_summary_df.empty
        or "factor" not in ic_summary_df.columns
        or "ic_mean" not in ic_summary_df.columns
    ):
        return {}
    out: dict[str, float] = {}
    for _, row in ic_summary_df.iterrows():
        val = pd.to_numeric(pd.Series([row.get("ic_mean")]), errors="coerce").iloc[0]
        out[str(row.get("factor"))] = 1.0 if pd.isna(val) or float(val) >= 0 else -1.0
    return out


def _positive_month_ratio(returns: pd.Series, period: int = 1) -> float:
    s = pd.to_numeric(returns, errors="coerce").dropna()
    if s.empty:
        return np.nan
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            return float((s > 0).mean())
    monthly = s.groupby(s.index.to_period("M")).sum()
    return float((monthly > 0).mean()) if len(monthly) else np.nan


def process_factors_individually(
    df_all: pd.DataFrame,
    factor_columns: list[str],
    market_value_column: str = "circ_mv",
    is_timeseries: bool = True,
    return_col: str = "pct_chg",
    period: int = 1,
    layers: int = 5,
    max_lag: int | None = None,
):
    """Run end-to-end single-factor workflow factor-by-factor."""
    ensure_columns(
        df_all,
        ["znz_code", "trade_date", return_col, market_value_column],
        caller="process_factors_individually",
    )

    summary_dfs = []
    long_short_metrics_dict: dict[str, dict] = {}
    layer_results_dict: dict[str, pd.DataFrame] = {}
    ic_dfs = []
    lag_analysis_results_list = []
    layer_results_for_visualization_dict: dict[str, pd.DataFrame] = {}

    active_return_col = resolve_future_return_column(df_all, return_col, period, caller="process_factors_individually")

    for factor in tqdm(factor_columns, desc="Processing factors"):
        try:
            subset_cols = [
                "znz_code",
                "trade_date",
                active_return_col,
                factor,
                market_value_column,
            ]
            df_processed = process_factor_data(df_all[subset_cols].copy(), [factor], market_value_column, is_timeseries)
            ic_results = calculate_icir(
                df_processed,
                [factor],
                return_col=return_col,
                period=period,
                max_lag=max_lag,
            )

            if max_lag is not None:
                ic_df, summary_df, lag_analysis_results = ic_results
                lag_analysis_results_list.append(
                    lag_analysis_results if isinstance(lag_analysis_results, list) else [lag_analysis_results]
                )
            else:
                ic_df, summary_df = ic_results

            summary_dfs.append(summary_df)
            ic_dfs.append(ic_df)

            layers_result = factor_layer_analysis(
                df_processed,
                [factor],
                return_col=return_col,
                period=period,
                layers=layers,
            )
            if factor in layers_result:
                layer_results_dict[factor] = layers_result[factor]
                long_short_metrics, layer_visual = calculate_long_short_metrics(
                    {factor: layers_result[factor]},
                    period=period,
                    direction_mode="by_ic_sign",
                    ic_summary_df=summary_df,
                )
                if factor in long_short_metrics:
                    long_short_metrics_dict[factor] = long_short_metrics[factor]
                if factor in layer_visual:
                    layer_results_for_visualization_dict[factor] = layer_visual[factor]
        except Exception:
            continue

    return (
        summary_dfs,
        long_short_metrics_dict,
        layer_results_dict,
        ic_dfs,
        lag_analysis_results_list,
        layer_results_for_visualization_dict,
    )
