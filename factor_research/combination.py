from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    apply_period_sampling,
    assign_quantile_labels,
    calculate_risk_metrics,
    cumulative_returns,
    ensure_columns,
    resolve_future_return_column,
)


def calculate_factor_weights(summary_df: pd.DataFrame, weighting_method: str = "icir") -> dict[str, float]:
    """Calculate factor weights with equal/sharpe/icir logic."""
    weights: dict[str, float] = {}

    if weighting_method == "equal":
        positive_factors = summary_df[summary_df["ir"] >= 0]
        negative_factors = summary_df[summary_df["ir"] < 0]

        if len(positive_factors) > 0:
            positive_weight = 1.0 / len(positive_factors)
            for _, row in positive_factors.iterrows():
                weights[row["factor"]] = positive_weight

        if len(negative_factors) > 0:
            negative_weight = -1.0 / len(negative_factors)
            for _, row in negative_factors.iterrows():
                weights[row["factor"]] = negative_weight

    elif weighting_method == "sharpe":
        positive_sharpe = summary_df["sharpe_ratio"].clip(lower=0)
        negative_sharpe = summary_df["sharpe_ratio"].clip(upper=0).abs()

        total_positive_sharpe = positive_sharpe.sum()
        total_negative_sharpe = negative_sharpe.sum()

        for _, row in summary_df.iterrows():
            factor = row["factor"]
            sharpe = row["sharpe_ratio"]
            if sharpe >= 0:
                weight = sharpe / total_positive_sharpe if total_positive_sharpe != 0 else 0.0
            else:
                weight = sharpe / total_negative_sharpe if total_negative_sharpe != 0 else 0.0
            weights[factor] = weight

        if total_positive_sharpe + total_negative_sharpe == 0:
            weights = dict(zip(summary_df["factor"], [1.0 / len(summary_df)] * len(summary_df)))

    else:
        positive_ir = summary_df["ir"].clip(lower=0)
        negative_ir = summary_df["ir"].clip(upper=0).abs()

        total_positive_ir = positive_ir.sum()
        total_negative_ir = negative_ir.sum()

        for _, row in summary_df.iterrows():
            factor = row["factor"]
            ir = row["ir"]
            if ir >= 0:
                weight = ir / total_positive_ir if total_positive_ir != 0 else 0.0
            else:
                weight = ir / total_negative_ir if total_negative_ir != 0 else 0.0
            weights[factor] = weight

        if total_positive_ir + total_negative_ir == 0:
            weights = dict(zip(summary_df["factor"], [1.0 / len(summary_df)] * len(summary_df)))

    return weights


def combine_factors_with_weights(df: pd.DataFrame, factor_cols: list[str], weights: dict[str, float]) -> pd.Series:
    """Combine factors linearly with user-provided weights."""
    standardized_df = df[factor_cols].copy()
    combined_factor = 0
    for factor in factor_cols:
        weight = weights.get(factor, 0)
        if weight != 0:
            combined_factor += standardized_df[factor] * weight
    return combined_factor


def backtest_factor_strategy(
    df: pd.DataFrame,
    combined_factor: pd.Series,
    return_col: str = "pct_chg",
    period: int = 1,
    quantiles: int = 10,
    layer_sharpe_penalty_divisor: float = 2.0,
    long_short_total_return_divisor: float = 2.0,
    long_short_sharpe_penalty_divisor: float = 1.0,
):
    """Backtest combined factor by cross-sectional quantiles."""
    ensure_columns(df, ["znz_code", "trade_date", return_col], caller="backtest_factor_strategy")

    df = df.copy()
    df["combined_factor"] = combined_factor
    df = df.sort_values(["znz_code", "trade_date"])

    active_return_col = resolve_future_return_column(df, return_col, period, caller="backtest_factor_strategy")
    df = apply_period_sampling(df, period=period, drop_tail_for_future=True)

    clean_df = df[["znz_code", "combined_factor", active_return_col, "trade_date"]].dropna()
    if len(clean_df) == 0:
        return None, pd.DataFrame(), {}

    clean_df["quantile"] = clean_df.groupby("trade_date")["combined_factor"].transform(
        lambda s: assign_quantile_labels(
            s,
            quantiles,
            labels_name="quantile",
            warn_context="backtest_factor_strategy",
        )
    )
    clean_df = clean_df.dropna(subset=["quantile"])
    if len(clean_df) == 0:
        return None, pd.DataFrame(), {}

    clean_df["quantile"] = clean_df["quantile"].astype(int)

    holdings_info = clean_df[["znz_code", "trade_date", "quantile", "combined_factor", active_return_col]].copy()
    holdings_info = holdings_info.rename(columns={"quantile": "layer"})

    portfolio_returns = clean_df.groupby(["trade_date", "quantile"])[active_return_col].mean().reset_index()
    portfolio_cumulative_returns = portfolio_returns.pivot(
        index="trade_date", columns="quantile", values=active_return_col
    ).fillna(0)
    cumulative_returns_by_layer = cumulative_returns(portfolio_cumulative_returns)

    metrics = {}
    numeric_columns = [col for col in cumulative_returns_by_layer.columns if isinstance(col, (int, np.integer))]
    if numeric_columns:
        min_layer = min(numeric_columns)
        max_layer = max(numeric_columns)

        long_short_returns = portfolio_cumulative_returns[max_layer] - portfolio_cumulative_returns[min_layer]
        long_short_curve = cumulative_returns(long_short_returns)

        for layer in numeric_columns:
            layer_returns = portfolio_cumulative_returns[layer]
            layer_metrics = calculate_risk_metrics(
                layer_returns,
                period=period,
                sharpe_penalty_divisor=layer_sharpe_penalty_divisor,
            )
            metrics[layer] = {
                "total_return": layer_metrics["total_return"],
                "annualized_return": layer_metrics["annualized_return"],
                "volatility": layer_metrics["volatility"],
                "sharpe_ratio": layer_metrics["sharpe_ratio"],
                "max_drawdown": layer_metrics["max_drawdown"],
            }

        long_short_metrics = calculate_risk_metrics(
            long_short_returns,
            period=period,
            sharpe_penalty_divisor=long_short_sharpe_penalty_divisor,
        )
        total_return_divisor = long_short_total_return_divisor if long_short_total_return_divisor > 0 else 1.0
        metrics["long_short"] = {
            "total_return": (long_short_curve.iloc[-1] if len(long_short_curve) else 0) / total_return_divisor,
            "annualized_return": long_short_metrics["annualized_return"],
            "volatility": long_short_metrics["volatility"],
            "sharpe_ratio": long_short_metrics["sharpe_ratio"],
            "max_drawdown": long_short_metrics["max_drawdown"],
        }

    return cumulative_returns_by_layer, holdings_info, metrics
