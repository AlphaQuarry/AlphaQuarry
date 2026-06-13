from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    apply_period_sampling,
    assign_quantile_labels,
    calculate_risk_metrics,
    cumulative_returns,
    ensure_columns,
    resolve_future_return_column,
)


def calculate_factor_returns(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_col: str = "pct_chg",
    period: int = 1,
    independent_by_factor: bool = True,
) -> pd.DataFrame:
    """
    Estimate cross-sectional factor returns by date-wise regression.

    When `independent_by_factor=True` (default), each factor is fit independently
    on its own valid rows (`dropna` per factor), so sparse factors do not shrink
    the sample of other factors.
    """
    ensure_columns(
        df,
        ["znz_code", "trade_date", return_col] + factor_cols,
        caller="calculate_factor_returns",
    )

    active_return_col = resolve_future_return_column(df, return_col, period, caller="calculate_factor_returns")
    sampled_df = apply_period_sampling(df, period=period, drop_tail_for_future=True)
    if sampled_df.empty:
        return pd.DataFrame(columns=["trade_date", "factor", "return"])

    sampled_df = sampled_df[["trade_date", active_return_col] + factor_cols].copy()
    sampled_df[active_return_col] = pd.to_numeric(sampled_df[active_return_col], errors="coerce")

    factor_return_rows: list[dict] = []
    if independent_by_factor:
        for factor in tqdm(factor_cols, desc="Calculating factor returns (factor-wise)"):
            work = sampled_df[["trade_date", active_return_col, factor]].copy()
            work[factor] = pd.to_numeric(work[factor], errors="coerce")
            work = work.dropna(subset=[active_return_col, factor])
            if work.empty:
                continue

            for date, date_data in work.groupby("trade_date", sort=False):
                x = np.asarray(date_data[factor].values, dtype=float)
                y = np.asarray(date_data[active_return_col].values, dtype=float)
                n = x.size
                if n <= 1:
                    continue

                x_mean = float(x.mean())
                y_mean = float(y.mean())
                x_centered = x - x_mean
                var_x = float(np.dot(x_centered, x_centered) / n)
                if var_x <= 0:
                    beta = 0.0
                else:
                    cov_xy = float(np.dot(x_centered, y - y_mean) / n)
                    beta = cov_xy / var_x

                factor_return_rows.append({"trade_date": date, "factor": factor, "return": float(beta)})
    else:
        clean_df = sampled_df.dropna(subset=[active_return_col] + factor_cols)
        for date, date_data in tqdm(
            clean_df.groupby("trade_date", sort=False),
            desc="Calculating factor returns",
        ):
            if len(date_data) <= len(factor_cols):
                continue

            X = np.asarray(date_data[factor_cols].values, dtype=float)
            y = np.asarray(date_data[active_return_col].values, dtype=float)

            try:
                X_with_const = np.column_stack([np.ones(len(X), dtype=float), X])
                beta, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
                coef = np.asarray(beta[1:], dtype=float).reshape(-1)
                if coef.size != len(factor_cols):
                    coef = np.zeros(len(factor_cols), dtype=float)
            except Exception:
                coef = np.zeros(len(factor_cols), dtype=float)

            factor_return_rows.extend(
                {"trade_date": date, "factor": factor, "return": float(coef[idx])}
                for idx, factor in enumerate(factor_cols)
            )

    if factor_return_rows:
        return pd.DataFrame(factor_return_rows)
    return pd.DataFrame(columns=["trade_date", "factor", "return"])


def calculate_ewma_factors(factor_returns_df: pd.DataFrame, window: int = 5, half_life: int = 5) -> pd.DataFrame:
    """Compute EWMA-smoothed factor return forecasts."""
    if factor_returns_df is None or factor_returns_df.empty:
        return pd.DataFrame(columns=["trade_date", "factor", "ewma_return"])

    lam = half_life
    alpha = 1 / (1 + lam)

    wide_df = factor_returns_df.pivot(index="trade_date", columns="factor", values="return").sort_index()
    if wide_df.empty:
        return pd.DataFrame(columns=["trade_date", "factor", "ewma_return"])
    ewma_values = []

    for i in range(len(wide_df)):
        current_window = min(i + 1, window)
        start_idx = max(0, i - current_window + 1)
        window_data = wide_df.iloc[start_idx : i + 1]

        if len(window_data) == 1:
            ewma_row = window_data.iloc[0]
        else:
            ewma_row = window_data.ewm(alpha=alpha, min_periods=1).mean().iloc[-1]

        ewma_values.append(ewma_row)

    ewma_factor_returns = pd.DataFrame(ewma_values, index=wide_df.index, columns=wide_df.columns).reset_index()
    ewma_factor_returns = pd.melt(
        ewma_factor_returns,
        id_vars=["trade_date"],
        var_name="factor",
        value_name="ewma_return",
    )
    return ewma_factor_returns


def predict_stock_returns(
    df: pd.DataFrame,
    factor_cols: list[str],
    ewma_factor_returns: pd.DataFrame,
    period: int = 1,
) -> pd.DataFrame:
    """Predict stock returns using current factor exposures and lagged EWMA factor returns."""
    ensure_columns(df, ["znz_code", "trade_date"] + factor_cols, caller="predict_stock_returns")
    ensure_columns(
        ewma_factor_returns,
        ["trade_date", "factor", "ewma_return"],
        caller="predict_stock_returns",
    )

    working_df = apply_period_sampling(df, period=period, drop_tail_for_future=True) if period > 1 else df.copy()
    if working_df.empty:
        return pd.DataFrame(columns=["znz_code", "trade_date", "predicted_return"])
    predictions = []

    unique_dates_sorted = sorted(working_df["trade_date"].unique())
    date_mapping = {
        unique_dates_sorted[i]: unique_dates_sorted[i - period] for i in range(period, len(unique_dates_sorted))
    }
    ewma_returns_by_date: dict[object, dict[str, float]] = {}
    for dt, g in ewma_factor_returns.groupby("trade_date", sort=False):
        factor_arr = np.asarray(g["factor"].values, dtype=object)
        ewma_arr = pd.to_numeric(g["ewma_return"], errors="coerce").values
        factor_map = {str(f): float(v) for f, v in zip(factor_arr, ewma_arr) if pd.notna(f) and pd.notna(v)}
        if factor_map:
            ewma_returns_by_date[dt] = factor_map

    for date, date_data in tqdm(working_df.groupby("trade_date", sort=False), desc="Predicting stock returns"):
        prev_date = date_mapping.get(date)
        if prev_date is None:
            continue

        factor_returns_dict = ewma_returns_by_date.get(prev_date)
        if factor_returns_dict is None:
            continue

        weights = np.asarray(
            [factor_returns_dict.get(factor, 0.0) for factor in factor_cols],
            dtype=float,
        )
        factor_exposure = np.asarray(
            date_data[factor_cols].apply(pd.to_numeric, errors="coerce").values,
            dtype=float,
        )
        factor_exposure = np.nan_to_num(factor_exposure, nan=0.0, posinf=0.0, neginf=0.0)
        predicted = factor_exposure.dot(weights)

        pred_df = date_data[["znz_code", "trade_date"]].copy()
        pred_df["predicted_return"] = predicted
        predictions.append(pred_df)

    if predictions:
        return pd.concat(predictions, ignore_index=True)
    return pd.DataFrame(columns=["znz_code", "trade_date", "predicted_return"])


def backtest_ewma_strategy(
    df: pd.DataFrame,
    stock_returns_pred: pd.DataFrame,
    quantiles: int = 10,
    period: int = 1,
    layer_sharpe_penalty_divisor: float = 2.0,
):
    """Backtest EWMA-predicted stock returns with quantile portfolios."""
    ensure_columns(df, ["znz_code", "trade_date", "pct_chg"], caller="backtest_ewma_strategy")
    ensure_columns(
        stock_returns_pred,
        ["znz_code", "trade_date", "predicted_return"],
        caller="backtest_ewma_strategy",
    )

    working_df = df.copy()
    return_col = resolve_future_return_column(working_df, "pct_chg", period, caller="backtest_ewma_strategy")
    working_df = apply_period_sampling(working_df, period=period, drop_tail_for_future=True)

    merged_df = pd.merge(
        stock_returns_pred,
        working_df[["znz_code", "trade_date", return_col]],
        on=["znz_code", "trade_date"],
        how="inner",
    )

    merged_df["quantile"] = merged_df.groupby("trade_date")["predicted_return"].transform(
        lambda s: assign_quantile_labels(s, quantiles, labels_name="quantile", warn_context="backtest_ewma_strategy")
    )
    merged_df = merged_df.dropna(subset=["quantile"])
    merged_df["quantile"] = merged_df["quantile"].astype(int)

    holdings_info = merged_df[["znz_code", "trade_date", "quantile", "predicted_return"]].copy()
    holdings_info = holdings_info.rename(columns={"quantile": "layer"})

    portfolio_returns = merged_df.groupby(["trade_date", "quantile"])[return_col].mean().reset_index()
    portfolio_cumulative_returns = portfolio_returns.pivot(
        index="trade_date", columns="quantile", values=return_col
    ).fillna(0)
    cumulative_returns_by_layer = cumulative_returns(portfolio_cumulative_returns)

    metrics = {}
    numeric_columns = [col for col in cumulative_returns_by_layer.columns if isinstance(col, (int, np.integer))]
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

    return portfolio_returns, holdings_info, metrics, return_col


def run_ewma_factor_strategy(
    df_all: pd.DataFrame,
    factor_columns: list[str],
    period: int = 1,
    window: int = 5,
    half_life: int = 5,
    quantiles: int = 10,
    independent_by_factor: bool = True,
) -> dict:
    """Run full EWMA factor workflow end-to-end."""
    factor_returns = calculate_factor_returns(
        df=df_all,
        factor_cols=factor_columns,
        return_col="pct_chg",
        period=period,
        independent_by_factor=independent_by_factor,
    )
    ewma_factor_returns = calculate_ewma_factors(factor_returns_df=factor_returns, window=window, half_life=half_life)
    stock_returns_pred = predict_stock_returns(
        df=df_all,
        factor_cols=factor_columns,
        ewma_factor_returns=ewma_factor_returns,
        period=period,
    )
    portfolio_returns, holdings_info, metrics, return_col = backtest_ewma_strategy(
        df=df_all,
        stock_returns_pred=stock_returns_pred,
        quantiles=quantiles,
        period=period,
    )

    if len(portfolio_returns) > 0:
        portfolio_cumulative_returns = portfolio_returns.pivot(
            index="trade_date", columns="quantile", values=return_col
        ).fillna(0)
        cumulative_returns_by_layer = cumulative_returns(portfolio_cumulative_returns).reset_index()
    else:
        cumulative_returns_by_layer = pd.DataFrame()

    return {
        "portfolio_returns": portfolio_returns,
        "holdings_info": holdings_info,
        "factor_returns": factor_returns,
        "ewma_factor_returns": ewma_factor_returns,
        "stock_returns_pred": stock_returns_pred,
        "cumulative_returns": cumulative_returns_by_layer,
        "metrics": metrics,
    }
