from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import (
    ensure_and_sort_panel,
    ensure_columns,
    get_logger,
    linear_regression_fit_predict,
)


def process_factor_data(
    df: pd.DataFrame,
    factor_cols: list[str],
    market_value_column: str = "circ_mv",
    is_timeseries: bool = True,
    do_clip: bool = True,
    do_neutralize: bool = True,
    do_standardize: bool = True,
) -> pd.DataFrame:
    """
    Process factor values with optional clipping, neutralization and standardization.

    Defaults preserve legacy behavior:
    - do_clip=True
    - do_neutralize=True
    - do_standardize=True
    """
    if not factor_cols:
        raise ValueError("[process_factor_data] factor_cols must not be empty")

    required = ["trade_date", "znz_code"] + factor_cols
    if do_neutralize:
        required.append(market_value_column)
    if "pct_chg" in df.columns:
        required.append("pct_chg")
    ensure_columns(df, required, caller="process_factor_data")

    # Keep panel rows and process each factor independently so one sparse factor
    # does not affect others in dropna/neutralization/standardization.
    df_processed = ensure_and_sort_panel(df.copy(), caller="process_factor_data")
    if df_processed.empty:
        get_logger().warning("[process_factor_data] empty frame after panel sort/dedup")
        return df_processed

    active_factor_cols = [col for col in factor_cols if df_processed[col].notna().any()]
    dropped_all_nan_factors = [col for col in factor_cols if col not in active_factor_cols]
    if dropped_all_nan_factors:
        get_logger().warning(
            "[process_factor_data] skipping all-NaN factors: %s",
            dropped_all_nan_factors,
        )
    if not active_factor_cols:
        get_logger().warning("[process_factor_data] all factors are NaN in input sample")
        empty_cols = [
            c for c in ["trade_date", "znz_code", "pct_chg", market_value_column] if c in df_processed.columns
        ] + factor_cols
        return df_processed.iloc[0:0][empty_cols].copy()

    for factor_col in active_factor_cols:
        if df_processed[factor_col].nunique(dropna=True) <= 1:
            get_logger().warning(
                "[process_factor_data] factor '%s' appears constant in input sample",
                factor_col,
            )

    if len(factor_cols) == 1 and factor_cols[0] == market_value_column:
        cols_to_keep = [c for c in ["trade_date", "znz_code", "pct_chg", "circ_mv"] if c in df_processed.columns]
        result_df = df_processed[cols_to_keep]
        if "circ_mv" in result_df.columns:
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        return result_df

    factors_for_outlier_removal = [col for col in active_factor_cols if col != market_value_column]

    if do_clip:
        if is_timeseries:
            if factors_for_outlier_removal:
                by_date = df_processed.groupby("trade_date")
                for col in factors_for_outlier_removal:
                    series = df_processed[col]
                    median = by_date[col].transform("median")
                    mad = by_date[col].transform(lambda s: (s - s.median()).abs().median())
                    upper_bound = median + 5 * mad
                    lower_bound = median - 5 * mad
                    df_processed[col] = series.clip(lower=lower_bound, upper=upper_bound)
        else:
            if factors_for_outlier_removal:
                for col in factors_for_outlier_removal:
                    series = df_processed[col]
                    median = series.median()
                    mad = (series - median).abs().median()
                    upper_bound = median + 5 * mad
                    lower_bound = median - 5 * mad
                    df_processed[col] = series.clip(lower=lower_bound, upper=upper_bound)

    for factor_col in factor_cols:
        df_processed[f"{factor_col}_neutralized"] = np.nan

    if do_neutralize:
        market_values = pd.to_numeric(df_processed[market_value_column], errors="coerce")
        df_processed["ln_market_value"] = np.log(market_values.where(market_values > 0))

        if is_timeseries:
            grouped_by_date = df_processed.groupby("trade_date", sort=False)
            for factor_col in active_factor_cols:
                for _, group in grouped_by_date:
                    y = pd.to_numeric(group[factor_col], errors="coerce")
                    x = pd.to_numeric(group["ln_market_value"], errors="coerce")
                    valid = y.notna() & x.notna()
                    n_valid = int(valid.sum())
                    if n_valid == 0:
                        continue
                    valid_index = group.index[valid]
                    if n_valid > 1:
                        X = x[valid].values.reshape(-1, 1)
                        y_valid = y[valid].values
                        _, _, y_pred = linear_regression_fit_predict(X, y_valid)
                        residuals = y_valid - y_pred
                        df_processed.loc[valid_index, f"{factor_col}_neutralized"] = residuals
                    else:
                        df_processed.loc[valid_index, f"{factor_col}_neutralized"] = y[valid].values
        else:
            x = pd.to_numeric(df_processed["ln_market_value"], errors="coerce")
            for factor_col in active_factor_cols:
                y = pd.to_numeric(df_processed[factor_col], errors="coerce")
                valid = y.notna() & x.notna()
                n_valid = int(valid.sum())
                if n_valid == 0:
                    continue
                if n_valid > 1:
                    X = x[valid].values.reshape(-1, 1)
                    y_valid = y[valid].values
                    _, _, y_pred = linear_regression_fit_predict(X, y_valid)
                    residuals = y_valid - y_pred
                    df_processed.loc[valid, f"{factor_col}_neutralized"] = residuals
                else:
                    df_processed.loc[valid, f"{factor_col}_neutralized"] = y[valid].values
    else:
        for factor_col in active_factor_cols:
            df_processed[f"{factor_col}_neutralized"] = pd.to_numeric(df_processed[factor_col], errors="coerce")

    if do_standardize:
        if is_timeseries:
            for factor_col in active_factor_cols:
                factor_neutralized = df_processed[f"{factor_col}_neutralized"]
                df_processed[factor_col] = factor_neutralized.groupby(df_processed["trade_date"]).transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
                )
        else:
            for factor_col in active_factor_cols:
                factor_neutralized = df_processed[f"{factor_col}_neutralized"]
                factor_mean = factor_neutralized.mean()
                factor_std = factor_neutralized.std()
                df_processed[factor_col] = (factor_neutralized - factor_mean) / factor_std if factor_std != 0 else 0
    else:
        for factor_col in active_factor_cols:
            df_processed[factor_col] = df_processed[f"{factor_col}_neutralized"]

    return df_processed


def process_future_return(
    df: pd.DataFrame,
    return_col: str = "pct_chg",
    period: int = 1,
    assume_sorted: bool = False,
) -> pd.DataFrame:
    """Create future return column `f"{return_col}_{period}d"` using compounding for period > 1."""
    ensure_columns(df, ["trade_date", "znz_code", return_col], caller="process_future_return")
    df_with_future_return = df.copy()

    if period < 1:
        return df_with_future_return

    if assume_sorted:
        sorted_returns = df_with_future_return.loc[:, ["znz_code", return_col]]
    else:
        sorted_idx = df_with_future_return.sort_values(["znz_code", "trade_date"], kind="mergesort").index
        sorted_returns = df_with_future_return.loc[sorted_idx, ["znz_code", return_col]]
    df_grouped = sorted_returns.groupby("znz_code", sort=False)[return_col]
    future_return_col = f"{return_col}_{period}d"

    if period == 1:
        shifted = df_grouped.shift(-1)
        df_with_future_return[future_return_col] = shifted.reindex(df_with_future_return.index)
        return df_with_future_return

    # Build compounded future returns without concatenating an N x period temporary frame.
    size = len(sorted_returns)
    compounded = np.ones(size, dtype=np.float64)
    valid_mask = np.ones(size, dtype=bool)

    for i in range(1, period + 1):
        shifted_values = np.asarray(pd.to_numeric(df_grouped.shift(-i), errors="coerce"), dtype=np.float64)
        step_valid = np.isfinite(shifted_values)
        valid_mask &= step_valid
        compounded *= 1.0 + np.where(step_valid, shifted_values, 0.0)

    future_values = np.where(valid_mask, compounded - 1.0, np.nan)
    future_series = pd.Series(future_values, index=sorted_returns.index)
    df_with_future_return[future_return_col] = future_series.reindex(df_with_future_return.index)

    return df_with_future_return


def build_return_semantics_metadata(
    base_return_col: str = "pct_chg",
    period: int = 1,
    signal_delay: int = 1,
) -> dict[str, object]:
    """Describe the date alignment implied by delayed signals and shifted returns."""
    period_i = int(period)
    delay_i = int(signal_delay)
    future_col = f"{base_return_col}_{period_i}d" if period_i >= 1 else str(base_return_col)
    raw_start = f"t+{delay_i}" if delay_i else "t"
    raw_end_offset = delay_i + period_i
    raw_end = f"t+{raw_end_offset}" if raw_end_offset else "t"
    exposure_end = f"d+{period_i}" if period_i else "d"
    return {
        "signal_delay": delay_i,
        "base_return_col": str(base_return_col),
        "future_return_col": future_col,
        "analysis_period": period_i,
        "analysis_exposure_date_rule": "alpha_used_date = raw_signal_date + signal_delay",
        "holding_window_from_analysis_exposure": f"close[d] -> close[{exposure_end}]",
        "effective_raw_signal_return_window": f"close[{raw_start}] -> close[{raw_end}]",
        "equivalent_exec_return_formula": f"close[{raw_end}] / close[{raw_start}] - 1",
        "ret_exec_cc_main_col": False,
    }


def add_execution_return_audit_columns(
    df: pd.DataFrame,
    price_col: str = "close",
    periods: tuple[int, ...] = (1,),
    prefix: str = "ret_exec_cc_audit",
) -> pd.DataFrame:
    """
    Add optional audit-only close-to-close execution return columns.

    The column for period h is anchored at raw signal date t and measures
    close[t+1+h] / close[t+1] - 1. These columns are intentionally named with
    "audit" and are not used by the default scoring or ranking path.
    """
    if price_col not in df.columns:
        return df.copy()
    ensure_columns(
        df,
        ["trade_date", "znz_code", price_col],
        caller="add_execution_return_audit_columns",
    )
    out = df.copy()
    sorted_idx = out.sort_values(["znz_code", "trade_date"], kind="mergesort").index
    prices = pd.to_numeric(out.loc[sorted_idx, price_col], errors="coerce")
    grouped = prices.groupby(out.loc[sorted_idx, "znz_code"], sort=False)
    base = grouped.shift(-1)
    for period in periods:
        h = int(period)
        if h < 1:
            continue
        future = grouped.shift(-(1 + h))
        audit = (future / base) - 1.0
        out[f"{prefix}_{h}d"] = audit.reindex(out.index)
    return out
