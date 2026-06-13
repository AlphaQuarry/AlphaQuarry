from __future__ import annotations

import logging
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


LOGGER_NAME = "factor_research"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_columns(df: pd.DataFrame, required: Sequence[str], caller: str = "") -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        prefix = f"[{caller}] " if caller else ""
        raise ValueError(f"{prefix}Missing required columns: {missing}")


def ensure_and_sort_panel(df: pd.DataFrame, caller: str = "") -> pd.DataFrame:
    ensure_columns(df, ["trade_date", "znz_code"], caller=caller)
    ordered = df.sort_values(["znz_code", "trade_date"], kind="mergesort")
    if not ordered.index.equals(df.index):
        get_logger().warning(
            "[%s] input panel is not sorted by znz_code/trade_date; auto-sorting applied",
            caller,
        )
    ordered = ordered.copy()
    dup_count = ordered.duplicated(subset=["trade_date", "znz_code"]).sum()
    if dup_count > 0:
        get_logger().warning(
            "[%s] found %s duplicated trade_date/znz_code rows, keeping first",
            caller,
            dup_count,
        )
        ordered = ordered.drop_duplicates(subset=["trade_date", "znz_code"], keep="first")
    return ordered


def resolve_future_return_column(
    df: pd.DataFrame,
    return_col: str,
    period: int,
    caller: str = "",
    required: bool = False,
) -> str:
    if period < 1:
        return return_col
    future_return_col = f"{return_col}_{period}d"
    if future_return_col in df.columns:
        return future_return_col
    if required:
        raise ValueError(f"[{caller}] future return column '{future_return_col}' not found")
    get_logger().warning(
        "[%s] future return column '%s' not found, fallback to '%s'",
        caller,
        future_return_col,
        return_col,
    )
    return return_col


def sampled_dates(unique_dates: Sequence, period: int, drop_tail_for_future: bool) -> list:
    if period <= 1:
        return list(unique_dates)
    dates = list(unique_dates)
    if drop_tail_for_future:
        dates = dates[:-period] if len(dates) > period else []
    return dates[::period]


def apply_period_sampling(df: pd.DataFrame, period: int, drop_tail_for_future: bool = False) -> pd.DataFrame:
    if period <= 1:
        return df
    dates = np.sort(df["trade_date"].unique())
    sampled = sampled_dates(dates, period=period, drop_tail_for_future=drop_tail_for_future)
    if not sampled:
        return df.iloc[0:0].copy()
    return df[df["trade_date"].isin(sampled)].copy()


def assign_quantile_labels(
    values: pd.Series,
    buckets: int,
    labels_name: str,
    warn_context: str | None = None,
    warn_stats: dict[str, float | int] | None = None,
) -> pd.Series:
    def _bump(key: str, value: float | int = 1) -> None:
        if warn_stats is None:
            return
        warn_stats[key] = float(warn_stats.get(key, 0)) + float(value)

    def _set_min(key: str, candidate: float | int) -> None:
        if warn_stats is None:
            return
        prev = warn_stats.get(key, None)
        if prev is None:
            warn_stats[key] = float(candidate)
        else:
            warn_stats[key] = float(min(float(prev), float(candidate)))

    if len(values) == 0:
        return pd.Series(index=values.index, dtype="float64")
    if len(values) < buckets:
        _bump("small_cross_section_count", 1)
        _set_min("small_cross_section_min_size", len(values))
        if warn_context:
            get_logger().warning(
                "[%s] cross-section size (%s) < buckets (%s); fallback to equal-width bins on rank index",
                warn_context,
                len(values),
                buckets,
            )
        result = pd.cut(range(len(values)), buckets, labels=range(1, buckets + 1))
        return pd.Series(result, index=values.index, name=labels_name)
    try:
        result = pd.qcut(values, buckets, labels=range(1, buckets + 1), duplicates="drop")
    except ValueError:
        _bump("qcut_fallback_count", 1)
        if warn_context:
            get_logger().warning(
                "[%s] qcut failed due to duplicated/constant values; fallback to cut",
                warn_context,
            )
        result = pd.cut(values, buckets, labels=range(1, buckets + 1))
    unique_groups = int(pd.Series(result).nunique(dropna=True))
    if unique_groups < buckets:
        _bump("fewer_groups_count", 1)
        _set_min("fewer_groups_min", unique_groups)
    if warn_context and unique_groups < buckets:
        get_logger().warning(
            "[%s] quantile grouping produced fewer groups than requested (%s < %s)",
            warn_context,
            unique_groups,
            buckets,
        )
    return pd.Series(result, index=values.index, name=labels_name)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return returns.cumsum()


def calculate_risk_metrics(
    returns: pd.Series, period: int = 1, sharpe_penalty_divisor: float = 2.0
) -> dict[str, float]:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "fitness_ratio": 0.0,
        }

    cumulative = cumulative_returns(returns)

    time_length = len(returns) * period
    annualized_return = 0.0
    returns_np = np.asarray(returns, dtype=np.float64)
    has_path_break = bool(np.any(returns_np <= -1.0))
    if has_path_break:
        # Spread/PnL style series may cross -100% and lose geometric compounding meaning.
        total_return = float(cumulative.iloc[-1])
    else:
        total_return = float(np.prod(1.0 + returns_np) - 1.0)

    if time_length > 0:
        years = time_length / 252
        if years >= 0.01:
            if has_path_break:
                # Long-short spread can be below -100% on a single day and is not always geometrically compounding.
                annualized_return = total_return / years
            else:
                log_growth = np.log1p(returns_np).sum()
                annualized_return = float(np.exp(log_growth / years) - 1.0)

    volatility = float(returns.std() * np.sqrt(252 / period)) if period > 0 else 0.0
    penalty_divisor = sharpe_penalty_divisor if sharpe_penalty_divisor and sharpe_penalty_divisor > 0 else 1.0
    sharpe_numerator = annualized_return / penalty_divisor
    sharpe_ratio = sharpe_numerator / volatility if volatility != 0 else 0.0

    if has_path_break:
        gross_curve = 1.0 + cumulative
    else:
        gross_curve = pd.Series(np.cumprod(1.0 + returns_np), index=returns.index)
    running_max = gross_curve.expanding().max()
    valid_max = running_max.replace(0, np.nan)
    drawdown = (gross_curve - valid_max) / valid_max
    max_drawdown = abs(float(drawdown.min())) if len(drawdown.dropna()) else 0.0

    fitness_ratio = (
        sharpe_ratio * np.sqrt(abs(sharpe_numerator) / max(max_drawdown, 0.125))
        if max_drawdown != 0 and np.isfinite(sharpe_ratio)
        else 0.0
    )

    annualized_return = float(np.real(annualized_return)) if np.isfinite(np.real(annualized_return)) else 0.0
    volatility = float(volatility) if np.isfinite(volatility) else 0.0
    sharpe_ratio = float(np.real(sharpe_ratio)) if np.isfinite(np.real(sharpe_ratio)) else 0.0
    max_drawdown = float(max_drawdown) if np.isfinite(max_drawdown) else 0.0
    fitness_ratio = float(np.real(fitness_ratio)) if np.isfinite(np.real(fitness_ratio)) else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "fitness_ratio": fitness_ratio,
    }


def infer_return_column_from_layer_frame(frame: pd.DataFrame) -> str:
    candidates = [c for c in frame.columns if c not in {"trade_date", "layer", "znz_code", "quantile"}]
    if not candidates:
        raise ValueError("Unable to infer return column from layer result frame")
    preferred = [
        c
        for c in candidates
        if str(c) in {"pct_chg", "ret_1d", "future_return"}
        or str(c).startswith("ret_")
        or str(c).endswith("_1d")
        or str(c).endswith("_5d")
        or str(c).endswith("_10d")
        or str(c).endswith("_20d")
    ]
    if preferred:
        return str(preferred[-1])
    return candidates[-1]


def flatten(items: Iterable) -> list:
    out: list = []
    for item in items:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def linear_regression_fit_predict(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit OLS with intercept using numpy and return (coef_without_intercept, intercept, prediction).
    Supports y as shape (n,) or (n, k).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_with_const = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
    pred = X_with_const @ beta
    if y.ndim == 1:
        intercept = np.array([beta[0]])
        coef = np.array([beta[1]] if beta.shape[0] > 1 else [0.0])
    else:
        intercept = beta[0, :]
        coef = beta[1:, :]
    return coef, intercept, pred
