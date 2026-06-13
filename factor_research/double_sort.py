from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .utils import assign_quantile_labels


@dataclass(frozen=True)
class DoubleSortConfig:
    control_col: str = "total_mv"
    fallback_control_col: str = "circ_mv"
    factor_bins: int = 5
    control_bins: int = 5
    method: str = "conditional"


def assign_quantile_groups(
    df: pd.DataFrame,
    value_col: str,
    by_cols: Sequence[str],
    bins: int = 5,
    output_col: str = "quantile_group",
) -> pd.DataFrame:
    out = df.copy()
    group_cols = [str(c) for c in by_cols if str(c) in out.columns]
    if not group_cols:
        out[output_col] = assign_quantile_labels(out[value_col], int(bins), labels_name=output_col)
        return out
    out[output_col] = out.groupby(group_cols, sort=False)[value_col].transform(
        lambda s: assign_quantile_labels(s, int(bins), labels_name=output_col)
    )
    return out


def newey_west_tstat(values: Sequence[float] | pd.Series, lag: int | None = None) -> float:
    return float(newey_west_stats(values, lag=lag).get("nw_t", np.nan))


def newey_west_stats(values: Sequence[float] | pd.Series, lag: int | None = None) -> dict[str, float]:
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = int(len(s))
    if n < 2:
        return {
            "mean": np.nan,
            "std": np.nan,
            "nw_se": np.nan,
            "nw_t": np.nan,
            "p_value": np.nan,
            "obs": float(n),
        }
    x = np.asarray(s, dtype=float)
    demeaned = x - float(np.mean(x))
    max_lag = int(lag) if lag is not None else min(5, n - 1)
    max_lag = max(0, min(max_lag, n - 1))
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    variance = gamma0
    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1.0)
        gamma = float(np.dot(demeaned[k:], demeaned[:-k]) / n)
        variance += 2.0 * weight * gamma
    se = float(np.sqrt(max(variance, 0.0) / n))
    std = float(np.std(x, ddof=1)) if n > 1 else np.nan
    if se <= 0 or not np.isfinite(se):
        tstat = np.nan
    else:
        tstat = float(np.mean(x) / se)
    p_value = float(math.erfc(abs(tstat) / math.sqrt(2.0))) if np.isfinite(tstat) else np.nan
    return {
        "mean": float(np.mean(x)),
        "std": std,
        "nw_se": se,
        "nw_t": tstat,
        "p_value": p_value,
        "obs": float(n),
    }


def double_sort_analysis(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    return_col: str,
    control_col: str = "total_mv",
    fallback_control_col: str = "circ_mv",
    factor_bins: int = 5,
    control_bins: int = 5,
    method: str = "conditional",
) -> dict[str, pd.DataFrame]:
    if df is None or df.empty:
        return _empty_result()
    date_col = "trade_date" if "trade_date" in df.columns else "date"
    if date_col not in df.columns or return_col not in df.columns:
        return _empty_result()
    control_used = str(control_col)
    if control_used not in df.columns and str(fallback_control_col) in df.columns:
        control_used = str(fallback_control_col)
    if control_used not in df.columns:
        return _empty_result()

    matrix_rows: list[dict[str, object]] = []
    spread_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    method_norm = str(method or "conditional").strip().lower()
    for factor in [str(c) for c in factor_cols if str(c) in df.columns]:
        work = df[[date_col, factor, control_used, return_col]].copy()
        if "znz_code" in df.columns:
            work["znz_code"] = df["znz_code"].astype(str)
        work = work.dropna(subset=[date_col, factor, control_used, return_col])
        if work.empty:
            continue
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
        work = assign_quantile_groups(
            work,
            control_used,
            by_cols=[date_col],
            bins=control_bins,
            output_col="control_group",
        )
        if method_norm == "conditional":
            work = assign_quantile_groups(
                work,
                factor,
                by_cols=[date_col, "control_group"],
                bins=factor_bins,
                output_col="factor_group",
            )
        else:
            work = assign_quantile_groups(
                work,
                factor,
                by_cols=[date_col],
                bins=factor_bins,
                output_col="factor_group",
            )
        work = work.dropna(subset=["control_group", "factor_group"])
        if work.empty:
            continue
        work["control_group"] = work["control_group"].astype(int)
        work["factor_group"] = work["factor_group"].astype(int)
        grouped = (
            work.groupby([date_col, "control_group", "factor_group"], sort=True)[return_col]
            .agg(mean_return="mean", count="count")
            .reset_index()
        )
        grouped["factor"] = factor
        grouped["control_col_used"] = control_used
        matrix_rows.extend(grouped.rename(columns={date_col: "trade_date"}).to_dict("records"))

        for date, g in grouped.groupby(date_col, sort=True):
            high = g[g["factor_group"] == g["factor_group"].max()]["mean_return"].mean()
            low = g[g["factor_group"] == g["factor_group"].min()]["mean_return"].mean()
            spread = float(high - low) if np.isfinite(high) and np.isfinite(low) else np.nan
            spread_rows.append(
                {
                    "trade_date": date,
                    "factor": factor,
                    "double_sort_spread_return": spread,
                    "control_col_used": control_used,
                    "method": method_norm,
                }
            )
        spread_series = pd.to_numeric(
            pd.Series([r["double_sort_spread_return"] for r in spread_rows if r["factor"] == factor]),
            errors="coerce",
        )
        stats = newey_west_stats(spread_series)
        monotonicity_values: list[float] = []
        for _, g_date in grouped.groupby(date_col, sort=True):
            by_factor = g_date.groupby("factor_group", sort=True)["mean_return"].mean().reset_index()
            if by_factor["factor_group"].nunique() >= 2:
                corr = by_factor["factor_group"].corr(by_factor["mean_return"], method="spearman")
                if pd.notna(corr):
                    monotonicity_values.append(float(corr))
        summary_rows.append(
            {
                "factor": factor,
                "control_col_used": control_used,
                "method": method_norm,
                "factor_bins": int(factor_bins),
                "control_bins": int(control_bins),
                "double_sort_spread_mean": float(spread_series.mean(skipna=True))
                if spread_series.notna().any()
                else np.nan,
                "double_sort_spread_tstat": float(stats.get("nw_t", np.nan)),
                "double_sort_p_value": float(stats.get("p_value", np.nan)),
                "double_sort_positive_ratio": float((spread_series.dropna() > 0).mean())
                if spread_series.notna().any()
                else np.nan,
                "double_sort_annualized_return": float(spread_series.mean(skipna=True) * 252.0)
                if spread_series.notna().any()
                else np.nan,
                "double_sort_monotonicity_spearman": float(np.mean(monotonicity_values))
                if monotonicity_values
                else np.nan,
                "double_sort_group_min_count": int(grouped["count"].min())
                if "count" in grouped.columns and not grouped.empty
                else 0,
                "double_sort_obs": int(work.shape[0]),
                "double_sort_dates": int(work[date_col].nunique()),
            }
        )
    return {
        "matrix_returns_df": pd.DataFrame(matrix_rows),
        "spread_returns_df": pd.DataFrame(spread_rows),
        "summary_df": pd.DataFrame(summary_rows),
    }


def _empty_result() -> dict[str, pd.DataFrame]:
    return {
        "matrix_returns_df": pd.DataFrame(),
        "spread_returns_df": pd.DataFrame(),
        "summary_df": pd.DataFrame(),
    }
