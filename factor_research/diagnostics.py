from __future__ import annotations

import gc
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .preprocess import process_factor_data, process_future_return
from .single_factor import (
    calculate_icir,
    calculate_long_short_metrics,
    calculate_turnover_rate,
    factor_layer_analysis,
)
from .utils import ensure_and_sort_panel, ensure_columns


def _ic_summary_stats(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    if s.empty:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ir": np.nan,
            "positive_ic_ratio": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ic_skew": np.nan,
            "ic_kurtosis": np.nan,
            "obs_count": 0,
        }

    ic_mean = float(s.mean())
    ic_std = float(s.std())
    n = len(s)
    t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std not in (0, np.nan) and ic_std != 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1)) if n > 1 and not np.isnan(t_stat) else np.nan
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ir": ic_mean / ic_std if ic_std not in (0, np.nan) and ic_std != 0 else np.nan,
        "positive_ic_ratio": float((s > 0).mean()),
        "t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        "ic_skew": float(s.skew()),
        "ic_kurtosis": float(s.kurtosis()),
        "obs_count": int(n),
    }


def calculate_factor_coverage(df: pd.DataFrame, factor_cols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Calculate factor coverage/missing rates by date and overall."""
    ensure_columns(
        df,
        ["trade_date", "znz_code"] + list(factor_cols),
        caller="calculate_factor_coverage",
    )

    by_date_rows: list[dict] = []
    total_obs = len(df)
    overall_rows: list[dict] = []

    for factor in factor_cols:
        non_missing = df[factor].notna().sum()
        coverage = non_missing / total_obs if total_obs else np.nan
        overall_rows.append(
            {
                "factor": factor,
                "total_obs": total_obs,
                "non_missing_obs": int(non_missing),
                "coverage_rate": coverage,
                "missing_rate": 1 - coverage if not np.isnan(coverage) else np.nan,
            }
        )

        grp = df.groupby("trade_date", sort=False)[factor]
        by_date = pd.DataFrame(
            {
                "trade_date": grp.count().index,
                "non_missing_obs": grp.count().values,
                "total_obs": grp.size().values,
            }
        )
        by_date["factor"] = factor
        by_date["coverage_rate"] = by_date["non_missing_obs"] / by_date["total_obs"]
        by_date["missing_rate"] = 1 - by_date["coverage_rate"]
        by_date_rows.extend(by_date.to_dict("records"))

    return {
        "overall": pd.DataFrame(overall_rows),
        "by_date": pd.DataFrame(by_date_rows),
    }


def calculate_ic_stability(ic_df: pd.DataFrame, factor_cols: Sequence[str] | None = None) -> pd.DataFrame:
    """Calculate IC stability statistics per factor."""
    ensure_columns(ic_df, ["trade_date"], caller="calculate_ic_stability")
    if factor_cols is None:
        factor_cols = [c[:-3] for c in ic_df.columns if c.endswith("_ic")]

    rows = []
    for factor in factor_cols:
        col = f"{factor}_ic"
        if col not in ic_df.columns:
            continue
        stats_row = _ic_summary_stats(ic_df[col])
        stats_row["factor"] = factor
        rows.append(stats_row)
    return pd.DataFrame(rows)


def calculate_ic_time_breakdown(
    ic_df: pd.DataFrame,
    factor_cols: Sequence[str] | None = None,
    freq: str = "Y",
) -> pd.DataFrame:
    """Break down IC statistics by year/month (`freq='Y'` or `freq='M'`)."""
    ensure_columns(ic_df, ["trade_date"], caller="calculate_ic_time_breakdown")
    if factor_cols is None:
        factor_cols = [c[:-3] for c in ic_df.columns if c.endswith("_ic")]

    work = ic_df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    if freq.upper() == "Y":
        work["period_label"] = work["trade_date"].dt.to_period("Y").astype(str)
    elif freq.upper() == "M":
        work["period_label"] = work["trade_date"].dt.to_period("M").astype(str)
    else:
        raise ValueError("freq must be 'Y' or 'M'")

    rows: list[dict] = []
    for factor in factor_cols:
        col = f"{factor}_ic"
        if col not in work.columns:
            continue
        for period_label, g in work.groupby("period_label", sort=True):
            stats_row = _ic_summary_stats(g[col])
            stats_row.update({"factor": factor, "period_label": period_label, "freq": freq.upper()})
            rows.append(stats_row)
    return pd.DataFrame(rows)


def summarize_ic_sign_consistency(
    ic_breakdown_df: pd.DataFrame,
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Summarize sign consistency for yearly/monthly IC breakdown.

    Consistency is measured as the ratio of periods where `ic_mean` sign
    matches the overall factor sign (from summary_df when provided).
    """
    if ic_breakdown_df is None or ic_breakdown_df.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "freq",
                "sign_consistency",
                "period_count",
                "matched_count",
            ]
        )
    required = {"factor", "freq", "ic_mean"}
    if not required.issubset(ic_breakdown_df.columns):
        return pd.DataFrame(
            columns=[
                "factor",
                "freq",
                "sign_consistency",
                "period_count",
                "matched_count",
            ]
        )

    overall_sign_map: dict[str, int] = {}
    if (
        isinstance(summary_df, pd.DataFrame)
        and not summary_df.empty
        and {"factor", "ic_mean"}.issubset(summary_df.columns)
    ):
        tmp = summary_df[["factor", "ic_mean"]].copy()
        tmp["factor"] = tmp["factor"].astype(str)
        for _, row in tmp.iterrows():
            v = float(row["ic_mean"]) if pd.notna(row["ic_mean"]) else np.nan
            overall_sign_map[str(row["factor"])] = 1 if v > 0 else (-1 if v < 0 else 0)

    rows: list[dict[str, object]] = []
    work = ic_breakdown_df.copy()
    work["factor"] = work["factor"].astype(str)
    work["freq"] = work["freq"].astype(str).str.upper()
    for (factor, freq), g in work.groupby(["factor", "freq"], sort=False):
        s = pd.to_numeric(g["ic_mean"], errors="coerce").dropna()
        if s.empty:
            rows.append(
                {
                    "factor": factor,
                    "freq": freq,
                    "sign_consistency": np.nan,
                    "period_count": 0,
                    "matched_count": 0,
                }
            )
            continue
        sign_ref = overall_sign_map.get(factor, 1 if float(s.mean()) > 0 else (-1 if float(s.mean()) < 0 else 0))
        signs = np.sign(s.to_numpy(dtype=float))
        valid = signs != 0
        if sign_ref == 0 or not np.any(valid):
            rows.append(
                {
                    "factor": factor,
                    "freq": freq,
                    "sign_consistency": np.nan,
                    "period_count": int(np.sum(valid)),
                    "matched_count": 0,
                }
            )
            continue
        matched = int(np.sum(signs[valid] == sign_ref))
        period_count = int(np.sum(valid))
        rows.append(
            {
                "factor": factor,
                "freq": freq,
                "sign_consistency": float(matched / period_count) if period_count else np.nan,
                "period_count": period_count,
                "matched_count": matched,
            }
        )
    return pd.DataFrame(rows)


def summarize_period_robustness(
    period_comparison_df: pd.DataFrame,
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Summarize holding-period robustness using existing period comparison outputs.
    """
    if period_comparison_df is None or period_comparison_df.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "robust_period_positive_ratio",
                "robust_ic_sign_consistency",
                "robust_ir_median",
                "period_count",
            ]
        )
    required = {"factor", "period", "ic_mean", "ir", "long_short_total_return"}
    if not required.issubset(period_comparison_df.columns):
        return pd.DataFrame(
            columns=[
                "factor",
                "robust_period_positive_ratio",
                "robust_ic_sign_consistency",
                "robust_ir_median",
                "period_count",
            ]
        )

    overall_sign_map: dict[str, int] = {}
    if (
        isinstance(summary_df, pd.DataFrame)
        and not summary_df.empty
        and {"factor", "ic_mean"}.issubset(summary_df.columns)
    ):
        tmp = summary_df[["factor", "ic_mean"]].copy()
        tmp["factor"] = tmp["factor"].astype(str)
        for _, row in tmp.iterrows():
            v = float(row["ic_mean"]) if pd.notna(row["ic_mean"]) else np.nan
            overall_sign_map[str(row["factor"])] = 1 if v > 0 else (-1 if v < 0 else 0)

    rows: list[dict[str, object]] = []
    work = period_comparison_df.copy()
    work["factor"] = work["factor"].astype(str)
    for factor, g in work.groupby("factor", sort=False):
        ret = pd.to_numeric(g["long_short_total_return"], errors="coerce")
        ic = pd.to_numeric(g["ic_mean"], errors="coerce")
        ir = pd.to_numeric(g["ir"], errors="coerce")

        ret_valid = ret.dropna()
        ic_valid = ic.dropna()
        sign_ref = overall_sign_map.get(
            factor,
            1 if float(ic_valid.mean()) > 0 else (-1 if float(ic_valid.mean()) < 0 else 0),
        )
        if sign_ref == 0 or ic_valid.empty:
            ic_consistency = np.nan
        else:
            signs = np.sign(ic_valid.to_numpy(dtype=float))
            valid = signs != 0
            ic_consistency = float((signs[valid] == sign_ref).mean()) if np.any(valid) else np.nan

        rows.append(
            {
                "factor": factor,
                "robust_period_positive_ratio": float((ret_valid > 0).mean()) if not ret_valid.empty else np.nan,
                "robust_ic_sign_consistency": ic_consistency,
                "robust_ir_median": float(ir.dropna().median()) if ir.notna().any() else np.nan,
                "period_count": int(g["period"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def calculate_layer_monotonicity(
    layer_results: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Measure layer monotonicity using Spearman correlation between layer id and mean layer return.
    """
    daily_rows: list[dict] = []
    summary_rows: list[dict] = []

    for factor, data in layer_results.items():
        return_col = data.columns[-1]
        daily_scores: list[float] = []

        for date, g in data.groupby("trade_date", sort=False):
            layer_mean = g.groupby("layer")[return_col].mean().reset_index()
            if len(layer_mean) < 2:
                score = np.nan
            else:
                score = stats.spearmanr(layer_mean["layer"], layer_mean[return_col], nan_policy="omit").correlation
            daily_rows.append({"factor": factor, "trade_date": date, "monotonicity_score": score})
            daily_scores.append(score)

        s = pd.Series(daily_scores).dropna()
        summary_rows.append(
            {
                "factor": factor,
                "monotonicity_mean": float(s.mean()) if not s.empty else np.nan,
                "monotonicity_std": float(s.std()) if not s.empty else np.nan,
                "monotonicity_positive_ratio": float((s > 0).mean()) if not s.empty else np.nan,
                "monotonicity_obs_count": int(len(s)),
            }
        )

    return {"daily": pd.DataFrame(daily_rows), "summary": pd.DataFrame(summary_rows)}


def analyze_holding_period_robustness(
    df: pd.DataFrame,
    factor_cols: Sequence[str],
    periods: Sequence[int] = (1, 5, 10, 20),
    return_col: str = "pct_chg",
    layers: int = 5,
    market_value_column: str = "circ_mv",
    is_timeseries: bool = True,
    do_clip: bool = True,
    do_neutralize: bool = True,
    do_standardize: bool = True,
    store_detailed_results: bool = False,
    run_gc_per_period: bool = False,
    already_computed_periods: Sequence[int] | None = None,
    precomputed_period_details: dict[int, dict[str, object]] | None = None,
) -> dict[str, object]:
    """
    Compare factor performance across holding periods.

    Memory notes:
    - `store_detailed_results=False` keeps only lightweight summaries in `details` to reduce memory pressure.
    - Set `store_detailed_results=True` if you need large per-date/per-stock intermediate tables.
    - Set `run_gc_per_period=True` in notebooks if kernels keep many historical references.
    - `already_computed_periods` skips periods that have been computed earlier in the notebook/session.
    - `precomputed_period_details` provides previously computed period outputs for display without recomputation.
    - `do_clip/do_neutralize/do_standardize` are forwarded to `process_factor_data`.
    """
    ensure_columns(
        df,
        ["trade_date", "znz_code", return_col, market_value_column] + list(factor_cols),
        caller="analyze_holding_period_robustness",
    )

    period_rows: list[dict] = []
    period_details: dict[int, dict] = {}
    base_cols = ["trade_date", "znz_code", return_col, market_value_column] + list(factor_cols)
    base_df = df.loc[:, base_cols]
    base_df_sorted = ensure_and_sort_panel(base_df, caller="analyze_holding_period_robustness")
    requested_periods = [int(p) for p in periods]
    unique_periods: list[int] = []
    for p in requested_periods:
        if p < 1:
            continue
        if p not in unique_periods:
            unique_periods.append(p)
    already_periods = {int(p) for p in (already_computed_periods or []) if int(p) >= 1}
    precomputed_map: dict[int, dict[str, object]] = {}
    for p, detail in (precomputed_period_details or {}).items():
        try:
            period_int = int(p)
        except (TypeError, ValueError):
            continue
        if period_int >= 1 and period_int in unique_periods and isinstance(detail, dict):
            precomputed_map[period_int] = detail

    loaded_precomputed_periods = [p for p in unique_periods if p in precomputed_map]
    periods_to_compute = [p for p in unique_periods if p not in already_periods and p not in precomputed_map]
    skipped_periods = [p for p in unique_periods if p in already_periods]
    missing_precomputed_periods = [p for p in skipped_periods if p not in precomputed_map]

    def _sanitize_period_outputs(
        summary_df_obj: object,
        long_short_metrics_obj: object,
        turnover_results_obj: object,
        mono_obj: object,
        ic_df_obj: object = None,
        layer_results_obj: object = None,
    ) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame, object, object]:
        summary_df_safe = (
            summary_df_obj.copy()
            if isinstance(summary_df_obj, pd.DataFrame)
            else pd.DataFrame(columns=["factor", "ic_mean", "ir", "p_value", "positive_ic_ratio"])
        )
        long_short_metrics_safe = long_short_metrics_obj if isinstance(long_short_metrics_obj, dict) else {}
        turnover_results_safe = turnover_results_obj if isinstance(turnover_results_obj, dict) else {}

        if isinstance(mono_obj, pd.DataFrame):
            mono_safe = mono_obj.copy()
        elif isinstance(mono_obj, dict) and isinstance(mono_obj.get("summary"), pd.DataFrame):
            mono_safe = mono_obj["summary"].copy()
        else:
            mono_safe = pd.DataFrame(columns=["factor", "monotonicity_mean"])

        return (
            summary_df_safe,
            long_short_metrics_safe,
            turnover_results_safe,
            mono_safe,
            ic_df_obj,
            layer_results_obj,
        )

    def _build_detail(
        summary_df_safe: pd.DataFrame,
        long_short_metrics_safe: dict,
        turnover_results_safe: dict,
        mono_safe: pd.DataFrame,
        ic_df_obj: object = None,
        layer_results_obj: object = None,
    ) -> dict[str, object]:
        detail_out: dict[str, object] = {
            "summary_df": summary_df_safe,
            "long_short_metrics": long_short_metrics_safe,
            "monotonicity_summary": mono_safe,
            "avg_turnover_by_factor": pd.DataFrame(
                [
                    {
                        "factor": factor,
                        "avg_min_layer_turnover": tr["min_layer_turnover"].mean()
                        if tr is not None and not tr.empty
                        else np.nan,
                        "avg_max_layer_turnover": tr["max_layer_turnover"].mean()
                        if tr is not None and not tr.empty
                        else np.nan,
                    }
                    for factor in factor_cols
                    for tr in [turnover_results_safe.get(factor)]
                ]
            ),
            "ic_df": None,
            "layer_results": None,
            "turnover_results": None,
        }
        if store_detailed_results:
            detail_out["ic_df"] = ic_df_obj if isinstance(ic_df_obj, pd.DataFrame) else None
            detail_out["layer_results"] = layer_results_obj if isinstance(layer_results_obj, dict) else None
            detail_out["turnover_results"] = turnover_results_safe
        return detail_out

    def _append_period_rows(
        period_val: int,
        summary_df_safe: pd.DataFrame,
        long_short_metrics_safe: dict,
        turnover_results_safe: dict,
        mono_safe: pd.DataFrame,
    ) -> None:
        has_summary_factor = "factor" in summary_df_safe.columns
        has_mono_factor = "factor" in mono_safe.columns
        for factor in factor_cols:
            row = {"period": period_val, "factor": factor}

            ic_row = summary_df_safe[summary_df_safe["factor"] == factor] if has_summary_factor else pd.DataFrame()
            if not ic_row.empty:
                ir = ic_row.iloc[0]
                row.update(
                    {
                        "ic_mean": ir.get("ic_mean", np.nan),
                        "ir": ir.get("ir", np.nan),
                        "p_value": ir.get("p_value", np.nan),
                        "positive_ic_ratio": ir.get("positive_ic_ratio", np.nan),
                    }
                )
            else:
                row.update(
                    {
                        "ic_mean": np.nan,
                        "ir": np.nan,
                        "p_value": np.nan,
                        "positive_ic_ratio": np.nan,
                    }
                )

            ls = long_short_metrics_safe.get(factor, {})
            row.update(
                {
                    "long_short_total_return": ls.get("total_return", np.nan),
                    "long_short_annualized_return": ls.get("annualized_return", np.nan),
                    "long_short_sharpe_ratio": ls.get("sharpe_ratio", np.nan),
                    "long_short_max_drawdown": ls.get("max_drawdown", np.nan),
                }
            )

            tr = turnover_results_safe.get(factor)
            if tr is not None and hasattr(tr, "empty") and not tr.empty:
                row.update(
                    {
                        "avg_min_layer_turnover": tr["min_layer_turnover"].mean(),
                        "avg_max_layer_turnover": tr["max_layer_turnover"].mean(),
                    }
                )
            else:
                row.update({"avg_min_layer_turnover": np.nan, "avg_max_layer_turnover": np.nan})

            mono_row = mono_safe[mono_safe["factor"] == factor] if has_mono_factor else pd.DataFrame()
            row["layer_monotonicity"] = (
                mono_row.iloc[0].get("monotonicity_mean", np.nan) if not mono_row.empty else np.nan
            )
            period_rows.append(row)

    computed_periods: list[int] = []
    for period in unique_periods:
        if period in precomputed_map:
            raw_detail = precomputed_map[period]
            mono_input = raw_detail.get("monotonicity_summary")
            if mono_input is None and "monotonicity" in raw_detail:
                mono_input = raw_detail["monotonicity"]
            (
                summary_df,
                long_short_metrics,
                turnover_results,
                mono,
                ic_df,
                layer_results,
            ) = _sanitize_period_outputs(
                summary_df_obj=raw_detail.get("summary_df"),
                long_short_metrics_obj=raw_detail.get("long_short_metrics"),
                turnover_results_obj=raw_detail.get("turnover_results"),
                mono_obj=mono_input,
                ic_df_obj=raw_detail.get("ic_df"),
                layer_results_obj=raw_detail.get("layer_results"),
            )
            period_details[period] = _build_detail(
                summary_df_safe=summary_df,
                long_short_metrics_safe=long_short_metrics,
                turnover_results_safe=turnover_results,
                mono_safe=mono,
                ic_df_obj=ic_df,
                layer_results_obj=layer_results,
            )
            _append_period_rows(period, summary_df, long_short_metrics, turnover_results, mono)
            continue

        if period not in periods_to_compute:
            continue

        work = process_future_return(base_df_sorted, return_col=return_col, period=period, assume_sorted=True)
        future_col = f"{return_col}_{period}d"
        cols = [
            "trade_date",
            "znz_code",
            return_col,
            future_col,
            market_value_column,
        ] + list(factor_cols)
        processed = process_factor_data(
            work[cols].copy(),
            factor_cols=list(factor_cols),
            market_value_column=market_value_column,
            is_timeseries=is_timeseries,
            do_clip=do_clip,
            do_neutralize=do_neutralize,
            do_standardize=do_standardize,
        )

        ic_df, summary_df = calculate_icir(processed, list(factor_cols), return_col=return_col, period=period)
        layer_results = factor_layer_analysis(
            processed,
            list(factor_cols),
            return_col=return_col,
            period=period,
            layers=layers,
        )
        long_short_metrics, _ = calculate_long_short_metrics(
            layer_results,
            period=period,
            include_long_short_visualization=False,
            direction_mode="by_ic_sign",
            ic_summary_df=summary_df,
        )
        turnover_results = calculate_turnover_rate(layer_results, period=period)
        mono = calculate_layer_monotonicity(layer_results)["summary"]

        (
            summary_df_safe,
            long_short_metrics_safe,
            turnover_results_safe,
            mono_safe,
            _,
            _,
        ) = _sanitize_period_outputs(
            summary_df_obj=summary_df,
            long_short_metrics_obj=long_short_metrics,
            turnover_results_obj=turnover_results,
            mono_obj=mono,
        )
        period_details[period] = _build_detail(
            summary_df_safe=summary_df_safe,
            long_short_metrics_safe=long_short_metrics_safe,
            turnover_results_safe=turnover_results_safe,
            mono_safe=mono_safe,
            ic_df_obj=ic_df,
            layer_results_obj=layer_results,
        )
        _append_period_rows(
            period,
            summary_df_safe,
            long_short_metrics_safe,
            turnover_results_safe,
            mono_safe,
        )
        computed_periods.append(period)

        del (
            work,
            processed,
            ic_df,
            summary_df,
            layer_results,
            long_short_metrics,
            turnover_results,
            mono,
        )
        if run_gc_per_period:
            gc.collect()

    comparison_df = pd.DataFrame(period_rows)
    # Keep backward-compatible aliases for notebook plotting cells.
    return {
        "comparison": comparison_df,
        "period_comparison_df": comparison_df,
        "robustness_comparison_df": comparison_df,
        "details": period_details,
        "requested_periods": unique_periods,
        "computed_periods": computed_periods,
        "loaded_precomputed_periods": loaded_precomputed_periods,
        "skipped_existing_periods": skipped_periods,
        "missing_precomputed_periods": missing_precomputed_periods,
    }
