from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_factor_summary_report(
    factor_cols: list[str],
    coverage_overall_df: pd.DataFrame,
    ic_summary_df: pd.DataFrame,
    monotonicity_summary_df: pd.DataFrame,
    long_short_metrics: dict[str, dict[str, Any]],
    turnover_results: dict[str, pd.DataFrame],
    effective_factors_df: pd.DataFrame | None = None,
    factor_effectiveness_df: pd.DataFrame | None = None,
    apply_filtering: bool = True,
) -> pd.DataFrame:
    """
    Build a summary report with pass/fail reason diagnostics.
    """
    effective_set = set()
    if effective_factors_df is not None and not effective_factors_df.empty and "factor" in effective_factors_df.columns:
        effective_set = set(effective_factors_df["factor"])

    cov_map = coverage_overall_df.set_index("factor").to_dict("index") if not coverage_overall_df.empty else {}
    ic_map = (
        ic_summary_df.set_index("factor").to_dict("index")
        if not ic_summary_df.empty and "factor" in ic_summary_df.columns
        else {}
    )
    mono_map = (
        monotonicity_summary_df.set_index("factor").to_dict("index")
        if not monotonicity_summary_df.empty and "factor" in monotonicity_summary_df.columns
        else {}
    )
    eff_map = (
        factor_effectiveness_df.set_index("factor").to_dict("index")
        if isinstance(factor_effectiveness_df, pd.DataFrame)
        and not factor_effectiveness_df.empty
        and "factor" in factor_effectiveness_df.columns
        else {}
    )

    rows: list[dict] = []
    for factor in factor_cols:
        row = {"factor": factor}
        cov = cov_map.get(factor, {})
        row["coverage_rate"] = cov.get("coverage_rate", np.nan)
        row["missing_rate"] = cov.get("missing_rate", np.nan)

        ic = ic_map.get(factor, {})
        row["ic_mean"] = ic.get("ic_mean", np.nan)
        row["ir"] = ic.get("ir", np.nan)
        row["p_value"] = ic.get("p_value", np.nan)
        row["positive_ic_ratio"] = ic.get("positive_ic_ratio", np.nan)

        mono = mono_map.get(factor, {})
        row["layer_monotonicity"] = mono.get("monotonicity_mean", np.nan)

        ls = long_short_metrics.get(factor, {})
        row["long_short_total_return"] = ls.get("total_return", np.nan)
        row["long_short_annualized_return"] = ls.get("annualized_return", np.nan)
        row["long_short_sharpe_ratio"] = ls.get("sharpe_ratio", np.nan)
        row["long_short_fitness_ratio"] = ls.get("fitness_ratio", np.nan)
        row["long_short_max_drawdown"] = ls.get("max_drawdown", np.nan)

        tr = turnover_results.get(factor)
        if tr is not None and not tr.empty:
            row["avg_min_layer_turnover"] = tr["min_layer_turnover"].mean()
            row["avg_max_layer_turnover"] = tr["max_layer_turnover"].mean()
        else:
            row["avg_min_layer_turnover"] = np.nan
            row["avg_max_layer_turnover"] = np.nan

        passed = factor in effective_set if apply_filtering else True
        row["passed_filter"] = passed

        eff = eff_map.get(factor, {})
        row["stage_a_pass"] = bool(eff.get("stage_a_pass", False)) if apply_filtering else True
        row["stage_b_pass"] = bool(eff.get("stage_b_pass", passed)) if apply_filtering else True
        row["effectiveness_score"] = eff.get("effectiveness_score", np.nan)
        row["effectiveness_tier"] = eff.get("effectiveness_tier", "")
        row["warning_reasons"] = str(eff.get("warning_reasons", "") or "")

        fail_reasons = str(eff.get("fail_reasons", "") or "")
        if apply_filtering and not passed and not fail_reasons:
            fallback: list[str] = []
            if np.isnan(row["ic_mean"]) or abs(row["ic_mean"]) < 0.02:
                fallback.append("|ic_mean|<0.02")
            if np.isnan(row["ir"]) or abs(row["ir"]) < 0.3:
                fallback.append("|ir|<0.3")
            if np.isnan(row["p_value"]) or row["p_value"] > 0.05:
                fallback.append("p_value>0.05")
            if np.isnan(row["long_short_sharpe_ratio"]) or row["long_short_sharpe_ratio"] < 1:
                fallback.append("sharpe_ratio<1")
            if np.isnan(row["long_short_fitness_ratio"]) or row["long_short_fitness_ratio"] < 1:
                fallback.append("fitness_ratio<1")
            if not fallback:
                fallback.append("layer_significance_rule_not_satisfied")
            fail_reasons = "; ".join(fallback)
        row["fail_reasons"] = fail_reasons
        rows.append(row)

    return pd.DataFrame(rows)


def build_structured_report(
    summary_report_df: pd.DataFrame,
    ic_stability_df: pd.DataFrame | None = None,
    ic_yearly_df: pd.DataFrame | None = None,
    ic_monthly_df: pd.DataFrame | None = None,
    coverage_by_date_df: pd.DataFrame | None = None,
    robustness_comparison_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Package report tables into one structured dict."""
    report = {"summary": summary_report_df}
    if ic_stability_df is not None:
        report["ic_stability"] = ic_stability_df
    if ic_yearly_df is not None:
        report["ic_yearly"] = ic_yearly_df
    if ic_monthly_df is not None:
        report["ic_monthly"] = ic_monthly_df
    if coverage_by_date_df is not None:
        report["coverage_by_date"] = coverage_by_date_df
    if robustness_comparison_df is not None:
        report["period_robustness"] = robustness_comparison_df
    return report


def export_report(report: dict[str, pd.DataFrame] | pd.DataFrame, output_path: str | Path) -> Path:
    """
    Export report to `.xlsx` or `.csv`.
    - If output is `.xlsx`: all tables are written as separate sheets.
    - If output is `.csv`: writes `summary` table for dict input, or the DataFrame itself.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(report, pd.DataFrame):
        if path.suffix.lower() not in {".csv", ".xlsx"}:
            path = path.with_suffix(".csv")
        if path.suffix.lower() == ".xlsx":
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                report.to_excel(writer, sheet_name="summary", index=False)
        else:
            report.to_csv(path, index=False)
        return path

    if path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, df in report.items():
                if df is None:
                    continue
                safe_sheet = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_sheet, index=False)
        return path

    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    summary_df = report.get("summary")
    if summary_df is None:
        raise ValueError("CSV export for dict report requires a 'summary' table")
    summary_df.to_csv(path, index=False)
    return path
