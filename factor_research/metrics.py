from __future__ import annotations

import pandas as pd


def restructure_factor_analysis_data(
    summary_dfs,
    long_short_metrics_dict,
    layer_results_dict,
    ic_dfs,
    lag_analysis_results_list,
    layer_results_for_visualization_dict,
):
    """Restructure scattered analysis outputs into one dict payload."""
    merged_summary_df = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()

    merged_lag_analysis_results = []
    for lag_results in lag_analysis_results_list:
        if isinstance(lag_results, list):
            merged_lag_analysis_results.extend(lag_results)
        elif lag_results is not None:
            merged_lag_analysis_results.append(lag_results)

    return {
        "merged_summary_df": merged_summary_df,
        "merged_long_short_metrics": long_short_metrics_dict,
        "merged_layer_results": layer_results_dict,
        "merged_ic_dfs": ic_dfs,
        "merged_lag_analysis_results": merged_lag_analysis_results,
        "merged_layer_results_for_visualization": layer_results_for_visualization_dict,
    }


def prepare_visualization_data_for_all_factors(structured_data: dict, factor_names: list[str]) -> dict:
    """Prepare merged IC/layer data for plotting selected factors."""
    ic_series_map: dict[str, pd.Series] = {}
    summary_dfs = []
    all_lag_analysis_results = []

    for factor_name in factor_names:
        factor_ic_df = None
        factor_summary_df = pd.DataFrame()

        if structured_data.get("merged_ic_dfs"):
            for df in structured_data["merged_ic_dfs"]:
                ic_col_name = f"{factor_name}_ic"
                if ic_col_name in df.columns:
                    factor_ic_df = df.set_index("trade_date")[ic_col_name].copy()
                    break

        merged_summary_df = structured_data.get("merged_summary_df", pd.DataFrame())
        if not merged_summary_df.empty:
            factor_summary_df = merged_summary_df[merged_summary_df["factor"] == factor_name].copy()

        if factor_ic_df is not None:
            ic_series_map[f"{factor_name}_ic"] = factor_ic_df
        if not factor_summary_df.empty:
            summary_dfs.append(factor_summary_df)

    merged_ic_df = None
    if ic_series_map:
        merged_ic_df = pd.DataFrame(ic_series_map).sort_index().reset_index()

    merged_summary_df = pd.concat(summary_dfs, ignore_index=True) if summary_dfs else pd.DataFrame()

    for lag_result in structured_data.get("merged_lag_analysis_results", []):
        if isinstance(lag_result, dict) and lag_result.get("factor") in factor_names:
            all_lag_analysis_results.append(lag_result)
        elif isinstance(lag_result, list):
            for sub_result in lag_result:
                if isinstance(sub_result, dict) and sub_result.get("factor") in factor_names:
                    all_lag_analysis_results.append(sub_result)

    layer_results_for_visualization = {}
    all_layer_results = structured_data.get("merged_layer_results_for_visualization", {})
    for factor_name in factor_names:
        if factor_name in all_layer_results:
            layer_results_for_visualization[factor_name] = all_layer_results[factor_name]

    return {
        "ic_df": merged_ic_df,
        "summary_df": merged_summary_df,
        "lag_analysis_results": all_lag_analysis_results,
        "layer_results_for_visualization": layer_results_for_visualization,
    }
