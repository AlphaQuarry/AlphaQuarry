from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Make project root importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_alalyze_lib import (
    calculate_icir,
    calculate_long_short_metrics,
    calculate_turnover_rate,
    factor_layer_analysis,
    filter_effective_factors,
    process_factor_data,
    process_future_return,
)


def run_single_factor_demo(
    data_path: str = "examples/sample_factor_data.csv",
    factor: str = "factor_value",
    period: int = 1,
    layers: int = 5,
) -> None:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Demo data not found: {path}")

    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["znz_code", "trade_date"]).reset_index(drop=True)

    if factor not in df.columns:
        raise ValueError(f"Factor column '{factor}' not in data columns: {list(df.columns)}")

    print("=== Single Factor Demo ===")
    print(f"data_path: {path}")
    print(f"factor: {factor}, period: {period}, layers: {layers}")
    print(f"rows: {len(df)}, stocks: {df['znz_code'].nunique()}, dates: {df['trade_date'].nunique()}")

    # 1) future return
    df = process_future_return(df, return_col="pct_chg", period=period)
    future_return_col = f"pct_chg_{period}d"
    print(f"[1/7] future return column created: {future_return_col}")

    # 2) preprocess
    cols = ["trade_date", "znz_code", "pct_chg", future_return_col, "circ_mv", factor]
    df_processed = process_factor_data(df[cols].copy(), [factor], market_value_column="circ_mv", is_timeseries=True)
    print(f"[2/7] process_factor_data done, rows={len(df_processed)}")

    # 3) IC/IR
    ic_df, summary_df = calculate_icir(df_processed, [factor], return_col="pct_chg", period=period)
    print("[3/7] calculate_icir done")
    if not summary_df.empty:
        print("IC summary:")
        print(summary_df.round(4).to_string(index=False))
    else:
        print("IC summary is empty (sample too small or data quality issue).")

    # 4) layer analysis
    layer_results = factor_layer_analysis(df_processed, [factor], return_col="pct_chg", period=period, layers=layers)
    print(f"[4/7] factor_layer_analysis done, available factors={list(layer_results.keys())}")

    # 5) long-short metrics
    long_short_metrics, layer_results_for_visualization = calculate_long_short_metrics(
        layer_results, period=period, direction_mode="by_ic_sign", ic_summary_df=summary_df,
    )
    print("[5/7] calculate_long_short_metrics done")
    if factor in long_short_metrics:
        print("Long-short metrics:")
        print(pd.Series(long_short_metrics[factor]).round(4).to_string())
    else:
        print("Long-short metrics unavailable for selected factor.")

    # 6) turnover
    turnover_results = calculate_turnover_rate(layer_results, period=period)
    print("[6/7] calculate_turnover_rate done")
    if factor in turnover_results:
        print("Turnover preview:")
        print(turnover_results[factor].head().round(4).to_string(index=False))
    else:
        print("Turnover unavailable for selected factor.")

    # 7) effective factor screening
    effective_df = filter_effective_factors(
        summary_df=summary_df,
        long_short_metrics=long_short_metrics,
        layer_results=layer_results,
        apply_filtering=True,
    )
    print("[7/7] filter_effective_factors done")
    if not effective_df.empty:
        print("Effective factors:")
        print(effective_df.round(4).to_string(index=False))
    else:
        print("No factor passed screening in this demo sample.")

    # keep this variable to show the full pipeline output is available for plotting/debug
    _ = layer_results_for_visualization
    print("=== Demo Finished ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-factor demo pipeline on sample data.")
    parser.add_argument("--data-path", default="examples/sample_factor_data.csv", help="CSV input path")
    parser.add_argument("--factor", default="factor_value", help="Factor column to test")
    parser.add_argument("--period", type=int, default=1, help="Future return period")
    parser.add_argument("--layers", type=int, default=5, help="Layer count")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_single_factor_demo(
        data_path=args.data_path,
        factor=args.factor,
        period=args.period,
        layers=args.layers,
    )
