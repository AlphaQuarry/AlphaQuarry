from __future__ import annotations

import json
from pathlib import Path


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code,
    }


def build_notebook() -> dict:
    cells: list[dict] = []
    cells.append(
        md_cell(
            "# Factor Analyze Notebook (Refreshed)\n\n"
            "This notebook runs the full factor workflow step-by-step `[1/7]` to `[7/7]` "
            "and now supports multiple factors."
        )
    )
    cells.append(
        code_cell(
            "import warnings\n"
            "from pathlib import Path\n\n"
            "import pandas as pd\n\n"
            "from factor_alalyze_lib import *\n\n"
            "warnings.filterwarnings('ignore')\n"
            "pd.set_option('display.max_columns', None)\n"
            "print('Environment ready.')"
        )
    )

    cells.append(md_cell("## Config"))
    cells.append(
        code_cell(
            "DATA_PATH = Path('examples/sample_factor_data.csv')\n"
            "FACTOR_COLS = ['factor_value', 'factor_quality', 'factor_momentum']\n"
            "PERIOD = 1\n"
            "LAYERS = 5\n"
            "IS_TIMESERIES = True\n\n"
            "print({\n"
            "    'DATA_PATH': str(DATA_PATH),\n"
            "    'FACTOR_COLS': FACTOR_COLS,\n"
            "    'PERIOD': PERIOD,\n"
            "    'LAYERS': LAYERS,\n"
            "    'IS_TIMESERIES': IS_TIMESERIES,\n"
            "})"
        )
    )

    cells.append(md_cell("## Load Data"))
    cells.append(
        code_cell(
            "if not DATA_PATH.exists():\n"
            "    raise FileNotFoundError(f'Data file not found: {DATA_PATH}')\n\n"
            "df_raw = pd.read_csv(DATA_PATH)\n"
            "df_raw['trade_date'] = pd.to_datetime(df_raw['trade_date'])\n"
            "df_raw = df_raw.sort_values(['znz_code', 'trade_date']).reset_index(drop=True)\n\n"
            "required_cols = ['trade_date', 'znz_code', 'pct_chg', 'circ_mv'] + FACTOR_COLS\n"
            "missing = [c for c in required_cols if c not in df_raw.columns]\n"
            "if missing:\n"
            "    raise ValueError(f'Missing required columns: {missing}')\n\n"
            "print(f\"rows={len(df_raw)}, stocks={df_raw['znz_code'].nunique()}, dates={df_raw['trade_date'].nunique()}\")\n"
            "df_raw.head()"
        )
    )

    cells.append(md_cell("## [1/7] process_future_return"))
    cells.append(
        code_cell(
            "df_step1 = process_future_return(df_raw.copy(), return_col='pct_chg', period=PERIOD)\n"
            "FUTURE_RETURN_COL = f'pct_chg_{PERIOD}d'\n"
            "print(f'future return column = {FUTURE_RETURN_COL}')\n"
            "df_step1[['trade_date', 'znz_code', 'pct_chg', FUTURE_RETURN_COL]].head(10)"
        )
    )

    cells.append(md_cell("## [2/7] process_factor_data (multi-factor)"))
    cells.append(
        code_cell(
            "cols_step2 = ['trade_date', 'znz_code', 'pct_chg', FUTURE_RETURN_COL, 'circ_mv'] + FACTOR_COLS\n"
            "df_step2 = process_factor_data(\n"
            "    df_step1[cols_step2].copy(),\n"
            "    factor_cols=FACTOR_COLS,\n"
            "    market_value_column='circ_mv',\n"
            "    is_timeseries=IS_TIMESERIES,\n"
            ")\n"
            "print(f'processed rows={len(df_step2)}')\n"
            "df_step2[['trade_date', 'znz_code'] + FACTOR_COLS].head()"
        )
    )

    cells.append(md_cell("## [3/7] calculate_icir (multi-factor)"))
    cells.append(
        code_cell(
            "ic_df, summary_df = calculate_icir(\n"
            "    df_step2,\n"
            "    factor_cols=FACTOR_COLS,\n"
            "    return_col='pct_chg',\n"
            "    period=PERIOD,\n"
            ")\n"
            "display(summary_df.sort_values('ir', ascending=False).round(4))\n"
            "ic_df.head()"
        )
    )

    cells.append(md_cell("## [4/7] factor_layer_analysis (multi-factor)"))
    cells.append(
        code_cell(
            "layer_results = factor_layer_analysis(\n"
            "    df_step2,\n"
            "    factor_cols=FACTOR_COLS,\n"
            "    return_col='pct_chg',\n"
            "    period=PERIOD,\n"
            "    layers=LAYERS,\n"
            ")\n"
            "print('layer factors:', list(layer_results.keys()))\n"
            "next_factor = list(layer_results.keys())[0] if layer_results else None\n"
            "layer_results[next_factor].head() if next_factor else 'No layer results'"
        )
    )

    cells.append(md_cell("## [5/7] calculate_long_short_metrics (multi-factor)"))
    cells.append(
        code_cell(
            "long_short_metrics, layer_results_for_visualization = calculate_long_short_metrics(layer_results, period=PERIOD, direction_mode='by_ic_sign', ic_summary_df=summary_df)\n"
            "long_short_df = pd.DataFrame(long_short_metrics).T.reset_index().rename(columns={'index': 'factor'})\n"
            "display(long_short_df.round(4))"
        )
    )

    cells.append(md_cell("## [6/7] calculate_turnover_rate (multi-factor)"))
    cells.append(
        code_cell(
            "turnover_results = calculate_turnover_rate(layer_results, period=PERIOD)\n"
            "turnover_summary = []\n"
            "for factor, tr in turnover_results.items():\n"
            "    turnover_summary.append({\n"
            "        'factor': factor,\n"
            "        'avg_min_layer_turnover': tr['min_layer_turnover'].mean(),\n"
            "        'avg_max_layer_turnover': tr['max_layer_turnover'].mean(),\n"
            "    })\n"
            "display(pd.DataFrame(turnover_summary).round(4))"
        )
    )

    cells.append(md_cell("## [7/7] filter_effective_factors"))
    cells.append(
        code_cell(
            "effective_factors_df = filter_effective_factors(\n"
            "    summary_df=summary_df,\n"
            "    long_short_metrics=long_short_metrics,\n"
            "    layer_results=layer_results,\n"
            "    apply_filtering=True,\n"
            ")\n"
            "display(effective_factors_df.round(4))"
        )
    )

    cells.append(md_cell("## Diagnostics (New, Optional)"))
    cells.append(
        code_cell(
            "coverage = calculate_factor_coverage(df_step2, FACTOR_COLS)\n"
            "ic_stability_df = calculate_ic_stability(ic_df, FACTOR_COLS)\n"
            "ic_yearly_df = calculate_ic_time_breakdown(ic_df, FACTOR_COLS, freq='Y')\n"
            "ic_monthly_df = calculate_ic_time_breakdown(ic_df, FACTOR_COLS, freq='M')\n"
            "monotonicity = calculate_layer_monotonicity(layer_results)\n\n"
            "display(coverage['overall'].round(4))\n"
            "display(ic_stability_df.round(4))\n"
            "display(monotonicity['summary'].round(4))"
        )
    )

    cells.append(md_cell("## Holding Period Robustness (New, Optional)"))
    cells.append(
        code_cell(
            "ROBUST_PERIODS = [1, 5, 10, 20]\n"
            "robustness = analyze_holding_period_robustness(\n"
            "    df=df_raw,\n"
            "    factor_cols=FACTOR_COLS,\n"
            "    periods=ROBUST_PERIODS,\n"
            "    return_col='pct_chg',\n"
            "    layers=LAYERS,\n"
            "    market_value_column='circ_mv',\n"
            "    is_timeseries=IS_TIMESERIES,\n"
            ")\n"
            "period_comparison_df = robustness['comparison']\n"
            "display(period_comparison_df.round(4))"
        )
    )

    cells.append(md_cell("## Structured Report Export (New, Optional)"))
    cells.append(
        code_cell(
            "summary_report_df = build_factor_summary_report(\n"
            "    factor_cols=FACTOR_COLS,\n"
            "    coverage_overall_df=coverage['overall'],\n"
            "    ic_summary_df=summary_df,\n"
            "    monotonicity_summary_df=monotonicity['summary'],\n"
            "    long_short_metrics=long_short_metrics,\n"
            "    turnover_results=turnover_results,\n"
            "    effective_factors_df=effective_factors_df,\n"
            "    apply_filtering=True,\n"
            ")\n"
            "report = build_structured_report(\n"
            "    summary_report_df=summary_report_df,\n"
            "    ic_stability_df=ic_stability_df,\n"
            "    ic_yearly_df=ic_yearly_df,\n"
            "    ic_monthly_df=ic_monthly_df,\n"
            "    coverage_by_date_df=coverage['by_date'],\n"
            "    robustness_comparison_df=period_comparison_df,\n"
            ")\n"
            "display(summary_report_df.round(4))\n"
            "export_report(report, 'examples/outputs/factor_report.xlsx')\n"
            "export_report(report, 'examples/outputs/factor_report_summary.csv')\n"
            "print('Reports exported to examples/outputs/')"
        )
    )

    cells.append(md_cell("## Optional Plots"))
    cells.append(
        code_cell(
            "# visualize_ic_analysis(ic_df, summary_df)\n"
            "# visualize_factor_distribution(df_step2, FACTOR_COLS)\n"
            "# visualize_ic_yearly_bar(ic_df, FACTOR_COLS)\n"
            "# if layer_results_for_visualization:\n"
            "#     visualize_layer_analysis(layer_results_for_visualization, show_long_short=True)\n"
            "#     visualize_layer_terminal_values(layer_results)\n"
            "# if turnover_results:\n"
            "#     visualize_turnover_analysis(turnover_results)\n"
            "# if 'period_comparison_df' in globals():\n"
            "#     visualize_period_comparison(period_comparison_df, metric='ir')\n"
            "print('Uncomment plotting lines above to visualize.')"
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    Path("factor_analyze.ipynb").write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("factor_analyze.ipynb refreshed")


if __name__ == "__main__":
    main()
