from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.adapters import to_factor_research_frame
from alpha_mining.workflow.analysis_cycle import (
    AnalysisLevelConfig,
    BatchAnalysisConfig,
    run_factor_analysis_batch,
)
from alpha_mining.workflow.universe_store import (
    load_universe_alpha_batch,
    load_universe_base_frame,
    load_universe_expression_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce stored alpha diagnostics on OOS dates for manual review.")
    parser.add_argument("--base-dir", default="data/alpha_universe_store")
    parser.add_argument("--universe", default="cn_all")
    parser.add_argument(
        "--alpha-names",
        default="",
        help="Comma-separated alpha names; defaults to all stored expressions",
    )
    parser.add_argument("--oos-start", default="2026-01-02")
    parser.add_argument("--oos-end", default="")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--code-col", default="code")
    parser.add_argument("--return-col", default="pct_chg")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--output", default="", help="Optional CSV path for the OOS review table")
    args = parser.parse_args()

    alpha_names = _resolve_alpha_names(
        base_dir=str(args.base_dir),
        universe_name=str(args.universe),
        alpha_names_arg=str(args.alpha_names or ""),
    )
    if not alpha_names:
        print("[oos] no alpha names found")
        return

    base_df = load_universe_base_frame(base_dir=str(args.base_dir), universe_name=str(args.universe))
    base_df = _filter_oos_dates(
        base_df,
        date_col=str(args.date_col),
        start=str(args.oos_start),
        end=str(args.oos_end),
    )
    if base_df.empty:
        print("[oos] no base rows in requested OOS window")
        return

    alpha_df = load_universe_alpha_batch(
        alpha_names=alpha_names,
        base_dir=str(args.base_dir),
        universe_name=str(args.universe),
        date_col=str(args.date_col),
        code_col=str(args.code_col),
    )
    alpha_df = _filter_oos_dates(
        alpha_df,
        date_col=str(args.date_col),
        start=str(args.oos_start),
        end=str(args.oos_end),
    )
    fr_input = to_factor_research_frame(
        raw_df=base_df,
        alpha_wide_df=alpha_df,
        date_col=str(args.date_col),
        code_col=str(args.code_col),
    )
    factor_cols = [name for name in alpha_names if name in fr_input.columns]
    if not factor_cols:
        print("[oos] no alpha value columns found in OOS window")
        return

    out = run_factor_analysis_batch(
        df_raw=fr_input,
        factor_cols=factor_cols,
        config=BatchAnalysisConfig(
            period=max(1, int(args.period)),
            layers=max(2, int(args.layers)),
            return_col=str(args.return_col),
            analysis_level=AnalysisLevelConfig(mode="light"),
            include_sample_split_analysis=False,
            include_double_sort=False,
            apply_tradability_constraints=False,
        ),
    )
    review = out.get("factor_metrics_df", pd.DataFrame()).copy()
    review["oos_start"] = str(args.oos_start)
    review["oos_end"] = str(args.oos_end or "")
    if str(args.output or "").strip():
        output = Path(str(args.output)).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        review.to_csv(output, index=False)
        print(f"[oos] wrote {len(review)} rows to {output}")
    else:
        print(review.to_csv(index=False))


def _resolve_alpha_names(base_dir: str, universe_name: str, alpha_names_arg: str) -> list[str]:
    requested = [x.strip() for x in alpha_names_arg.split(",") if x.strip()]
    if requested:
        return requested
    registry = load_universe_expression_registry(base_dir=base_dir, universe_name=universe_name)
    if registry.empty or "alpha_name" not in registry.columns:
        return []
    return registry["alpha_name"].dropna().astype(str).tolist()


def _filter_oos_dates(df: pd.DataFrame, date_col: str, start: str, end: str = "") -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    mask = dates >= pd.Timestamp(start)
    if str(end or "").strip():
        mask &= dates <= pd.Timestamp(end)
    return out.loc[mask].copy()


if __name__ == "__main__":
    main()
