from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    deltas_path = Path(str(args.grid_deltas_csv)).resolve()
    if not deltas_path.exists():
        raise FileNotFoundError(f"grid deltas file not found: {deltas_path}")

    deltas_df = pd.read_csv(deltas_path)
    if deltas_df.empty:
        raise ValueError(f"grid deltas file is empty: {deltas_path}")

    rank_df = _load_rank_df(Path(str(args.grid_rank_csv)).resolve(), deltas_df=deltas_df)
    overall = _build_overall_metrics(deltas_df=deltas_df)
    gates = _evaluate_gates(overall=overall, args=args)
    grid_rank = _build_grid_rank_with_pass_flags(rank_df=rank_df, args=args)

    out_dir = Path(str(args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "phase32_convergence_summary.csv"
    grid_csv = out_dir / "phase32_grid_rank_with_gate.csv"
    report_md = out_dir / "phase32_convergence_report.md"

    pd.DataFrame([overall | gates]).to_csv(summary_csv, index=False)
    grid_rank.to_csv(grid_csv, index=False)
    report_md.write_text(
        _build_report(
            overall=overall,
            gates=gates,
            grid_rank=grid_rank,
            deltas_path=deltas_path,
            rank_path=Path(str(args.grid_rank_csv)).resolve(),
            args=args,
        ),
        encoding="utf-8",
    )

    print(f"[phase32] summary={summary_csv.as_posix()}")
    print(f"[phase32] grid_rank={grid_csv.as_posix()}")
    print(f"[phase32] report={report_md.as_posix()}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3.2 convergence gate from A/B grid outputs.")
    parser.add_argument("--grid-deltas-csv", default="artifacts/dev/ab_9runs_deltas.csv")
    parser.add_argument(
        "--grid-rank-csv",
        default="",
        help="Optional ab_9runs_grid_rank.csv. If missing, aggregate from deltas.",
    )
    parser.add_argument("--out-dir", default="artifacts/dev")
    parser.add_argument("--min-overall-win-rate", type=float, default=0.55)
    parser.add_argument("--min-quality-win-rate", type=float, default=0.50)
    parser.add_argument("--min-mechanism-pass-rate", type=float, default=1.00)
    parser.add_argument("--min-delta-score-mean", type=float, default=0.0)
    parser.add_argument("--min-delta-positive-ratio", type=float, default=0.0)
    parser.add_argument("--max-delta-turnover-mean", type=float, default=0.03)
    parser.add_argument("--min-mutation-ratio", type=float, default=0.02)
    parser.add_argument("--max-mutation-ratio", type=float, default=0.18)
    return parser


def _load_rank_df(rank_path: Path, deltas_df: pd.DataFrame) -> pd.DataFrame:
    if rank_path.exists():
        rank_df = pd.read_csv(rank_path)
        if not rank_df.empty:
            return rank_df
    grouped = (
        deltas_df.groupby("grid", as_index=False)
        .agg(
            runs=("grid", "size"),
            overall_wins=(
                "overall_win",
                lambda s: int(pd.Series(s).astype(bool).sum()),
            ),
            quality_wins=(
                "quality_win",
                lambda s: int(pd.Series(s).astype(bool).sum()),
            ),
            mechanism_passes=(
                "mechanism_pass",
                lambda s: int(pd.Series(s).astype(bool).sum()),
            ),
            avg_delta_topn_score_mean=("delta_topn_score_mean", "mean"),
            avg_delta_topn_score_median=("delta_topn_score_median", "mean"),
            avg_delta_topn_positive_ratio=("delta_topn_positive_ratio", "mean"),
            avg_delta_topn_turnover_mean=("delta_topn_turnover_mean", "mean"),
            avg_mutation_ratio=("mutation_ratio", "mean"),
            avg_fragment_cooldown_ratio=("fragment_cooldown_ratio", "mean"),
        )
        .copy()
    )
    grouped["overall_win_rate"] = grouped["overall_wins"] / grouped["runs"].clip(lower=1)
    grouped["quality_win_rate"] = grouped["quality_wins"] / grouped["runs"].clip(lower=1)
    grouped["mechanism_pass_rate"] = grouped["mechanism_passes"] / grouped["runs"].clip(lower=1)
    return grouped


def _build_overall_metrics(deltas_df: pd.DataFrame) -> dict[str, Any]:
    work = deltas_df.copy()
    for col in ["overall_win", "quality_win", "mechanism_pass"]:
        if col in work.columns:
            work[col] = work[col].astype(str).str.lower().isin(["1", "true", "yes"])
        else:
            work[col] = False
    metrics = {
        "runs": int(len(work)),
        "overall_win_rate": float(work["overall_win"].mean()) if len(work) else 0.0,
        "quality_win_rate": float(work["quality_win"].mean()) if len(work) else 0.0,
        "mechanism_pass_rate": float(work["mechanism_pass"].mean()) if len(work) else 0.0,
        "avg_delta_topn_score_mean": _mean_or_zero(work.get("delta_topn_score_mean")),
        "avg_delta_topn_score_median": _mean_or_zero(work.get("delta_topn_score_median")),
        "avg_delta_topn_positive_ratio": _mean_or_zero(work.get("delta_topn_positive_ratio")),
        "avg_delta_topn_turnover_mean": _mean_or_zero(work.get("delta_topn_turnover_mean")),
        "avg_mutation_ratio": _mean_or_zero(work.get("mutation_ratio")),
        "avg_fragment_cooldown_ratio": _mean_or_zero(work.get("fragment_cooldown_ratio")),
    }
    return metrics


def _evaluate_gates(overall: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    g1 = float(overall["overall_win_rate"]) >= float(args.min_overall_win_rate)
    g2 = float(overall["quality_win_rate"]) >= float(args.min_quality_win_rate)
    g3 = float(overall["mechanism_pass_rate"]) >= float(args.min_mechanism_pass_rate)
    g4 = float(overall["avg_delta_topn_score_mean"]) >= float(args.min_delta_score_mean)
    g5 = float(overall["avg_delta_topn_positive_ratio"]) >= float(args.min_delta_positive_ratio)
    g6 = float(overall["avg_delta_topn_turnover_mean"]) <= float(args.max_delta_turnover_mean)
    g7 = float(args.min_mutation_ratio) <= float(overall["avg_mutation_ratio"]) <= float(args.max_mutation_ratio)
    passed = bool(g1 and g2 and g3 and g4 and g5 and g6 and g7)
    return {
        "gate_overall_win_rate": bool(g1),
        "gate_quality_win_rate": bool(g2),
        "gate_mechanism_pass_rate": bool(g3),
        "gate_delta_score_mean": bool(g4),
        "gate_delta_positive_ratio": bool(g5),
        "gate_delta_turnover_mean": bool(g6),
        "gate_mutation_ratio_band": bool(g7),
        "phase32_converged": bool(passed),
    }


def _build_grid_rank_with_pass_flags(rank_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = rank_df.copy()
    for col in [
        "overall_win_rate",
        "quality_win_rate",
        "mechanism_pass_rate",
        "avg_delta_topn_score_mean",
        "avg_delta_topn_positive_ratio",
        "avg_delta_topn_turnover_mean",
        "avg_mutation_ratio",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out["gate_pass"] = (
        (out["overall_win_rate"] >= float(args.min_overall_win_rate))
        & (out["quality_win_rate"] >= float(args.min_quality_win_rate))
        & (out["mechanism_pass_rate"] >= float(args.min_mechanism_pass_rate))
        & (out["avg_delta_topn_score_mean"] >= float(args.min_delta_score_mean))
        & (out["avg_delta_topn_positive_ratio"] >= float(args.min_delta_positive_ratio))
        & (out["avg_delta_topn_turnover_mean"] <= float(args.max_delta_turnover_mean))
        & (out["avg_mutation_ratio"] >= float(args.min_mutation_ratio))
        & (out["avg_mutation_ratio"] <= float(args.max_mutation_ratio))
    )
    out["rank_score"] = (
        out["overall_win_rate"] * 100.0
        + out["quality_win_rate"] * 60.0
        + out["mechanism_pass_rate"] * 20.0
        + out["avg_delta_topn_score_mean"] * 10.0
        + out["avg_delta_topn_positive_ratio"] * 10.0
        - out["avg_delta_topn_turnover_mean"] * 8.0
        - out["avg_fragment_cooldown_ratio"] * 2.0
    )
    out = out.sort_values(["gate_pass", "rank_score"], ascending=[False, False]).reset_index(drop=True)
    return out


def _build_report(
    overall: dict[str, Any],
    gates: dict[str, Any],
    grid_rank: pd.DataFrame,
    deltas_path: Path,
    rank_path: Path,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Phase 3.2 Convergence Report",
        "",
        f"- deltas_csv: `{deltas_path.as_posix()}`",
        f"- rank_csv: `{rank_path.as_posix()}`",
        "",
        "## Acceptance Gates",
        "",
        f"- gate_overall_win_rate (>= {float(args.min_overall_win_rate):.2f}): `{bool(gates['gate_overall_win_rate'])}`",
        f"- gate_quality_win_rate (>= {float(args.min_quality_win_rate):.2f}): `{bool(gates['gate_quality_win_rate'])}`",
        f"- gate_mechanism_pass_rate (>= {float(args.min_mechanism_pass_rate):.2f}): `{bool(gates['gate_mechanism_pass_rate'])}`",
        f"- gate_delta_score_mean (>= {float(args.min_delta_score_mean):.4f}): `{bool(gates['gate_delta_score_mean'])}`",
        f"- gate_delta_positive_ratio (>= {float(args.min_delta_positive_ratio):.4f}): `{bool(gates['gate_delta_positive_ratio'])}`",
        f"- gate_delta_turnover_mean (<= {float(args.max_delta_turnover_mean):.4f}): `{bool(gates['gate_delta_turnover_mean'])}`",
        (
            f"- gate_mutation_ratio_band ({float(args.min_mutation_ratio):.3f}~{float(args.max_mutation_ratio):.3f}): "
            f"`{bool(gates['gate_mutation_ratio_band'])}`"
        ),
        "",
        f"- final_phase32_converged: `{bool(gates['phase32_converged'])}`",
        "",
        "## Overall Metrics",
        "",
        f"- runs: `{int(overall['runs'])}`",
        f"- overall_win_rate: `{float(overall['overall_win_rate']):.4f}`",
        f"- quality_win_rate: `{float(overall['quality_win_rate']):.4f}`",
        f"- mechanism_pass_rate: `{float(overall['mechanism_pass_rate']):.4f}`",
        f"- avg_delta_topn_score_mean: `{float(overall['avg_delta_topn_score_mean']):.6f}`",
        f"- avg_delta_topn_score_median: `{float(overall['avg_delta_topn_score_median']):.6f}`",
        f"- avg_delta_topn_positive_ratio: `{float(overall['avg_delta_topn_positive_ratio']):.6f}`",
        f"- avg_delta_topn_turnover_mean: `{float(overall['avg_delta_topn_turnover_mean']):.6f}`",
        f"- avg_mutation_ratio: `{float(overall['avg_mutation_ratio']):.6f}`",
        f"- avg_fragment_cooldown_ratio: `{float(overall['avg_fragment_cooldown_ratio']):.6f}`",
        "",
        "## Grid Ranking",
        "",
    ]
    if grid_rank.empty:
        lines.append("No grid rows.")
    else:
        cols = [
            "grid",
            "overall_win_rate",
            "quality_win_rate",
            "mechanism_pass_rate",
            "avg_delta_topn_score_mean",
            "avg_delta_topn_positive_ratio",
            "avg_delta_topn_turnover_mean",
            "avg_mutation_ratio",
            "gate_pass",
            "rank_score",
        ]
        view = grid_rank[[c for c in cols if c in grid_rank.columns]].copy()
        lines.extend(_df_to_markdown_table(view))

    if not bool(gates["phase32_converged"]):
        lines.extend(
            [
                "",
                "## Next Action",
                "",
                "Phase 3.2 未达标。建议先继续参数网格与窗口扩展验证，不建议直接进入 Phase 4 平台化。",
            ]
        )
    return "\n".join(lines) + "\n"


def _mean_or_zero(series: Any) -> float:
    if series is None:
        return 0.0
    work = pd.to_numeric(pd.Series(series), errors="coerce")
    if work.dropna().empty:
        return 0.0
    return float(work.mean())


def _df_to_markdown_table(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return ["No rows."]
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


if __name__ == "__main__":
    main()
