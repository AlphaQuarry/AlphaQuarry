from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

from ..atomic_io import atomic_write_dataframe_csv, atomic_write_json


VISUALIZATION_MANIFEST_KEY = "visualization_manifest"
MANIFEST_COLUMNS = [
    "plot_id",
    "scope",
    "factor",
    "category",
    "title",
    "relative_path",
    "width",
    "height",
    "sort_order",
    "created_at_utc",
    "source",
]
SOURCE_NAME = "alpha_mining.workflow.visualization_artifacts"


def save_factor_visualization_artifacts(
    analysis_dir: str | Path,
    factor_cols: Sequence[str],
    df_step2: pd.DataFrame | None = None,
    ic_df: pd.DataFrame | None = None,
    summary_df: pd.DataFrame | None = None,
    lag_analysis_results: Sequence[dict[str, Any]] | None = None,
    layer_results: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Render static per-factor visualization PNG artifacts and write a manifest."""
    root = Path(analysis_dir)
    root.mkdir(parents=True, exist_ok=True)
    if plt is None:
        empty = pd.DataFrame(columns=MANIFEST_COLUMNS)
        atomic_write_dataframe_csv(root / "visualization_manifest.csv", empty, index=False, backup=True)
        return empty
    factors = [str(factor) for factor in factor_cols if str(factor)]
    rows: list[dict[str, Any]] = []
    created_at = _utc_now()
    summary_by_factor = _summary_by_factor(summary_df)
    lag_by_factor = _lag_by_factor(lag_analysis_results)

    for factor in factors:
        renderers = [
            (
                "distribution",
                "distribution",
                "Factor Distribution",
                10,
                _plot_distribution,
            ),
            ("ic_overview", "ic", "IC Overview", 20, _plot_ic_overview),
            ("ic_decay", "ic", "IC Decay", 30, _plot_ic_decay),
            ("yearly_ic", "ic", "Yearly IC Mean", 40, _plot_yearly_ic),
            (
                "layer_terminal",
                "layer",
                "Layer Terminal Return",
                50,
                _plot_layer_terminal,
            ),
        ]
        for plot_name, category, title_suffix, sort_order, renderer in renderers:
            fig = None
            try:
                fig = renderer(
                    factor=factor,
                    df_step2=df_step2,
                    ic_df=ic_df,
                    summary_row=summary_by_factor.get(factor, {}),
                    lag_result=lag_by_factor.get(factor),
                    layer_frame=(layer_results or {}).get(factor),
                )
                if fig is None:
                    continue
                rows.append(
                    _save_figure(
                        fig=fig,
                        analysis_dir=root,
                        factor=factor,
                        plot_name=plot_name,
                        category=category,
                        title=f"{factor} {title_suffix}",
                        sort_order=sort_order,
                        created_at_utc=created_at,
                    )
                )
            except Exception:
                # Visualization artifacts should never make the analysis run fail.
                continue
            finally:
                if fig is not None:
                    plt.close(fig)

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    atomic_write_dataframe_csv(root / "visualization_manifest.csv", manifest, index=False, backup=True)
    return manifest


def attach_visualization_manifest_to_analysis_meta(
    analysis_meta_path: str | Path,
    visualization_manifest_path: str | Path,
) -> dict[str, Any]:
    meta_path = Path(analysis_meta_path)
    payload: dict[str, Any] = {}
    if meta_path.exists():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    table_paths = dict(payload.get("table_paths") or {})
    table_paths[VISUALIZATION_MANIFEST_KEY] = Path(visualization_manifest_path).as_posix()
    payload["table_paths"] = table_paths
    atomic_write_json(meta_path, payload, backup=True)
    return payload


def _plot_distribution(
    *,
    factor: str,
    df_step2: pd.DataFrame | None,
    ic_df: pd.DataFrame | None,
    summary_row: dict[str, Any],
    lag_result: dict[str, Any] | None,
    layer_frame: pd.DataFrame | None,
) -> plt.Figure | None:
    if df_step2 is None or df_step2.empty or factor not in df_step2.columns:
        return None
    values = pd.to_numeric(df_step2[factor], errors="coerce").dropna()
    if values.empty:
        return None
    if len(values) > 100_000:
        values = values.sample(n=100_000, random_state=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(values, bins=50, color="#2563eb", alpha=0.75, edgecolor="white")
    ax.axvline(
        values.mean(),
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label=f"mean={values.mean():.4g}",
    )
    ax.set_title(f"{factor} Distribution")
    ax.set_xlabel("factor value")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_ic_overview(
    *,
    factor: str,
    df_step2: pd.DataFrame | None,
    ic_df: pd.DataFrame | None,
    summary_row: dict[str, Any],
    lag_result: dict[str, Any] | None,
    layer_frame: pd.DataFrame | None,
) -> plt.Figure | None:
    data = _factor_ic_frame(ic_df, factor)
    if data is None or data.empty:
        return None
    ic_col = f"{factor}_ic"
    values = data[ic_col].dropna()
    if values.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(data["trade_date"], data[ic_col], linewidth=0.8, alpha=0.7, label="IC")
    axes[0].plot(
        data["trade_date"],
        data[ic_col].rolling(22, min_periods=1).mean(),
        linewidth=1.1,
        label="MA22",
    )
    axes[0].axhline(0, color="#64748b", linewidth=0.8, linestyle="--")
    axes[0].set_title(f"IC Time Series (IR={_fmt(summary_row.get('ir'))})")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].hist(values, bins=40, color="#0f766e", alpha=0.75, edgecolor="white")
    axes[1].axvline(
        values.mean(),
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label=f"mean={values.mean():.4g}",
    )
    axes[1].set_title("IC Distribution")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="best")

    cumulative = values.cumsum()
    axes[2].plot(data.loc[values.index, "trade_date"], cumulative, color="#16a34a", linewidth=1.2)
    axes[2].axhline(0, color="#64748b", linewidth=0.8, linestyle="--")
    axes[2].set_title("Cumulative IC")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.25)
    fig.suptitle(f"{factor} IC Overview", y=1.02)
    fig.tight_layout()
    return fig


def _plot_ic_decay(
    *,
    factor: str,
    df_step2: pd.DataFrame | None,
    ic_df: pd.DataFrame | None,
    summary_row: dict[str, Any],
    lag_result: dict[str, Any] | None,
    layer_frame: pd.DataFrame | None,
) -> plt.Figure | None:
    if not lag_result:
        return None
    lag_values = [float(v) for v in lag_result.get("lag_ic_values", []) if _is_finite(v)]
    if not lag_values:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(lag_values)), lag_values, color="#38bdf8", alpha=0.8)
    half_life = lag_result.get("half_life")
    if _is_finite(half_life):
        ax.axvline(
            float(half_life),
            color="#dc2626",
            linestyle="--",
            linewidth=1,
            label=f"half-life={half_life}",
        )
        ax.legend(loc="best")
    ax.axhline(0, color="#64748b", linewidth=0.8)
    ax.set_title(f"{factor} IC Decay")
    ax.set_xlabel("lag")
    ax.set_ylabel("IC")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_yearly_ic(
    *,
    factor: str,
    df_step2: pd.DataFrame | None,
    ic_df: pd.DataFrame | None,
    summary_row: dict[str, Any],
    lag_result: dict[str, Any] | None,
    layer_frame: pd.DataFrame | None,
) -> plt.Figure | None:
    data = _factor_ic_frame(ic_df, factor)
    if data is None or data.empty:
        return None
    ic_col = f"{factor}_ic"
    work = data.copy()
    work["year"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.year
    yearly = work.groupby("year")[ic_col].mean().dropna()
    if yearly.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly.index.astype(str), yearly.values, color="#0f766e", alpha=0.85)
    ax.axhline(0, color="#0f172a", linewidth=0.8)
    ax.set_title(f"{factor} Yearly IC Mean")
    ax.set_xlabel("year")
    ax.set_ylabel("IC mean")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_layer_terminal(
    *,
    factor: str,
    df_step2: pd.DataFrame | None,
    ic_df: pd.DataFrame | None,
    summary_row: dict[str, Any],
    lag_result: dict[str, Any] | None,
    layer_frame: pd.DataFrame | None,
) -> plt.Figure | None:
    if (
        layer_frame is None
        or layer_frame.empty
        or "trade_date" not in layer_frame.columns
        or "layer" not in layer_frame.columns
    ):
        return None
    return_col = _infer_return_column(layer_frame)
    if return_col is None:
        return None
    work = layer_frame.copy()
    work[return_col] = pd.to_numeric(work[return_col], errors="coerce")
    work = work.dropna(subset=[return_col])
    if work.empty:
        return None
    daily = work.groupby(["trade_date", "layer"], as_index=False)[return_col].mean()
    wide = daily.pivot(index="trade_date", columns="layer", values=return_col).sort_index()
    if wide.empty:
        return None
    terminal = wide.cumsum().iloc[-1].dropna()
    terminal = terminal[terminal.index.astype(str) != "long_short"]
    if terminal.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(terminal.index.astype(str), terminal.values, color="#7c3aed", alpha=0.85)
    ax.axhline(0, color="#0f172a", linewidth=0.8)
    ax.set_title(f"{factor} Layer Terminal Return")
    ax.set_xlabel("layer")
    ax.set_ylabel("cumulative return")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _save_figure(
    *,
    fig: plt.Figure,
    analysis_dir: Path,
    factor: str,
    plot_name: str,
    category: str,
    title: str,
    sort_order: int,
    created_at_utc: str,
) -> dict[str, Any]:
    safe_factor = _safe_name(factor)
    image_dir = analysis_dir / "visualizations" / safe_factor
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{plot_name}.png"
    fig.savefig(image_path, dpi=120, bbox_inches="tight")
    width, height = fig.get_size_inches()
    return {
        "plot_id": f"{factor}__{plot_name}",
        "scope": "factor",
        "factor": factor,
        "category": category,
        "title": title,
        "relative_path": image_path.relative_to(analysis_dir).as_posix(),
        "width": int(round(float(width) * 120)),
        "height": int(round(float(height) * 120)),
        "sort_order": int(sort_order),
        "created_at_utc": created_at_utc,
        "source": SOURCE_NAME,
    }


def _summary_by_factor(summary_df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if summary_df is None or summary_df.empty or "factor" not in summary_df.columns:
        return {}
    return {str(row.get("factor")): row.to_dict() for _, row in summary_df.iterrows()}


def _lag_by_factor(
    lag_analysis_results: Sequence[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in lag_analysis_results or []:
        factor = str(row.get("factor", "") or "")
        if factor:
            out[factor] = dict(row)
    return out


def _factor_ic_frame(ic_df: pd.DataFrame | None, factor: str) -> pd.DataFrame | None:
    ic_col = f"{factor}_ic"
    if ic_df is None or ic_df.empty or "trade_date" not in ic_df.columns or ic_col not in ic_df.columns:
        return None
    work = ic_df[["trade_date", ic_col]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work[ic_col] = pd.to_numeric(work[ic_col], errors="coerce")
    work = work.dropna(subset=["trade_date", ic_col]).sort_values("trade_date", kind="mergesort")
    return work


def _infer_return_column(frame: pd.DataFrame) -> str | None:
    for col in ["return", "future_return", "layer_return", "mean_return"]:
        if col in frame.columns:
            return col
    excluded = {"trade_date", "layer", "factor"}
    numeric_cols = [col for col in frame.columns if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])]
    return str(numeric_cols[-1]) if numeric_cols else None


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "factor"


def _fmt(value: Any) -> str:
    return f"{float(value):.3f}" if _is_finite(value) else "nan"


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
