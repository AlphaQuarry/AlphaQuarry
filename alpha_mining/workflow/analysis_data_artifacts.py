from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from factor_research import SampleSplitConfig, assign_phase, calculate_icir


DISTRIBUTION_HISTOGRAM_KEY = "analysis_distribution_histogram"
IC_DECAY_KEY = "analysis_ic_decay"


def build_distribution_histogram_table(
    df_step2: pd.DataFrame | None,
    factor_cols: Sequence[str],
    *,
    sample_split_config: SampleSplitConfig | None = None,
    bins: int = 50,
) -> pd.DataFrame:
    """Build compact per-phase factor distribution histograms without storing raw values."""
    columns = [
        "factor",
        "phase",
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_mid",
        "count",
        "total_count",
    ]
    if df_step2 is None or df_step2.empty or "trade_date" not in df_step2.columns:
        return pd.DataFrame(columns=columns)

    factors = [str(factor) for factor in factor_cols if str(factor) in df_step2.columns]
    if not factors:
        return pd.DataFrame(columns=columns)

    work = assign_phase(
        df_step2[["trade_date"] + factors].copy(),
        config=sample_split_config or SampleSplitConfig(),
        date_col="trade_date",
    )
    phase_order = [phase for phase in ["train", "val", "test"] if phase in set(work["sample_phase"].astype(str))]
    rows: list[dict[str, Any]] = []
    n_bins = max(5, min(int(bins), 100))

    for factor in factors:
        values = pd.to_numeric(work[factor], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        edges = _histogram_edges(finite, n_bins)
        for phase in phase_order:
            phase_values = values[work["sample_phase"].astype(str) == phase]
            phase_values = phase_values[np.isfinite(phase_values)]
            if phase_values.empty:
                continue
            counts, _ = np.histogram(phase_values.to_numpy(dtype=float), bins=edges)
            total = int(counts.sum())
            for idx, count in enumerate(counts):
                left = float(edges[idx])
                right = float(edges[idx + 1])
                rows.append(
                    {
                        "factor": factor,
                        "phase": phase,
                        "bin_index": int(idx),
                        "bin_left": left,
                        "bin_right": right,
                        "bin_mid": float((left + right) / 2.0),
                        "count": int(count),
                        "total_count": total,
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def build_phase_ic_decay_table(
    df_step2: pd.DataFrame | None,
    factor_cols: Sequence[str],
    *,
    return_col: str = "pct_chg",
    period: int = 1,
    max_lag: int = 10,
    sample_split_config: SampleSplitConfig | None = None,
) -> pd.DataFrame:
    """Build compact per-phase IC decay summaries."""
    columns = ["factor", "phase", "lag", "ic", "half_life", "ic_decay_rank_corr"]
    if df_step2 is None or df_step2.empty or "trade_date" not in df_step2.columns:
        return pd.DataFrame(columns=columns)
    factors = [str(factor) for factor in factor_cols if str(factor) in df_step2.columns]
    if not factors:
        return pd.DataFrame(columns=columns)

    work = assign_phase(
        df_step2.copy(),
        config=sample_split_config or SampleSplitConfig(),
        date_col="trade_date",
    )
    rows: list[dict[str, Any]] = []
    for phase in ["train", "val", "test"]:
        phase_frame = work[work["sample_phase"].astype(str) == phase].copy()
        if phase_frame.empty:
            continue
        try:
            _, _, lag_results = calculate_icir(
                phase_frame,
                factor_cols=factors,
                return_col=return_col,
                period=int(period),
                max_lag=max(0, int(max_lag)),
            )
        except Exception:
            continue
        for result in lag_results or []:
            factor = str(result.get("factor", "") or "")
            if not factor:
                continue
            half_life = result.get("half_life")
            decay_corr = result.get("ic_decay_rank_corr")
            for lag, value in enumerate(result.get("lag_ic_values", []) or []):
                rows.append(
                    {
                        "factor": factor,
                        "phase": phase,
                        "lag": int(lag),
                        "ic": _finite_or_nan(value),
                        "half_life": _finite_or_nan(half_life),
                        "ic_decay_rank_corr": _finite_or_nan(decay_corr),
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def _histogram_edges(values: pd.Series, bins: int) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if not np.isfinite(low) or not np.isfinite(high):
        return np.linspace(-0.5, 0.5, int(bins) + 1)
    if low == high:
        pad = max(abs(low) * 0.05, 0.5)
        low -= pad
        high += pad
    return np.linspace(low, high, int(bins) + 1)


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except Exception:
        return np.nan
