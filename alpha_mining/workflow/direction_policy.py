from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from factor_research import SampleSplitConfig, assign_phase, build_phase_windows

# t 统计量阈值：95% 置信度对应 1.96
_DIRECTION_T_STAT_THRESHOLD: float = 1.96


def build_direction_policy_tables(
    *,
    ic_df: pd.DataFrame,
    factors: Sequence[str],
    sample_split_config: SampleSplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build official train-locked direction and phase-local diagnostics."""
    factor_list = [str(f) for f in factors if str(f)]
    if not factor_list:
        return pd.DataFrame(), pd.DataFrame()
    phased = _phase_ic_frame(ic_df, sample_split_config)
    policy_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    windows = build_phase_windows(sample_split_config, max_date=_max_date(phased), include_test=True)

    for factor in factor_list:
        col = f"{factor}_ic"
        if col not in phased.columns:
            policy_rows.append(_policy_row(factor, np.nan, 0.0, 0, "missing"))
            continue
        train_values = _clean_values(phased.loc[phased["sample_phase"].astype(str) == "train", col])
        source_phase = "train"
        values = train_values
        if values.empty:
            values = _clean_values(phased[col])
            source_phase = "full_sample_fallback"
        mean_ic = float(values.mean()) if not values.empty else np.nan
        ic_std = float(values.std()) if not values.empty else 0.0
        n_obs = int(len(values))
        policy_rows.append(_policy_row(factor, mean_ic, ic_std, n_obs, source_phase))

        for window in windows:
            phase_values = _clean_values(phased.loc[phased["sample_phase"].astype(str) == window.key, col])
            phase_mean = float(phase_values.mean()) if not phase_values.empty else np.nan
            sign, _ = _direction_sign_with_confidence(phase_mean)
            phase_rows.append(
                {
                    "factor": factor,
                    "phase": window.key,
                    "direction_policy": "phase_local",
                    "direction_sign": sign,
                    "direction_ic_mean": phase_mean,
                    "direction_obs": int(len(phase_values)),
                    "best_layer_direction_phase_local": "top" if sign >= 0 else "bottom",
                }
            )
    return pd.DataFrame(policy_rows), pd.DataFrame(phase_rows)


def direction_sign_map(direction_policy_df: pd.DataFrame | None) -> dict[str, float]:
    if direction_policy_df is None or direction_policy_df.empty:
        return {}
    if "factor" not in direction_policy_df.columns or "direction_sign" not in direction_policy_df.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in direction_policy_df.iterrows():
        factor = str(row.get("factor") or "")
        if not factor:
            continue
        try:
            sign = float(row.get("direction_sign"))
        except Exception:
            sign = 1.0
        out[factor] = 1.0 if not np.isfinite(sign) or sign >= 0 else -1.0
    return out


def _phase_ic_frame(ic_df: pd.DataFrame | None, sample_split_config: SampleSplitConfig | None) -> pd.DataFrame:
    if ic_df is None or ic_df.empty:
        return pd.DataFrame(columns=["trade_date", "sample_phase"])
    work = ic_df.copy()
    if "trade_date" not in work.columns:
        work["trade_date"] = pd.NaT
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    return assign_phase(
        work,
        date_col="trade_date",
        config=sample_split_config,
        output_col="sample_phase",
        legacy_output_col=None,
    )


def _policy_row(factor: str, mean_ic: float, ic_std: float, n_obs: int, source_phase: str) -> dict[str, object]:
    sign, confidence = _direction_sign_with_confidence(mean_ic, ic_std, n_obs)
    return {
        "factor": factor,
        "direction_policy": "train_locked",
        "direction_source_phase": source_phase,
        "direction_sign": sign,
        "direction_ic_mean": mean_ic,
        "direction_ic_std": ic_std,
        "direction_obs": int(n_obs),
        "direction_confidence": confidence,
        "best_layer_direction_train_locked": "top" if sign >= 0 else "bottom",
    }


def _direction_sign_with_confidence(
    value: float,
    ic_std: float = 0.0,
    n_obs: int = 0,
    threshold: float = _DIRECTION_T_STAT_THRESHOLD,
) -> tuple[float, str]:
    """
    返回 (direction_sign, confidence_label)。

    direction_sign: 1.0 (top) 或 -1.0 (bottom)
    confidence_label: "high" / "low" / "uncertain" / "no_std"
    """
    try:
        x = float(value)
    except Exception:
        return 1.0, "uncertain"
    if not np.isfinite(x):
        return 1.0, "uncertain"

    # 无标准差信息时，回退到原始逻辑
    if ic_std <= 0 or n_obs <= 1:
        return (1.0 if x >= 0 else -1.0), "no_std"

    t_stat = abs(x) / (ic_std / np.sqrt(n_obs))
    if t_stat >= threshold:
        return (1.0 if x >= 0 else -1.0), "high"
    if t_stat >= threshold * 0.5:
        return (1.0 if x >= 0 else -1.0), "low"
    return 1.0, "uncertain"


def _direction_sign(value: float) -> float:
    """向后兼容：只返回符号，不返回置信度。"""
    sign, _ = _direction_sign_with_confidence(value)
    return sign


def _clean_values(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").dropna()


def _max_date(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame is None or frame.empty or "trade_date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())
