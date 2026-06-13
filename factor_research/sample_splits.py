from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SampleSplitConfig:
    train_start: str = "2016-01-01"
    train_end: str = "2024-12-31"
    validation_start: str = "2025-01-01"
    validation_end: str = "2025-12-31"
    oos_start: str = "2026-01-01"

    @property
    def val_start(self) -> str:
        return self.validation_start

    @property
    def val_end(self) -> str:
        return self.validation_end

    @property
    def test_start(self) -> str:
        return self.oos_start


@dataclass(frozen=True)
class PhaseWindow:
    key: str
    label: str
    start: str
    end: str | None
    available: bool = True
    visible_default: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LEGACY_TO_PHASE = {"validation": "val", "oos": "test"}
PHASE_TO_LEGACY = {"train": "train", "val": "validation", "test": "oos"}


def build_phase_windows(
    cfg: SampleSplitConfig | None = None,
    max_date: str | pd.Timestamp | None = None,
    min_date: str | pd.Timestamp | None = None,
    include_test: bool = True,
) -> list[PhaseWindow]:
    del min_date
    config = cfg or SampleSplitConfig()
    max_ts = _to_timestamp_or_none(max_date)
    windows = [
        PhaseWindow("train", "Train", config.train_start, config.train_end, True, True),
    ]
    if max_ts is None or max_ts >= pd.Timestamp(config.validation_start):
        windows.append(PhaseWindow("val", "Val", config.validation_start, config.validation_end, True, True))
    if bool(include_test) and (max_ts is None or max_ts >= pd.Timestamp(config.oos_start)):
        end = max_ts.strftime("%Y-%m-%d") if max_ts is not None else None
        windows.append(PhaseWindow("test", "Test", config.oos_start, end, True, False))
    return windows


def assign_phase(
    df: pd.DataFrame,
    date_col: str = "trade_date",
    config: SampleSplitConfig | None = None,
    output_col: str = "sample_phase",
    legacy_output_col: str | None = "sample_split",
) -> pd.DataFrame:
    cfg = config or SampleSplitConfig()
    out = df.copy()
    if out.empty or date_col not in out.columns:
        out[output_col] = ""
        if legacy_output_col:
            out[legacy_output_col] = ""
        return out
    dates = pd.to_datetime(out[date_col], errors="coerce")
    out[output_col] = ""
    train_start = pd.Timestamp(cfg.train_start)
    train_end = pd.Timestamp(cfg.train_end)
    val_start = pd.Timestamp(cfg.validation_start)
    val_end = pd.Timestamp(cfg.validation_end)
    oos_start = pd.Timestamp(cfg.oos_start)
    out.loc[(dates >= train_start) & (dates <= train_end), output_col] = "train"
    out.loc[(dates >= val_start) & (dates <= val_end), output_col] = "val"
    out.loc[dates >= oos_start, output_col] = "test"
    if legacy_output_col:
        out[legacy_output_col] = out[output_col].map(PHASE_TO_LEGACY).fillna("")
    return out


def assign_sample_split(
    df: pd.DataFrame,
    date_col: str = "trade_date",
    config: SampleSplitConfig | None = None,
    output_col: str = "sample_split",
) -> pd.DataFrame:
    out = assign_phase(
        df,
        date_col=date_col,
        config=config,
        output_col="_sample_phase",
        legacy_output_col=output_col,
    )
    return out.drop(columns=["_sample_phase"], errors="ignore")


def summarize_split_metrics(
    df: pd.DataFrame,
    factor_col: str = "factor",
    split_col: str = "sample_split",
    metric_cols: Sequence[str] = ("score_total", "ir", "ic_mean"),
) -> pd.DataFrame:
    if df is None or df.empty or factor_col not in df.columns or split_col not in df.columns:
        return pd.DataFrame()
    metrics = [str(c) for c in metric_cols if str(c) in df.columns]
    factors = list(dict.fromkeys(df[factor_col].astype(str).tolist()))
    rows: list[dict[str, object]] = []
    for factor in factors:
        g_factor = df[df[factor_col].astype(str) == factor]
        row: dict[str, object] = {"factor": factor}
        for split in ["train", "validation", "oos"]:
            g_split = g_factor[g_factor[split_col].astype(str) == split]
            row[f"{split}_obs"] = int(len(g_split))
            for metric in metrics:
                values = pd.to_numeric(g_split[metric], errors="coerce")
                row[f"{split}_{metric}_mean"] = float(values.mean(skipna=True)) if values.notna().any() else np.nan
        score_metric = (
            "score_total"
            if "score_total" in metrics
            else ("ir" if "ir" in metrics else (metrics[0] if metrics else ""))
        )
        for split in ["train", "validation", "oos"]:
            key = f"{split}_{score_metric}_mean" if score_metric else ""
            row[f"{split}_score"] = row.get(key, np.nan) if key else np.nan
        train_score = _to_float_or_nan(row.get("train_score"))
        validation_score = _to_float_or_nan(row.get("validation_score"))
        oos_score = _to_float_or_nan(row.get("oos_score"))
        row["validation_decay_ratio"] = _ratio_or_nan(validation_score, train_score)
        row["oos_decay_ratio"] = _ratio_or_nan(oos_score, train_score)
        warnings: list[str] = []
        if not np.isfinite(train_score):
            warnings.append("missing_train_score")
        if not np.isfinite(validation_score):
            warnings.append("missing_validation_score")
        if np.isfinite(train_score) and np.isfinite(validation_score) and train_score * validation_score < 0:
            warnings.append("validation_score_sign_flip")
        row["split_pass"] = len(warnings) == 0
        row["split_warning_reasons"] = "; ".join(warnings)
        rows.append(row)
    return pd.DataFrame(rows)


def _to_float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _ratio_or_nan(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _to_timestamp_or_none(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None or str(value) == "":
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)
