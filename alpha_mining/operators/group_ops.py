from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import OperatorRegistry


def register_operators(registry: OperatorRegistry) -> None:
    registry.register("group_rank", _group_rank)
    registry.register("group_zscore", _group_zscore)
    registry.register("group_mean", _group_mean)
    registry.register("group_sum", _group_sum)
    registry.register("group_median", _group_median)
    registry.register("group_scale", _group_scale)
    registry.register("group_neutralize", _group_neutralize)
    registry.register("group_normalize", _group_normalize)
    registry.register("densify", _densify)
    registry.register("bucket", _bucket)
    registry.register("group_cartesian_product", _group_cartesian_product)


def _align_panels(
    value_panel: pd.DataFrame,
    group_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_idx = value_panel.index.intersection(group_panel.index)
    common_cols = value_panel.columns.intersection(group_panel.columns)
    return (
        value_panel.reindex(index=common_idx, columns=common_cols),
        group_panel.reindex(index=common_idx, columns=common_cols),
    )


def _stack_compat(panel: pd.DataFrame) -> pd.Series:
    try:
        return panel.stack(dropna=False)
    except ValueError as exc:
        if "dropna must be unspecified" in str(exc):
            return panel.stack(future_stack=True)
        raise
    except TypeError:
        return panel.stack()


def _group_apply(value_panel: pd.DataFrame, group_panel: pd.DataFrame, fn):
    x_aligned, g_aligned = _align_panels(value_panel, group_panel)
    out = pd.DataFrame(np.nan, index=x_aligned.index, columns=x_aligned.columns, dtype=float)
    for dt in x_aligned.index:
        x = x_aligned.loc[dt]
        g = g_aligned.loc[dt]
        valid = x.notna() & g.notna()
        if not bool(valid.any()):
            continue
        x_valid = x[valid].astype(float)
        g_valid = g[valid]
        out_row = fn(x_valid, g_valid)
        out.loc[dt, out_row.index] = np.asarray(out_row, dtype=float)
    return out


def _group_rank(x: pd.DataFrame, group_panel: pd.DataFrame) -> pd.DataFrame:
    def fn(xv: pd.Series, gv: pd.Series):
        return xv.groupby(gv, sort=False).rank(pct=True)

    return _group_apply(x, group_panel, fn)


def _group_zscore(x: pd.DataFrame, group_panel: pd.DataFrame) -> pd.DataFrame:
    def fn(xv: pd.Series, gv: pd.Series):
        grp = xv.groupby(gv, sort=False)
        return (xv - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)

    return _group_apply(x, group_panel, fn)


def _group_mean(x: pd.DataFrame, group_panel: pd.DataFrame) -> pd.DataFrame:
    def fn(xv: pd.Series, gv: pd.Series):
        return xv.groupby(gv, sort=False).transform("mean")

    return _group_apply(x, group_panel, fn)


def _group_sum(x: pd.DataFrame, group_panel: pd.DataFrame) -> pd.DataFrame:
    def fn(xv: pd.Series, gv: pd.Series):
        return xv.groupby(gv, sort=False).transform("sum")

    return _group_apply(x, group_panel, fn)


def _group_median(x: pd.DataFrame, group_panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date group median broadcast back to each member."""

    def fn(xv: pd.Series, gv: pd.Series):
        return xv.groupby(gv, sort=False).transform("median")

    return _group_apply(x, group_panel, fn)


def _group_scale(x: pd.DataFrame, group_panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date group abs-sum scaling with zero-denominator guard."""

    def fn(xv: pd.Series, gv: pd.Series):
        denom = xv.abs().groupby(gv, sort=False).transform("sum").replace(0, np.nan)
        return xv / denom

    return _group_apply(x, group_panel, fn)


def _group_neutralize(x: pd.DataFrame, group_panel: pd.DataFrame) -> pd.DataFrame:
    def fn(xv: pd.Series, gv: pd.Series):
        grp_mean = xv.groupby(gv, sort=False).transform("mean")
        return xv - grp_mean

    return _group_apply(x, group_panel, fn)


def _group_normalize(x: pd.DataFrame, group_panel: pd.DataFrame) -> pd.DataFrame:
    def fn(xv: pd.Series, gv: pd.Series):
        centered = xv - xv.groupby(gv, sort=False).transform("mean")
        denom = centered.abs().groupby(gv, sort=False).transform("sum").replace(0, np.nan)
        return centered / denom

    return _group_apply(x, group_panel, fn)


def _densify(group_panel: pd.DataFrame) -> pd.DataFrame:
    out = group_panel.copy()
    for dt in out.index:
        labels = out.loc[dt].dropna().astype(str)
        mapping = {k: i for i, k in enumerate(sorted(labels.unique()), start=1)}
        out.loc[dt, labels.index] = labels.map(mapping).values
    return out


def _bucket(x: pd.DataFrame, range: str = "0.1,1,0.1") -> pd.DataFrame:
    try:
        parts = [float(v) for v in str(range).split(",")]
        if len(parts) != 3:
            raise ValueError("bucket range must be start,end,step")
        start, end, step = parts
        if step <= 0 or end <= start:
            raise ValueError("bucket range must satisfy end > start and step > 0")
    except Exception:
        return pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    bins = np.arange(start, end + step, step)
    stacked = _stack_compat(x)
    bucketed = pd.cut(stacked, bins=bins, labels=False, include_lowest=True)
    out = bucketed.unstack(level=1)
    return out.reindex(index=x.index, columns=x.columns)


def _group_cartesian_product(group_a: pd.DataFrame, group_b: pd.DataFrame) -> pd.DataFrame:
    common_idx = group_a.index.intersection(group_b.index)
    common_cols = group_a.columns.intersection(group_b.columns)
    a = group_a.reindex(index=common_idx, columns=common_cols)
    b = group_b.reindex(index=common_idx, columns=common_cols)
    out = pd.DataFrame(np.nan, index=common_idx, columns=common_cols, dtype=object)
    valid = a.notna() & b.notna()
    if bool(valid.any().any()):
        combined = a.astype(str).add("__").add(b.astype(str))
        out[valid] = combined[valid]
    return out.reindex(index=group_a.index, columns=group_a.columns)
