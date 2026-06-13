from __future__ import annotations

import numpy as np
import pandas as pd


SUPPORTED_NEUTRALIZATION_MODES = ("NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY")
GROUP_NEUTRALIZATION_FIELD_MAP = {
    "SECTOR": "sector",
    "INDUSTRY": "industry",
    "SUBINDUSTRY": "subindustry",
}
_GROUP_ALIASES = {
    "GROUP:SECTOR": "SECTOR",
    "GROUP:INDUSTRY": "INDUSTRY",
    "GROUP:SUBINDUSTRY": "SUBINDUSTRY",
}


def normalize_neutralization_mode(value: str | None) -> str:
    mode = str(value or "NONE").strip().upper()
    if not mode:
        mode = "NONE"
    mode = _GROUP_ALIASES.get(mode, mode)
    if mode not in SUPPORTED_NEUTRALIZATION_MODES:
        supported = ", ".join(SUPPORTED_NEUTRALIZATION_MODES)
        raise ValueError(f"Unsupported neutralization mode: {mode}. Supported modes: {supported}.")
    return mode


def neutralization_group_field(value: str | None) -> str | None:
    mode = normalize_neutralization_mode(value)
    return GROUP_NEUTRALIZATION_FIELD_MAP.get(mode)


def apply_neutralization(
    alpha_panel: pd.DataFrame,
    neutralization: str = "NONE",
    group_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    mode = normalize_neutralization_mode(neutralization)
    if mode == "NONE":
        return alpha_panel
    if mode == "MARKET":
        return alpha_panel.sub(alpha_panel.mean(axis=1, skipna=True), axis=0)
    if group_panel is None:
        raise ValueError(f"neutralization={mode} requires group_panel")

    aligned_group = group_panel.reindex(index=alpha_panel.index, columns=alpha_panel.columns)
    vals_long = alpha_panel.stack().rename("_val")
    groups_long = aligned_group.stack().rename("_grp")
    combined = pd.DataFrame({"_val": vals_long, "_grp": groups_long})
    combined["_val"] = pd.to_numeric(combined["_val"], errors="coerce")
    valid = combined["_val"].notna() & combined["_grp"].notna()
    centered = pd.Series(np.nan, index=combined.index, dtype=float)
    if bool(valid.any()):
        valid_vals = combined.loc[valid, "_val"]
        # Use (date, group) composite key to demean within each date×group cell
        dates = combined.index.get_level_values(0)[valid]
        grps = combined.loc[valid, "_grp"].values
        group_key = pd.Series(list(zip(dates, grps)), index=valid_vals.index)
        centered.loc[valid] = valid_vals - valid_vals.groupby(group_key, sort=False).transform("mean")
    return centered.unstack(level=-1).reindex(index=alpha_panel.index, columns=alpha_panel.columns)
