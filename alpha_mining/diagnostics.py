from __future__ import annotations

import numpy as np
import pandas as pd


def alpha_basic_diagnostics(alpha_panel: pd.DataFrame) -> dict[str, float]:
    """Compute cheap diagnostics for one date x code alpha panel."""
    total = alpha_panel.size
    non_na = int(alpha_panel.notna().sum().sum())
    coverage_rate = non_na / total if total else 0.0
    values = alpha_panel.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    cross_section_std_mean = float(alpha_panel.std(axis=1, skipna=True).mean()) if len(alpha_panel) else np.nan
    unique_ratio = float(len(np.unique(finite_values)) / len(finite_values)) if len(finite_values) else 0.0
    return {
        "coverage_rate": coverage_rate,
        "cross_section_std_mean": cross_section_std_mean,
        "unique_ratio": unique_ratio,
        "date_count": int(alpha_panel.shape[0]),
    }


def diagnostics_table(
    alpha_wide_df: pd.DataFrame, id_cols: tuple[str, str] = ("trade_date", "znz_code")
) -> pd.DataFrame:
    """Build diagnostics table for a wide alpha dataframe."""
    alpha_cols = [c for c in alpha_wide_df.columns if c not in set(id_cols)]
    rows: list[dict] = []
    if not alpha_cols:
        return pd.DataFrame(
            columns=[
                "alpha_name",
                "coverage_rate",
                "cross_section_std_mean",
                "unique_ratio",
                "date_count",
            ]
        )

    for col in alpha_cols:
        panel = alpha_wide_df.pivot(index=id_cols[0], columns=id_cols[1], values=col)
        stat = alpha_basic_diagnostics(panel)
        stat["alpha_name"] = col
        rows.append(stat)
    return pd.DataFrame(rows)
