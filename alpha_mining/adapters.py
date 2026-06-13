from __future__ import annotations

import pandas as pd


def to_factor_research_frame(
    raw_df: pd.DataFrame,
    alpha_wide_df: pd.DataFrame,
    code_col: str = "code",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Merge raw fields and mined alpha columns into factor_research-ready frame.

    Output required keys:
    - trade_date
    - znz_code
    """
    if code_col not in raw_df.columns or date_col not in raw_df.columns:
        raise ValueError(f"raw_df must contain '{code_col}' and '{date_col}'")
    if code_col not in alpha_wide_df.columns or date_col not in alpha_wide_df.columns:
        raise ValueError(f"alpha_wide_df must contain '{code_col}' and '{date_col}'")

    base = raw_df.copy()
    alpha = alpha_wide_df.copy()
    base[date_col] = pd.to_datetime(base[date_col])
    alpha[date_col] = pd.to_datetime(alpha[date_col])

    merged = pd.merge(base, alpha, on=[date_col, code_col], how="left")
    merged = merged.rename(columns={date_col: "trade_date", code_col: "znz_code"})
    merged = merged.sort_values(["znz_code", "trade_date"]).reset_index(drop=True)
    return merged
