from __future__ import annotations

import pandas as pd


def apply_universe(alpha_panel: pd.DataFrame, universe_panel: pd.DataFrame | None) -> pd.DataFrame:
    """Mask alpha outside the external universe definition."""
    if universe_panel is None:
        return alpha_panel

    aligned = universe_panel.reindex(index=alpha_panel.index, columns=alpha_panel.columns)
    mask = aligned.fillna(False).astype(bool)
    return alpha_panel.where(mask)
