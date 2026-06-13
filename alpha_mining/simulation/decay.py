from __future__ import annotations

import pandas as pd


def apply_decay(alpha_panel: pd.DataFrame, decay: int) -> pd.DataFrame:
    if decay <= 1:
        return alpha_panel
    return alpha_panel.ewm(span=int(decay), min_periods=1, adjust=False).mean()
