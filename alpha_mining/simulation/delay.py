from __future__ import annotations

import pandas as pd


def apply_delay(alpha_panel: pd.DataFrame, delay: int) -> pd.DataFrame:
    if delay <= 0:
        return alpha_panel
    return alpha_panel.shift(int(delay))
