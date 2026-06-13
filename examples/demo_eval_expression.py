from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha_mining import ExpressionEngine, PanelStore


def main() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["A", "B", "A", "B"],
            "close": [10.0, 20.0, 11.0, 19.0],
            "volume": [100, 200, 120, 210],
        }
    )
    store = PanelStore.from_long_frame(df, date_col="date", code_col="code")
    engine = ExpressionEngine(panel_store=store)
    panel = engine.eval("rank(ts_delta(close, 1))")
    print(panel.tail())


if __name__ == "__main__":
    main()
