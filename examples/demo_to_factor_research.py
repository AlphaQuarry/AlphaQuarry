from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha_mining.adapters import to_factor_research_frame


def main() -> None:
    raw_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "code": ["A", "B"],
            "pct_chg": [0.01, -0.02],
            "circ_mv": [1e9, 2e9],
        }
    )
    alpha_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "code": ["A", "B"],
            "alpha_0001": [0.1, -0.1],
        }
    )
    out = to_factor_research_frame(raw_df, alpha_df, code_col="code", date_col="date")
    print(out)


if __name__ == "__main__":
    main()
