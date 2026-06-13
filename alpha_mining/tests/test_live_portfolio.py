from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_mining.live.config import LiveConfig
from alpha_mining.live.portfolio import build_target_holdings


def test_portfolio_blocks_when_signal_is_empty(tmp_path: Path) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)

    result = build_target_holdings(
        config=cfg,
        superalpha_id="superalpha_demo",
        signal=pd.DataFrame(columns=["date", "code", "superalpha"]),
        market=pd.DataFrame(),
        signal_date="2026-05-22",
        execute_date="2026-05-25",
    )

    assert result["status"] == "blocked"
    assert "signal_empty" in result["blocking_reasons"]


def test_portfolio_blocks_too_few_buyable_names(tmp_path: Path) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)
    cfg.portfolio.target_count = 5
    cfg.portfolio.min_target_count = 3
    cfg.portfolio.min_target_fill_ratio = 0.7
    signal = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-22"] * 5),
            "code": [f"00000{i}.SZ" for i in range(5)],
            "superalpha": [5, 4, 3, 2, 1],
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-22"] * 5),
            "code": [f"00000{i}.SZ" for i in range(5)],
            "close": [10.0] * 5,
            "circ_mv": [1000.0] * 5,
            "can_buy": [1, 0, 0, 0, 0],
            "can_sell": [1] * 5,
            "is_st": [0] * 5,
            "is_suspended": [0] * 5,
            "up_limit": [11.0] * 5,
            "down_limit": [9.0] * 5,
            "is_limit_up_close": [0] * 5,
            "is_limit_down_close": [0] * 5,
        }
    )

    result = build_target_holdings(
        config=cfg,
        superalpha_id="superalpha_demo",
        signal=signal,
        market=market,
        signal_date="2026-05-22",
        execute_date="2026-05-25",
    )

    assert result["status"] == "blocked"
    assert "too_few_buyable_names" in result["blocking_reasons"]
