from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_mining.live.config import LiveConfig
from alpha_mining.live.parity import (
    compare_signal_cross_sections,
    evaluate_parity_thresholds,
)
from alpha_mining.workflow.superalpha import SUPERALPHA_FACTOR


def test_signal_parity_metrics_cover_rank_missing_and_overlap(tmp_path: Path) -> None:
    live = pd.DataFrame(
        {
            "date": ["2026-05-25"] * 5,
            "code": [f"00000{i}.SZ" for i in range(5)],
            SUPERALPHA_FACTOR: [5, 4, 3, 2, 1],
        }
    )
    ref = pd.DataFrame(
        {
            "date": ["2026-05-25"] * 5,
            "code": [f"00000{i}.SZ" for i in range(5)],
            SUPERALPHA_FACTOR: [50, 40, 30, 20, 10],
        }
    )

    metrics = compare_signal_cross_sections(live, ref, signal_date="2026-05-25", top_n=2)

    assert metrics["status"] == "ok"
    assert metrics["rank_correlation"] == 1.0
    assert metrics["valid_sample_count"] == 5
    assert metrics["missing_ratio_live"] == 0.0
    assert metrics["top_overlap"] == 1.0
    assert metrics["bottom_overlap"] == 1.0


def test_parity_thresholds_warn_or_block_by_strict_config(tmp_path: Path) -> None:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)
    cfg.parity.min_rank_corr = 0.98
    metrics = {
        "status": "ok",
        "rank_correlation": 0.5,
        "top_overlap": 0.9,
        "bottom_overlap": 0.9,
        "missing_ratio_delta": 0.0,
    }

    warning = evaluate_parity_thresholds(config=cfg, metrics=metrics)
    cfg.parity.strict = True
    blocked = evaluate_parity_thresholds(config=cfg, metrics=metrics)

    assert warning["status"] == "warning"
    assert blocked["status"] == "blocked"
    assert "rank_correlation_below_threshold" in blocked["reasons"]
