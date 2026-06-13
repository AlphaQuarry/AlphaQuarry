from __future__ import annotations

import pandas as pd

from alpha_mining.workflow.closed_loop import ClosedLoopConfig
from alpha_mining.workflow.closed_loop_config_summary import (
    closed_loop_config_hash,
    config_summary,
)


def test_closed_loop_config_summary_omits_heavy_benchmark_returns() -> None:
    cfg = ClosedLoopConfig(benchmark_returns=pd.Series([0.01, -0.02], name="bench"))

    summary = config_summary(cfg)

    assert "benchmark_returns" not in summary
    assert summary["universe_name"] == "cn_all"


def test_closed_loop_config_hash_is_stable_for_equivalent_configs() -> None:
    left = ClosedLoopConfig(benchmark_returns=pd.Series([0.01, -0.02], name="bench"))
    right = ClosedLoopConfig(benchmark_returns=pd.Series([0.03], name="bench"))

    assert closed_loop_config_hash(left) == closed_loop_config_hash(right)
