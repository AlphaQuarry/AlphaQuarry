from __future__ import annotations

import numpy as np
import pandas as pd

from factor_research import (
    TransactionCostConfig,
    build_portfolio_pnl_table,
    calculate_layer_portfolio_turnover,
)
from factor_research.single_factor import (
    _transaction_cost_for_date,
    _weight_turnover_components,
)


def test_transaction_cost_config_defaults_to_enabled_after_fee_pnl() -> None:
    assert TransactionCostConfig().enabled is True


def test_weight_turnover_components_respect_initial_position_flag() -> None:
    target = {"A": 0.5, "B": 0.5}

    no_initial = _weight_turnover_components(
        target_weights=target,
        prev_weights=None,
        prev_returns=None,
        charge_initial_position=False,
    )
    assert no_initial == {"turnover": 0.0, "buy_turnover": 0.0, "sell_turnover": 0.0}

    charged = _weight_turnover_components(
        target_weights=target,
        prev_weights=None,
        prev_returns=None,
        charge_initial_position=True,
    )
    assert np.isclose(charged["turnover"], 0.5)
    assert np.isclose(charged["buy_turnover"], 1.0)
    assert np.isclose(charged["sell_turnover"], 0.0)


def test_weight_turnover_components_split_buy_and_sell_after_drift() -> None:
    unchanged = _weight_turnover_components(
        target_weights={"A": 0.5, "B": 0.5},
        prev_weights={"A": 0.5, "B": 0.5},
        prev_returns={"A": 0.0, "B": 0.0},
    )
    assert np.isclose(unchanged["turnover"], 0.0)
    assert np.isclose(unchanged["buy_turnover"], 0.0)
    assert np.isclose(unchanged["sell_turnover"], 0.0)

    replaced = _weight_turnover_components(
        target_weights={"C": 1.0},
        prev_weights={"A": 1.0},
        prev_returns={"A": 0.0},
    )
    assert np.isclose(replaced["turnover"], 1.0)
    assert np.isclose(replaced["buy_turnover"], 1.0)
    assert np.isclose(replaced["sell_turnover"], 1.0)


def test_transaction_cost_uses_configurable_buy_and_sell_rates() -> None:
    cfg = TransactionCostConfig(
        enabled=True,
        commission_bps_per_side=1.0,
        slippage_bps_per_side=2.0,
        stamp_tax_bps_sell=5.0,
        transfer_fee_bps_per_side=0.0,
        exchange_fee_bps_per_side=0.0,
        regulatory_fee_bps_per_side=0.0,
    )

    cost = _transaction_cost_for_date(buy_turnover=0.25, sell_turnover=0.50, config=cfg)

    assert np.isclose(cost, 0.25 * 0.0003 + 0.50 * 0.0008)


def test_layer_portfolio_turnover_generates_fee_aware_layer_rows() -> None:
    layer_results = {
        "alpha_a": pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2025-01-01",
                        "2025-01-01",
                        "2025-01-01",
                        "2025-01-01",
                        "2025-01-02",
                        "2025-01-02",
                        "2025-01-02",
                        "2025-01-02",
                    ]
                ),
                "znz_code": ["A", "B", "C", "D", "A", "C", "B", "D"],
                "layer": [1, 1, 2, 2, 1, 1, 2, 2],
                "alpha_a": [0.1, 0.2, 0.8, 0.9, 0.1, 0.3, 0.7, 0.9],
                "pct_chg_1d": [0.01, 0.03, 0.05, 0.07, 0.02, 0.04, 0.06, 0.08],
            }
        )
    }
    cfg = TransactionCostConfig(enabled=True, commission_bps_per_side=10.0, stamp_tax_bps_sell=10.0)

    result = calculate_layer_portfolio_turnover(layer_results, transaction_cost_config=cfg)["alpha_a"]

    assert set(result["portfolio"]) == {"layer_1", "layer_2"}
    second_layer_1 = result[(result["portfolio"] == "layer_1")].sort_values("trade_date").iloc[1]
    assert np.isclose(float(second_layer_1["portfolio_return_layer"]), 0.03)
    assert float(second_layer_1["buy_turnover_layer"]) > 0
    assert float(second_layer_1["sell_turnover_layer"]) > 0
    assert float(second_layer_1["portfolio_return_layer_net"]) < float(second_layer_1["portfolio_return_layer"])


def test_portfolio_pnl_table_adds_gross_net_schema_without_net_for_long_short() -> None:
    dates = pd.to_datetime(["2025-01-01", "2025-01-02"])
    layer_visual = {
        "alpha_a": pd.DataFrame(
            {
                "trade_date": [dates[0], dates[0], dates[1], dates[1]],
                "layer": [1, "long_short", 1, "long_short"],
                "pct_chg_1d": [0.01, 0.02, 0.03, 0.04],
            }
        )
    }
    layer_turnover = {
        "alpha_a": pd.DataFrame(
            {
                "trade_date": dates,
                "factor": ["alpha_a", "alpha_a"],
                "layer": [1, 1],
                "portfolio": ["layer_1", "layer_1"],
                "holding_count_layer": [2, 2],
                "portfolio_return_layer": [0.01, 0.03],
                "turnover_layer": [0.0, 0.4],
                "buy_turnover_layer": [0.0, 0.4],
                "sell_turnover_layer": [0.0, 0.4],
                "transaction_cost_layer": [0.0, 0.001],
                "portfolio_return_layer_net": [0.01, 0.029],
            }
        )
    }

    pnl = build_portfolio_pnl_table(layer_visual, layer_turnover_results=layer_turnover)

    required = {
        "return_gross",
        "cum_return_gross",
        "transaction_cost",
        "return_net",
        "cum_return_net",
        "has_net_pnl",
        "cost_model",
        "buy_turnover",
        "sell_turnover",
    }
    assert required.issubset(set(pnl.columns))
    assert np.allclose(pnl["return"].to_numpy(dtype=float), pnl["return_gross"].to_numpy(dtype=float))
    assert np.allclose(
        pnl["cum_return"].to_numpy(dtype=float),
        pnl["cum_return_gross"].to_numpy(dtype=float),
    )
    layer_rows = pnl[pnl["portfolio"] == "layer_1"].sort_values("trade_date")
    assert layer_rows["has_net_pnl"].tolist() == [True, True]
    assert layer_rows["cum_return_net"].notna().all()
    long_short = pnl[pnl["portfolio"] == "long_short"].sort_values("trade_date")
    assert long_short["has_net_pnl"].tolist() == [False, False]
    assert long_short["cum_return_net"].isna().all()
