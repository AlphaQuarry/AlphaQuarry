from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_mining.live.config import LiveConfig
from alpha_mining.live.orders import build_rebalance_orders, validate_orders_scope


def _cfg(tmp_path: Path) -> LiveConfig:
    cfg = LiveConfig(universe="u1", store_root=tmp_path)
    cfg.orders.board_lot_size = 100
    cfg.orders.min_order_value = 500.0
    cfg.orders.preserve_unsellable_positions = True
    cfg.orders.target_gross_exposure = 0.98
    return cfg


def test_orders_block_multi_active_without_single_sa_scope(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    result = validate_orders_scope(config=cfg, active_superalpha_ids=["sa1", "sa2"], requested_superalpha_id="all")

    assert result["status"] == "blocked"
    assert "orders_require_single_superalpha" in result["blocking_reasons"]


def test_target_and_positions_code_formats_are_normalized_before_join(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    target = pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "target_weight": [0.20],
            "close": [10.0],
            "can_buy": [1],
            "can_sell": [1],
        }
    )
    positions = pd.DataFrame(
        {
            "code": ["000001.XSHE"],
            "shares": [100],
            "available_shares": [100],
            "last_price": [10.0],
            "market_value": [1000.0],
        }
    )
    account = {
        "account_total_value": 10000.0,
        "cash": 9000.0,
        "position_market_value": 1000.0,
    }

    result = build_rebalance_orders(
        config=cfg,
        superalpha_id="sa1",
        target_holdings=target,
        positions=positions,
        account=account,
        execute_date="2026-05-26",
    )

    assert result["status"] == "ok"
    assert result["orders"]["code"].tolist() == ["000001.SZ"]


def test_orders_emit_weight_delta_only_without_money_basis(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    target = pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "target_weight": [0.10],
            "close": [10.0],
            "can_buy": [1],
            "can_sell": [1],
        }
    )
    positions = pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "shares": [100],
            "available_shares": [100],
            "last_price": [10.0],
        }
    )
    account = {
        "account_total_value": None,
        "cash": None,
        "position_market_value": 1000.0,
    }

    result = build_rebalance_orders(
        config=cfg,
        superalpha_id="sa1",
        target_holdings=target,
        positions=positions,
        account=account,
        execute_date="2026-05-26",
    )

    assert result["status"] == "ok"
    assert result["summary"]["orders_reviewable"] is False
    assert result["orders"]["delta_weight"].round(4).tolist() == [0.10]
    assert result["orders"]["order_value"].isna().all()
    assert result["orders"]["order_shares"].isna().all()


def test_orders_round_lots_cap_cash_and_preserve_unsellable_positions(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.orders.min_order_value = 100.0
    target = pd.DataFrame(
        {
            "code": ["000001.SZ", "000002.SZ"],
            "target_weight": [0.40, 0.30],
            "close": [10.0, 20.0],
            "can_buy": [1, 1],
            "can_sell": [1, 1],
        }
    )
    positions = pd.DataFrame(
        {
            "code": ["000001.SZ", "000003.SZ"],
            "shares": [100, 500],
            "available_shares": [100, 0],
            "last_price": [10.0, 5.0],
            "market_value": [1000.0, 2500.0],
        }
    )
    account = {
        "account_total_value": 10000.0,
        "cash": 6500.0,
        "position_market_value": 3500.0,
    }

    result = build_rebalance_orders(
        config=cfg,
        superalpha_id="sa1",
        target_holdings=target,
        positions=positions,
        account=account,
        execute_date="2026-05-26",
    )

    orders = result["orders"].sort_values("code").reset_index(drop=True)
    summary = result["summary"]
    blocked = orders[orders["code"] == "000003.SZ"].iloc[0]
    buy = orders[(orders["code"] == "000002.SZ") & (orders["side"] == "BUY")].iloc[0]
    assert blocked["blocked_reason"] == "blocked_sell:available_shares"
    assert buy["order_shares"] % 100 == 0
    assert summary["blocked_sell_value"] == 2500.0
    assert 0 < summary["scaled_buy_ratio"] <= 1
    assert summary["estimated_buy_value"] <= account["cash"] * (1 - cfg.portfolio.cash_buffer_ratio)
    assert summary["orders_reviewable"] is True


def test_sell_order_never_exceeds_available_shares_and_small_orders_filtered(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.orders.min_order_value = 2000.0
    target = pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "target_weight": [0.0],
            "close": [10.0],
            "can_buy": [1],
            "can_sell": [1],
        }
    )
    positions = pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "shares": [500],
            "available_shares": [120],
            "last_price": [10.0],
            "market_value": [5000.0],
        }
    )
    account = {
        "account_total_value": 10000.0,
        "cash": 5000.0,
        "position_market_value": 5000.0,
    }

    result = build_rebalance_orders(
        config=cfg,
        superalpha_id="sa1",
        target_holdings=target,
        positions=positions,
        account=account,
        execute_date="2026-05-26",
    )

    order = result["orders"].iloc[0]
    assert order["order_shares"] == 0
    assert order["blocked_reason"] == "small_order_filtered"
    assert result["summary"]["small_order_filtered_count"] == 1


def test_current_positions_blocked_by_market_tradability_are_preserved(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.orders.min_order_value = 100.0
    target = pd.DataFrame(
        {
            "code": ["000001.SZ"],
            "target_weight": [0.50],
            "close": [10.0],
            "can_buy": [1],
            "can_sell": [1],
        }
    )
    positions = pd.DataFrame(
        {
            "code": ["000003.SZ", "000004.SZ"],
            "shares": [200, 300],
            "available_shares": [200, 300],
            "last_price": [5.0, 6.0],
            "market_value": [1000.0, 1800.0],
        }
    )
    tradability = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-26", "2026-05-26"]),
            "code": ["000003.SZ", "000004.SZ"],
            "can_sell": [0, 1],
            "is_suspended": [0, 0],
            "is_limit_down_close": [0, 1],
        }
    )
    account = {
        "account_total_value": 10000.0,
        "cash": 7200.0,
        "position_market_value": 2800.0,
    }

    result = build_rebalance_orders(
        config=cfg,
        superalpha_id="sa1",
        target_holdings=target,
        positions=positions,
        account=account,
        execute_date="2026-05-26",
        position_tradability=tradability,
    )

    orders = result["orders"].set_index("code")
    assert orders.loc["000003.SZ", "side"] == "HOLD"
    assert orders.loc["000004.SZ", "side"] == "HOLD"
    assert orders.loc["000003.SZ", "blocked_reason"] == "blocked_sell:can_sell"
    assert orders.loc["000004.SZ", "blocked_reason"] == "blocked_sell:is_limit_down_close"
    assert result["summary"]["blocked_sell_count"] == 2
    assert result["summary"]["blocked_sell_value"] == 2800.0
