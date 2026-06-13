from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .account import normalize_stock_code
from .artifacts import live_paths, utc_now_iso, write_frame, write_json


def validate_orders_scope(
    *, config: Any, active_superalpha_ids: list[str], requested_superalpha_id: str
) -> dict[str, Any]:
    if not bool(config.orders.enabled):
        return {"status": "skipped", "reason": "orders_disabled"}
    if bool(config.orders.require_single_superalpha):
        requested = str(requested_superalpha_id or "").strip()
        if requested in {"", "all"} and len(active_superalpha_ids) != 1:
            return {
                "status": "blocked",
                "blocking_reasons": ["orders_require_single_superalpha"],
            }
    return {"status": "ok"}


def build_rebalance_orders(
    *,
    config: Any,
    superalpha_id: str,
    target_holdings: pd.DataFrame,
    positions: pd.DataFrame,
    account: dict[str, Any],
    execute_date: str,
    dry_run: bool = False,
    position_tradability: pd.DataFrame | None = None,
) -> dict[str, Any]:
    target = _targets(target_holdings)
    pos = _positions(positions)
    merged = pd.merge(target, pos, on="code", how="outer", suffixes=("", "_current"))
    tradability = _position_tradability(position_tradability)
    if not tradability.empty:
        merged = merged.merge(tradability, on="code", how="left")
    numeric_zero_cols = [
        "target_weight",
        "current_weight",
        "shares",
        "available_shares",
        "last_price",
        "market_value",
        "close",
        "target_shares_hint",
    ]
    for col in numeric_zero_cols:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    for col in [
        "position_can_sell",
        "position_is_suspended",
        "position_is_limit_down_close",
    ]:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["price"] = merged["close"].where(merged["close"] > 0, merged["last_price"])
    money_reliable = _is_money_reliable(account)
    total = float(account.get("account_total_value") or 0.0)
    cash = float(account.get("cash") or 0.0)
    target_gross = float(account.get("target_gross_exposure") or config.orders.target_gross_exposure)
    initial_desired_value = (
        merged["target_weight"] * total * target_gross
        if money_reliable
        else merged["target_shares_hint"] * merged["price"]
    )
    needs_sell = (merged["shares"] > 0) & (
        (merged["target_weight"] <= 0) | (merged["market_value"] > initial_desired_value + 1e-9)
    )
    blocked_reasons = _blocked_sell_reasons(merged)
    blocked_sell_mask = needs_sell & blocked_reasons.ne("") & bool(config.orders.preserve_unsellable_positions)
    blocked_sell_value = float(merged.loc[blocked_sell_mask, "market_value"].sum())
    investable_value = max(0.0, total * target_gross - blocked_sell_value) if money_reliable else 0.0
    desired_value = merged["target_weight"] * investable_value
    current_value = merged["market_value"]
    raw_delta = desired_value - current_value
    buy_mask = raw_delta > 0
    buy_budget = max(
        0.0,
        cash * (1.0 - float(account.get("cash_buffer_ratio") or config.portfolio.cash_buffer_ratio)),
    )
    raw_buy_sum = float(raw_delta[buy_mask].sum())
    scaled_buy_ratio = min(1.0, buy_budget / raw_buy_sum) if raw_buy_sum > 0 else 1.0
    adjusted_delta = raw_delta.copy()
    adjusted_delta.loc[buy_mask] = adjusted_delta.loc[buy_mask] * scaled_buy_ratio
    orders = merged[
        [
            "code",
            "target_weight",
            "current_weight",
            "shares",
            "available_shares",
            "price",
            "market_value",
        ]
    ].copy()
    orders["side"] = np.where(adjusted_delta > 0, "BUY", np.where(adjusted_delta < 0, "SELL", "HOLD"))
    current_weight_basis = orders["current_weight"] if money_reliable else 0.0
    orders["delta_weight"] = orders["target_weight"] - current_weight_basis
    orders["target_value"] = desired_value if money_reliable else np.nan
    orders["current_value"] = current_value if money_reliable else np.nan
    orders["order_value"] = adjusted_delta if money_reliable else np.nan
    orders["blocked_reason"] = ""
    orders.loc[blocked_sell_mask, "side"] = "HOLD"
    orders.loc[blocked_sell_mask, "order_value"] = 0.0 if money_reliable else np.nan
    orders.loc[blocked_sell_mask, "blocked_reason"] = blocked_reasons.loc[blocked_sell_mask]
    orders["order_shares"] = np.nan
    if money_reliable:
        orders = _round_and_filter(config, orders)
    summary = _summary(
        config,
        orders,
        account,
        money_reliable,
        blocked_sell_value,
        buy_budget,
        scaled_buy_ratio,
    )
    if dry_run:
        return {"status": "ok", "orders": orders, "summary": summary}
    paths = live_paths(config.store_root, config.universe)
    order_dir = paths.live_root / "orders" / str(superalpha_id)
    saved_parquet = write_frame(orders, order_dir / str(execute_date), preferred="parquet")
    saved_csv = write_frame(orders, order_dir / str(execute_date), preferred="csv")
    latest = {
        "schema_version": 1,
        "status": "ok",
        "superalpha_id": str(superalpha_id),
        "execute_date": str(execute_date),
        "row_count": int(len(orders)),
        "artifact_path": saved_parquet["path"],
        "orders_path": saved_parquet["path"],
        "orders_csv_path": saved_csv["path"],
        "account": {k: _json_scalar(v) for k, v in dict(account).items()},
        "summary": summary,
        "created_at_utc": utc_now_iso(),
    }
    write_json(order_dir / "latest.json", latest)
    return {**latest, "orders": orders, "summary": summary}


def _targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "code" in out.columns:
        out["code"] = out["code"].map(normalize_stock_code)
    if "close" not in out.columns:
        out["close"] = np.nan
    out["target_shares_hint"] = 0.0
    return out[
        ["code", "target_weight", "close", "can_buy", "can_sell", "target_shares_hint"]
        if {"can_buy", "can_sell"}.issubset(out.columns)
        else ["code", "target_weight", "close", "target_shares_hint"]
    ]


def _positions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "code",
                "shares",
                "available_shares",
                "last_price",
                "market_value",
                "current_weight",
            ]
        )
    out["code"] = out["code"].map(normalize_stock_code)
    out["market_value"] = pd.to_numeric(
        out.get("market_value", out.get("shares", 0) * out.get("last_price", 0)),
        errors="coerce",
    ).fillna(0.0)
    total = float(out["market_value"].sum())
    out["current_weight"] = out["market_value"] / total if total > 0 else 0.0
    return out[
        [
            "code",
            "shares",
            "available_shares",
            "last_price",
            "market_value",
            "current_weight",
        ]
    ]


def _position_tradability(frame: pd.DataFrame | None) -> pd.DataFrame:
    cols = [
        "code",
        "position_can_sell",
        "position_is_suspended",
        "position_is_limit_down_close",
        "tradability_date",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=cols)
    out = frame.copy()
    if "code" not in out.columns:
        return pd.DataFrame(columns=cols)
    out["code"] = out["code"].map(normalize_stock_code)
    rename = {
        "can_sell": "position_can_sell",
        "is_suspended": "position_is_suspended",
        "is_limit_down_close": "position_is_limit_down_close",
        "date": "tradability_date",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan if col != "code" else ""
    if "tradability_date" in out.columns:
        out["tradability_date"] = (
            pd.to_datetime(out["tradability_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        )
        out = out.sort_values(["code", "tradability_date"], kind="mergesort")
    return out[cols].drop_duplicates(subset=["code"], keep="last")


def _blocked_sell_reasons(frame: pd.DataFrame) -> pd.Series:
    reasons: list[str] = []
    for _, row in frame.iterrows():
        row_reasons: list[str] = []
        if float(row.get("available_shares") or 0.0) <= 0:
            row_reasons.append("available_shares")
        if (
            "position_can_sell" in frame.columns
            and pd.notna(row.get("position_can_sell"))
            and float(row.get("position_can_sell") or 0.0) <= 0
        ):
            row_reasons.append("can_sell")
        if (
            "position_is_suspended" in frame.columns
            and pd.notna(row.get("position_is_suspended"))
            and float(row.get("position_is_suspended") or 0.0) > 0
        ):
            row_reasons.append("is_suspended")
        if (
            "position_is_limit_down_close" in frame.columns
            and pd.notna(row.get("position_is_limit_down_close"))
            and float(row.get("position_is_limit_down_close") or 0.0) > 0
        ):
            row_reasons.append("is_limit_down_close")
        reasons.append("blocked_sell:" + ",".join(row_reasons) if row_reasons else "")
    return pd.Series(reasons, index=frame.index)


def _round_and_filter(config: Any, orders: pd.DataFrame) -> pd.DataFrame:
    out = orders.copy()
    lot = max(1, int(config.orders.board_lot_size))
    min_value = float(config.orders.min_order_value)
    for idx, row in out.iterrows():
        value = float(row.get("order_value") or 0.0)
        price = float(row.get("price") or 0.0)
        if row.get("blocked_reason"):
            out.at[idx, "order_shares"] = 0
            continue
        if price <= 0 or value == 0:
            out.at[idx, "order_shares"] = 0
            continue
        if value > 0:
            shares = int(value // price)
            if not bool(config.orders.allow_fractional_shares):
                shares = (shares // lot) * lot
            final_value = shares * price
            if final_value < min_value:
                out.at[idx, "order_shares"] = 0
                out.at[idx, "order_value"] = 0.0
                out.at[idx, "blocked_reason"] = "small_order_filtered"
            else:
                out.at[idx, "order_shares"] = shares
                out.at[idx, "order_value"] = final_value
        else:
            desired = int(abs(value) // price)
            max_sell = int(row.get("available_shares") or 0)
            shares = min(desired, max_sell)
            if not bool(config.orders.allow_odd_lot_sell):
                shares = (shares // lot) * lot
            final_value = shares * price
            if final_value < min_value:
                out.at[idx, "order_shares"] = 0
                out.at[idx, "order_value"] = 0.0
                out.at[idx, "blocked_reason"] = "small_order_filtered"
            else:
                out.at[idx, "order_shares"] = -shares
                out.at[idx, "order_value"] = -final_value
    return out


def _summary(
    config: Any,
    orders: pd.DataFrame,
    account: dict[str, Any],
    money_reliable: bool,
    blocked_sell_value: float,
    buy_budget: float,
    scaled_buy_ratio: float,
) -> dict[str, Any]:
    values = pd.to_numeric(orders.get("order_value"), errors="coerce").fillna(0.0)
    buys = values[values > 0]
    sells = -values[values < 0]
    buy_fee = (
        _trade_fee(
            float(buys.sum()),
            float(config.fee.buy_fee_bps),
            0.0,
            float(config.fee.min_commission),
        )
        if not buys.empty
        else 0.0
    )
    sell_fee = (
        _trade_fee(
            float(sells.sum()),
            float(config.fee.sell_fee_bps),
            float(config.fee.stamp_tax_bps),
            float(config.fee.min_commission),
        )
        if not sells.empty
        else 0.0
    )
    estimated_fee = buy_fee + sell_fee
    cash_before = float(account.get("cash") or 0.0) if money_reliable else None
    cash_after = (
        cash_before - float(buys.sum()) + float(sells.sum()) - estimated_fee if cash_before is not None else None
    )
    return {
        "orders_reviewable": bool(money_reliable),
        "review_block_reason": "" if money_reliable else "account_money_basis_missing",
        "estimated_turnover": float(float(buys.sum()) + float(sells.sum())),
        "estimated_buy_value": float(buys.sum()),
        "estimated_sell_value": float(sells.sum()),
        "estimated_fee": float(estimated_fee),
        "cash_before": cash_before,
        "cash_after_estimated": cash_after,
        "blocked_buy_count": int((orders.get("blocked_reason") == "blocked_buy").sum())
        if "blocked_reason" in orders
        else 0,
        "blocked_sell_count": int(
            orders.get("blocked_reason", pd.Series(dtype=object)).astype(str).str.startswith("blocked_sell").sum()
        )
        if "blocked_reason" in orders
        else 0,
        "small_order_filtered_count": int((orders.get("blocked_reason") == "small_order_filtered").sum())
        if "blocked_reason" in orders
        else 0,
        "blocked_sell_value": float(blocked_sell_value),
        "available_rebalance_cash": float(buy_budget),
        "scaled_buy_ratio": float(scaled_buy_ratio),
        "rounding_residual_cash": None if cash_after is None else float(cash_after),
    }


def _trade_fee(value: float, fee_bps: float, tax_bps: float, min_commission: float) -> float:
    if value <= 0:
        return 0.0
    commission = max(float(min_commission), value * float(fee_bps) / 10000.0)
    tax = value * float(tax_bps) / 10000.0
    return commission + tax


def _is_money_reliable(account: dict[str, Any]) -> bool:
    try:
        return float(account.get("account_total_value")) > 0 and float(account.get("cash")) >= 0
    except Exception:
        return False


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
