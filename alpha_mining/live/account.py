from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_POSITION_COLUMNS = {"code", "shares", "available_shares"}
EXCHANGE_SUFFIX_MAP = {
    "XSHE": "SZ",
    "XSHG": "SH",
    "SZSE": "SZ",
    "SSE": "SH",
}


def normalize_stock_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        left, right = text.split(".", 1)
        suffix = EXCHANGE_SUFFIX_MAP.get(right, right)
        return f"{left.zfill(6)}.{suffix}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 6:
        digits = digits.zfill(6)
    suffix = "SH" if digits.startswith(("6", "9")) or digits.startswith("688") else "SZ"
    return f"{digits}.{suffix}"


def load_account_inputs(
    *,
    config: Any,
    execute_date: str,
    position_path: str | Path | None = None,
    account_snapshot_path: str | Path | None = None,
    account_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_path = position_path if position_path not in (None, "") else config.account.position_path
    if raw_path in (None, ""):
        return {
            "status": "skipped",
            "reason": "position_path_missing",
            "warnings": ["position_path_missing"],
            "blocking_reasons": [],
        }
    path = Path(raw_path)
    if not path.exists():
        return _blocked(["positions_file_missing"])
    if not path.is_file():
        return _blocked(["positions_file_not_file"])
    frame = pd.read_csv(path)
    validation = validate_positions_frame(frame, execute_date=execute_date, config=config)
    if validation["status"] == "blocked":
        return validation
    positions = validation["positions"]
    snapshot = _read_account_snapshot(account_snapshot_path or config.account.account_snapshot_path)
    account = _merge_account_basis(
        config=config,
        positions=positions,
        snapshot=snapshot,
        overrides=account_overrides or {},
    )
    money_reliable = _is_number(account.get("account_total_value")) and _is_number(account.get("cash"))
    return {
        "status": "ok",
        "positions": positions,
        "account": account,
        "money_reliable": bool(money_reliable),
        "warnings": validation.get("warnings", []),
        "blocking_reasons": [],
    }


def validate_positions_frame(frame: pd.DataFrame, *, execute_date: str, config: Any) -> dict[str, Any]:
    missing = sorted(REQUIRED_POSITION_COLUMNS - set(frame.columns))
    if "last_price" not in frame.columns and "market_value" not in frame.columns:
        missing.append("last_price_or_market_value")
    if "position_date" not in frame.columns and "updated_at" not in frame.columns:
        missing.append("position_date_or_updated_at")
    if missing:
        return _blocked([f"missing_position_column:{col}" for col in missing])
    out = frame.copy()
    out["code"] = out["code"].map(normalize_stock_code)
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0.0)
    out["available_shares"] = pd.to_numeric(out["available_shares"], errors="coerce").fillna(0.0)
    warnings: list[str] = []
    if "last_price" not in out.columns or pd.to_numeric(out.get("last_price"), errors="coerce").isna().any():
        if "market_value" in out.columns:
            mv = pd.to_numeric(out["market_value"], errors="coerce")
            shares = out["shares"].replace(0, pd.NA)
            out["last_price"] = (
                pd.to_numeric(out.get("last_price"), errors="coerce") if "last_price" in out.columns else pd.NA
            )
            out["last_price"] = out["last_price"].fillna(mv / shares)
            warnings.append("last_price_derived_from_market_value")
    if "market_value" not in out.columns:
        out["market_value"] = out["shares"] * pd.to_numeric(out["last_price"], errors="coerce")
    out["market_value"] = pd.to_numeric(out["market_value"], errors="coerce").fillna(0.0)
    date_col = "position_date" if "position_date" in out.columns else "updated_at"
    dates = pd.to_datetime(out[date_col], errors="coerce")
    max_stale = int(getattr(config.account, "max_position_staleness_trade_days", 1))
    if dates.isna().any() or _positions_are_stale_by_trading_days(
        config=config,
        position_dates=dates,
        execute_date=execute_date,
        max_stale=max_stale,
    ):
        reason = "stale_positions"
        if str(getattr(config.account, "stale_position_policy", "block")).lower() == "block":
            return _blocked([reason])
        warnings.append(reason)
    out["position_date"] = dates.dt.strftime("%Y-%m-%d")
    return {"status": "ok", "positions": out, "warnings": warnings}


def _read_account_snapshot(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    if p.suffix.lower() == ".json":
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    try:
        frame = pd.read_csv(p)
    except Exception:
        return {}
    return frame.iloc[0].to_dict() if not frame.empty else {}


def _merge_account_basis(
    *,
    config: Any,
    positions: pd.DataFrame,
    snapshot: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    out = {
        "account_id": getattr(config.account, "default_account_id", None),
        "account_total_value": None,
        "cash": None,
        "position_market_value": float(pd.to_numeric(positions.get("market_value"), errors="coerce").fillna(0.0).sum())
        if "market_value" in positions
        else None,
        "cash_buffer_ratio": float(config.portfolio.cash_buffer_ratio),
        "target_gross_exposure": float(config.orders.target_gross_exposure),
        "position_date": str(positions["position_date"].max())
        if "position_date" in positions.columns and not positions.empty
        else None,
    }
    for source in (snapshot, overrides):
        for key in (
            "account_id",
            "account_total_value",
            "cash",
            "position_market_value",
            "cash_buffer_ratio",
            "target_gross_exposure",
            "position_date",
        ):
            value = source.get(key)
            if value not in (None, ""):
                out[key] = value
    for key in (
        "account_total_value",
        "cash",
        "position_market_value",
        "cash_buffer_ratio",
        "target_gross_exposure",
    ):
        if out.get(key) not in (None, ""):
            try:
                out[key] = float(out[key])
            except Exception:
                out[key] = None
    return out


def _is_number(value: Any) -> bool:
    try:
        return pd.notna(float(value))
    except Exception:
        return False


def _positions_are_stale_by_trading_days(
    *, config: Any, position_dates: pd.Series, execute_date: str, max_stale: int
) -> bool:
    clean = pd.to_datetime(position_dates, errors="coerce").dropna()
    if clean.empty:
        return True
    execute_ts = pd.Timestamp(str(execute_date))
    for value in clean.dt.strftime("%Y-%m-%d"):
        distance = _trading_day_distance(config=config, position_date=str(value), execute_date=str(execute_date))
        if distance is None:
            return bool(((execute_ts - clean).dt.days > int(max_stale)).any())
        if distance > int(max_stale):
            return True
    return False


def _trading_day_distance(*, config: Any, position_date: str, execute_date: str) -> int | None:
    try:
        import duckdb  # type: ignore

        conn = duckdb.connect(str(config.duckdb_path), read_only=True)
        try:
            row = conn.execute(
                f'SELECT COUNT(DISTINCT "{config.data.date_col}") FROM "{config.source_view}" '
                f'WHERE "{config.data.date_col}" > ? AND "{config.data.date_col}" <= ?',
                [str(position_date), str(execute_date)],
            ).fetchone()
        finally:
            conn.close()
        count = int(row[0] or 0) if row else 0
        if count == 0 and str(position_date) < str(execute_date):
            return _business_day_distance(position_date=position_date, execute_date=execute_date)
        return count
    except Exception:
        return None


def _business_day_distance(*, position_date: str, execute_date: str) -> int:
    dates = pd.bdate_range(pd.Timestamp(position_date) + pd.Timedelta(days=1), pd.Timestamp(execute_date))
    return int(len(dates))


def _blocked(reasons: list[str]) -> dict[str, Any]:
    return {"status": "blocked", "blocking_reasons": reasons, "warnings": []}
