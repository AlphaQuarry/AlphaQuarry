from __future__ import annotations

from typing import Any


def load_trading_dates(*, config: Any, end_date: str | None = None) -> list[str]:
    import duckdb  # type: ignore

    date_col = config.data.date_col
    query = f'SELECT DISTINCT "{date_col}" AS d FROM "{config.source_view}"'
    params: list[Any] = []
    if end_date:
        query += f' WHERE "{date_col}" <= ?'
        params.append(str(end_date))
    query += " ORDER BY d"
    conn = duckdb.connect(str(config.duckdb_path), read_only=True)
    try:
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    return [str(r[0])[:10] for r in rows if r[0] is not None]


def previous_trading_window(*, config: Any, signal_date: str, window_days: int) -> tuple[str, list[str]]:
    dates = load_trading_dates(config=config, end_date=signal_date)
    dates = [d for d in dates if d <= str(signal_date)]
    if not dates:
        return str(signal_date), []
    size = max(1, int(window_days))
    window = dates[-size:]
    return window[0], window


def resolve_execute_date(*, config: Any, signal_date: str, requested_execute_date: str | None = None) -> dict[str, Any]:
    dates = load_trading_dates(config=config)
    signal = str(signal_date)
    later = [d for d in dates if d > signal]
    warnings: list[str] = []
    if requested_execute_date:
        req = str(requested_execute_date)
        valid = req in dates and req > signal
        if req <= signal:
            warnings.append("execute_date must be greater than signal_date")
        if req not in dates:
            warnings.append(f"execute_date is not a trading date: {req}")
        return {"execute_date": req, "valid": valid, "warnings": warnings}
    if not later:
        return {
            "execute_date": "",
            "valid": False,
            "warnings": ["no trading date after signal_date"],
        }
    return {"execute_date": later[0], "valid": True, "warnings": []}
