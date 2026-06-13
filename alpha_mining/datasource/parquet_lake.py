from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import LakePathSettings


P0_TABLES: tuple[str, ...] = (
    "trade_cal",
    "stock_basic",
    "index_classify",
    "index_member_all",
    "daily",
    "daily_basic",
    "adj_factor",
)

P1_TABLES: tuple[str, ...] = (
    "stk_limit",
    "suspend_d",
    "namechange",
)

P2_TABLES: tuple[str, ...] = (
    "income_vip",
    "balancesheet_vip",
    "cashflow_vip",
    "fina_indicator_vip",
)

P3_TABLES: tuple[str, ...] = (
    "moneyflow",
    "moneyflow_ths",
    "ths_index",
    "ths_member",
)


class ParquetLake:
    def __init__(self, paths: LakePathSettings):
        self.paths = paths
        self.ensure_layout()

    def ensure_layout(self) -> None:
        for path in [
            self.paths.lake_root_path,
            self.paths.vendor_raw_path,
            self.paths.curated_path,
            self.paths.snapshots_path,
            self.paths.meta_path,
            self.paths.duckdb_path_obj.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def vendor_table_root(self, table: str) -> Path:
        return self.paths.vendor_raw_path / str(table)

    def curated_table_root(self, table: str) -> Path:
        return self.paths.curated_path / str(table)

    def write_vendor_snapshot(
        self,
        table: str,
        snapshot_date: str,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        if df is None:
            df = pd.DataFrame()
        if len(df.columns) == 0:
            return {
                "path": "",
                "rows": 0,
                "table": str(table),
                "mode": "snapshot",
                "status": "skipped_empty_schema",
            }
        root = self.vendor_table_root(table) / f"snapshot_date={_normalize_date_key(snapshot_date)}"
        root.mkdir(parents=True, exist_ok=True)
        target = root / "part-000.parquet"
        _atomic_write_parquet(df, target)
        return {
            "path": str(target.as_posix()),
            "rows": int(len(df)),
            "table": str(table),
            "mode": "snapshot",
        }

    def write_curated_snapshot(
        self,
        table: str,
        snapshot_date: str,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        if df is None:
            df = pd.DataFrame()
        if len(df.columns) == 0:
            return {
                "path": "",
                "rows": 0,
                "table": str(table),
                "mode": "snapshot",
                "status": "skipped_empty_schema",
            }
        root = self.curated_table_root(table) / f"snapshot_date={_normalize_date_key(snapshot_date)}"
        root.mkdir(parents=True, exist_ok=True)
        target = root / "part-000.parquet"
        _atomic_write_parquet(df, target)
        return {
            "path": str(target.as_posix()),
            "rows": int(len(df)),
            "table": str(table),
            "mode": "snapshot",
        }

    def write_vendor_trade_table(
        self,
        table: str,
        df: pd.DataFrame,
        date_col: str = "trade_date",
        key_cols: Iterable[str] = ("ts_code", "trade_date"),
        mode: str = "upsert",
    ) -> dict[str, Any]:
        return self._write_partitioned_by_month(
            root=self.vendor_table_root(table),
            df=df,
            date_col=date_col,
            key_cols=tuple(key_cols),
            mode=mode,
        )

    def write_curated_trade_table(
        self,
        table: str,
        df: pd.DataFrame,
        date_col: str = "date",
        key_cols: Iterable[str] = ("code", "date"),
        mode: str = "upsert",
    ) -> dict[str, Any]:
        return self._write_partitioned_by_month(
            root=self.curated_table_root(table),
            df=df,
            date_col=date_col,
            key_cols=tuple(key_cols),
            mode=mode,
        )

    def read_vendor_table(
        self,
        table: str,
        start_date: str | None = None,
        end_date: str | None = None,
        date_col: str = "trade_date",
    ) -> pd.DataFrame:
        return self._read_partitioned(
            self.vendor_table_root(table),
            start_date=start_date,
            end_date=end_date,
            date_col=date_col,
        )

    def read_curated_table(
        self,
        table: str,
        start_date: str | None = None,
        end_date: str | None = None,
        date_col: str = "date",
    ) -> pd.DataFrame:
        return self._read_partitioned(
            self.curated_table_root(table),
            start_date=start_date,
            end_date=end_date,
            date_col=date_col,
        )

    def ingestion_state_path(self) -> Path:
        return self.paths.meta_path / "ingestion_state.json"

    def load_ingestion_state(self) -> dict[str, Any]:
        path = self.ingestion_state_path()
        if not path.exists():
            return {"tables": {}, "updated_at_utc": ""}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"tables": {}, "updated_at_utc": ""}
        payload.setdefault("tables", {})
        return payload

    def save_ingestion_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self.ingestion_state_path()
        out = dict(payload)
        out["updated_at_utc"] = _utc_now_iso()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return out

    def update_ingestion_state(
        self,
        table: str,
        last_trade_date: str,
        row_count: int,
        extra: dict[str, Any] | None = None,
        allow_rewind: bool = False,
    ) -> dict[str, Any]:
        payload = self.load_ingestion_state()
        tables = payload.setdefault("tables", {})
        table_key = str(table)
        prev_state = tables.get(table_key, {})
        prev_last_trade = _parse_trade_date(prev_state.get("last_trade_date", ""))
        incoming_last_trade = _parse_trade_date(last_trade_date)

        effective_last_trade = incoming_last_trade
        if bool(allow_rewind):
            if incoming_last_trade is None:
                effective_last_trade = prev_last_trade
        else:
            if prev_last_trade is not None and incoming_last_trade is not None:
                effective_last_trade = max(prev_last_trade, incoming_last_trade)
            elif prev_last_trade is not None and incoming_last_trade is None:
                effective_last_trade = prev_last_trade

        merged_extra: dict[str, Any] = {}
        if isinstance(prev_state, dict) and isinstance(prev_state.get("extra"), dict):
            merged_extra.update(prev_state.get("extra", {}))
        merged_extra.update(dict(extra or {}))

        if (
            not bool(allow_rewind)
            and prev_last_trade is not None
            and incoming_last_trade is not None
            and effective_last_trade == prev_last_trade
            and incoming_last_trade < prev_last_trade
        ):
            merged_extra["incoming_last_trade_date_ignored"] = str(_format_trade_date(incoming_last_trade))
        elif bool(allow_rewind):
            merged_extra.pop("incoming_last_trade_date_ignored", None)

        keep_prev_row_count = (
            not bool(allow_rewind)
            and prev_last_trade is not None
            and incoming_last_trade is not None
            and effective_last_trade == prev_last_trade
            and incoming_last_trade < prev_last_trade
            and isinstance(prev_state, dict)
            and str(prev_state.get("row_count", "")).strip() != ""
        )
        effective_row_count = int(prev_state.get("row_count", row_count)) if keep_prev_row_count else int(row_count)

        tables[table_key] = {
            "last_trade_date": _format_trade_date(effective_last_trade) if effective_last_trade is not None else "",
            "row_count": effective_row_count,
            "updated_at_utc": _utc_now_iso(),
            "extra": merged_extra,
        }
        return self.save_ingestion_state(payload)

    def infer_vendor_table_max_trade_date(
        self,
        table: str,
        date_col: str = "trade_date",
    ) -> str:
        return _infer_max_trade_date_from_table_root(
            root=self.vendor_table_root(table),
            preferred_date_col=str(date_col),
        )

    def infer_curated_table_max_trade_date(
        self,
        table: str,
        date_col: str = "date",
    ) -> str:
        return _infer_max_trade_date_from_table_root(
            root=self.curated_table_root(table),
            preferred_date_col=str(date_col),
        )

    def write_field_catalog(self, field_catalog_df: pd.DataFrame) -> str:
        target = self.paths.meta_path / "field_catalog.parquet"
        _atomic_write_parquet(field_catalog_df, target)
        return str(target.as_posix())

    def _write_partitioned_by_month(
        self,
        root: Path,
        df: pd.DataFrame,
        date_col: str,
        key_cols: tuple[str, ...],
        mode: str,
    ) -> dict[str, Any]:
        root.mkdir(parents=True, exist_ok=True)
        if df is None or df.empty:
            return {
                "rows": 0,
                "partitions": [],
                "mode": mode,
                "root": str(root.as_posix()),
            }
        if date_col not in df.columns:
            raise ValueError(f"Missing date column '{date_col}' for partitioned write")

        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work[work[date_col].notna()].copy()
        if work.empty:
            return {
                "rows": 0,
                "partitions": [],
                "mode": mode,
                "root": str(root.as_posix()),
            }

        work["_year"] = work[date_col].dt.year.astype(int)
        work["_month"] = work[date_col].dt.month.astype(int)

        mode_norm = str(mode or "upsert").strip().lower()
        if mode_norm not in {"upsert", "overwrite"}:
            raise ValueError(f"Unsupported write mode: {mode}")

        written: list[dict[str, Any]] = []
        for (year, month), part_df in work.groupby(["_year", "_month"], sort=True):
            partition_dir = root / f"year={int(year):04d}" / f"month={int(month):02d}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            target = partition_dir / "part-000.parquet"

            part_payload = part_df.drop(columns=["_year", "_month"]).copy()
            if mode_norm == "upsert" and target.exists():
                existing = pd.read_parquet(target)
                merged = pd.concat([existing, part_payload], ignore_index=True)
                dedupe_cols = [c for c in key_cols if c in merged.columns]
                if dedupe_cols:
                    merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
                else:
                    merged = merged.drop_duplicates(keep="last")
                part_payload = merged

            _atomic_write_parquet(part_payload, target)
            written.append(
                {
                    "year": int(year),
                    "month": int(month),
                    "rows": int(len(part_payload)),
                    "path": str(target.as_posix()),
                }
            )

        return {
            "rows": int(len(work)),
            "partitions": written,
            "mode": mode_norm,
            "root": str(root.as_posix()),
        }

    def _read_partitioned(
        self,
        root: Path,
        start_date: str | None,
        end_date: str | None,
        date_col: str,
    ) -> pd.DataFrame:
        if not root.exists():
            return pd.DataFrame()
        files = sorted(root.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        out = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        if date_col in out.columns:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
            if start_date:
                out = out[out[date_col] >= pd.to_datetime(start_date, errors="coerce")]
            if end_date:
                out = out[out[date_col] <= pd.to_datetime(end_date, errors="coerce")]
        return out.reset_index(drop=True)


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.parquet")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False)
    if target.exists():
        target.unlink()
    tmp.replace(target)


def _normalize_date_key(value: str) -> str:
    dt = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Invalid date key: {value}")
    return dt.strftime("%Y-%m-%d")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_trade_date(value: Any) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text:
        return None
    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt)


def _format_trade_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _infer_max_trade_date_from_table_root(
    root: Path,
    preferred_date_col: str,
) -> str:
    if not root.exists():
        return ""

    files = sorted(root.rglob("*.parquet"), reverse=True)
    if not files:
        return ""

    candidate_cols = [
        str(preferred_date_col),
        "trade_date",
        "date",
        "cal_date",
        "snapshot_date",
    ]
    seen_cols: set[str] = set()
    ordered_cols: list[str] = []
    for col in candidate_cols:
        key = str(col or "").strip()
        if not key or key in seen_cols:
            continue
        seen_cols.add(key)
        ordered_cols.append(key)

    for parquet_path in files:
        for col in ordered_cols:
            try:
                col_df = pd.read_parquet(parquet_path, columns=[col])
            except Exception:
                continue
            if col not in col_df.columns or col_df.empty:
                continue
            series = pd.to_datetime(col_df[col], errors="coerce")
            if series.notna().any():
                return pd.Timestamp(series.max()).strftime("%Y-%m-%d")
    return ""
