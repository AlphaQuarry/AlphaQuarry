from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.datasource import load_datasource_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact monthly Parquet partitions in the local lake")
    parser.add_argument("--config", default="", help="Datasource config yaml path")
    parser.add_argument(
        "--roots",
        default="vendor,curated",
        help="Comma-separated roots to scan: vendor,curated",
    )
    parser.add_argument(
        "--tables",
        default="",
        help="Comma-separated table paths under selected roots. Empty scans all tables.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = load_datasource_settings(str(args.config or "") or None)
    roots = _parse_csv(str(args.roots or "vendor,curated"))
    tables = _parse_csv(str(args.tables or ""))
    summary = compact_lake(
        vendor_root=settings.paths.vendor_raw_path,
        curated_root=settings.paths.curated_path,
        roots=roots,
        tables=tables,
        dry_run=bool(args.dry_run),
    )
    print("[compact] done")
    print(summary)


def compact_lake(
    vendor_root: Path,
    curated_root: Path,
    roots: list[str] | tuple[str, ...] = ("vendor", "curated"),
    tables: list[str] | tuple[str, ...] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    root_map = {
        "vendor": Path(vendor_root),
        "curated": Path(curated_root),
    }
    selected_roots = [str(x).strip().lower() for x in roots if str(x).strip()]
    selected_tables = [str(x).strip().replace("\\", "/").strip("/") for x in tables if str(x).strip()]

    out: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "roots": {},
        "partitions_compacted": 0,
        "files_removed": 0,
    }
    for root_name in selected_roots:
        if root_name not in root_map:
            raise ValueError(f"Unsupported root: {root_name}")
        root = root_map[root_name]
        root_summary = compact_root(root=root, tables=selected_tables, dry_run=bool(dry_run))
        out["roots"][root_name] = root_summary
        out["partitions_compacted"] = int(out["partitions_compacted"]) + int(
            root_summary.get("partitions_compacted", 0)
        )
        out["files_removed"] = int(out["files_removed"]) + int(root_summary.get("files_removed", 0))
    return out


def compact_root(root: Path, tables: list[str] | tuple[str, ...] = (), dry_run: bool = False) -> dict[str, Any]:
    root = Path(root)
    if not root.exists():
        return {
            "root": str(root.as_posix()),
            "status": "missing",
            "partitions_scanned": 0,
            "partitions_compacted": 0,
            "files_removed": 0,
        }

    table_roots = _resolve_table_roots(root=root, tables=list(tables))
    summary: dict[str, Any] = {
        "root": str(root.as_posix()),
        "tables": {},
        "partitions_scanned": 0,
        "partitions_compacted": 0,
        "files_removed": 0,
    }
    for table_root in table_roots:
        table_key = table_root.relative_to(root).as_posix()
        table_summary = compact_table_root(table_root=table_root, dry_run=bool(dry_run))
        summary["tables"][table_key] = table_summary
        summary["partitions_scanned"] = int(summary["partitions_scanned"]) + int(
            table_summary.get("partitions_scanned", 0)
        )
        summary["partitions_compacted"] = int(summary["partitions_compacted"]) + int(
            table_summary.get("partitions_compacted", 0)
        )
        summary["files_removed"] = int(summary["files_removed"]) + int(table_summary.get("files_removed", 0))
    return summary


def compact_table_root(table_root: Path, dry_run: bool = False) -> dict[str, Any]:
    partitions = _find_month_partitions(table_root=Path(table_root))
    out = {
        "table_root": str(Path(table_root).as_posix()),
        "partitions_scanned": 0,
        "partitions_compacted": 0,
        "files_removed": 0,
        "files_remove_failed": 0,
    }
    for partition in partitions:
        files = sorted(partition.glob("*.parquet"))
        out["partitions_scanned"] += 1
        if len(files) <= 1:
            continue
        if not bool(dry_run):
            _merge_partition_files(files=files, target=partition / "part-000.parquet")
            for path in files:
                if path.name == "part-000.parquet":
                    continue
                if not _unlink_with_retries(path):
                    out["files_remove_failed"] += 1
        out["partitions_compacted"] += 1
        out["files_removed"] += max(0, len(files) - 1 - int(out["files_remove_failed"]))
    return out


def _resolve_table_roots(root: Path, tables: list[str]) -> list[Path]:
    if tables:
        return [root / table for table in tables if (root / table).exists()]
    out: list[Path] = []
    for path in sorted(root.rglob("year=*")):
        if not path.is_dir():
            continue
        parent = path.parent
        if parent not in out:
            out.append(parent)
    return out


def _find_month_partitions(table_root: Path) -> list[Path]:
    out: list[Path] = []
    for year_dir in sorted(table_root.glob("year=*")):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.glob("month=*")):
            if month_dir.is_dir():
                out.append(month_dir)
    return out


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        df.to_parquet(target, index=False)
        return
    tmp = target.with_suffix(".tmp.parquet")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False)
    tmp.replace(target)


def _merge_partition_files(files: list[Path], target: Path) -> None:
    try:
        payload = {
            "files": [str(path) for path in files],
            "target": str(target),
        }
        code = (
            "import duckdb,json,sys,pathlib;"
            "p=json.loads(sys.argv[1]);"
            "target=pathlib.Path(p['target']);"
            "tmp=target.with_suffix('.tmp.parquet');"
            "tmp.unlink(missing_ok=True);"
            "files=', '.join([repr(str(pathlib.Path(x)).replace(chr(92), '/')) for x in p['files']]);"
            "out=str(tmp).replace(chr(92), '/').replace(\"'\", \"''\");"
            "con=duckdb.connect();"
            "con.execute(f\"COPY (SELECT * FROM read_parquet([{files}], union_by_name=True)) TO '{out}' (FORMAT PARQUET)\");"
            "con.close();"
        )
        completed = subprocess.run([sys.executable, "-c", code, json.dumps(payload)], check=False)
        if int(completed.returncode) == 0:
            tmp = target.with_suffix(".tmp.parquet")
            if target.exists():
                _unlink_with_retries(target)
            tmp.replace(target)
            return
    except Exception:
        pass

    try:
        import duckdb  # type: ignore

        tmp = target.with_suffix(".tmp.parquet")
        if tmp.exists():
            tmp.unlink()
        files_sql = ", ".join("'" + str(path).replace("\\", "/").replace("'", "''") + "'" for path in files)
        conn = duckdb.connect()
        try:
            conn.execute(
                f"COPY (SELECT * FROM read_parquet([{files_sql}], union_by_name=True)) "
                f"TO '{str(tmp).replace(chr(92), '/').replace(chr(39), chr(39) + chr(39))}' (FORMAT PARQUET)"
            )
        finally:
            conn.close()
        if target.exists():
            _unlink_with_retries(target)
        tmp.replace(target)
        return
    except Exception:
        frames = [_read_parquet_detached(path) for path in files]
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        del frames
        gc.collect()
        _atomic_write_parquet(merged, target)
        del merged
        gc.collect()


def _read_parquet_detached(path: Path) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path, memory_map=False)
        out = table.to_pandas().copy(deep=True)
        del table
        gc.collect()
        return out
    except Exception:
        return pd.read_parquet(path)


def _unlink_with_retries(path: Path, attempts: int = 5) -> bool:
    for idx in range(max(1, int(attempts))):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            gc.collect()
            time.sleep(0.1 * (idx + 1))
    try:
        path.unlink(missing_ok=True)
        return True
    except PermissionError:
        return False


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


if __name__ == "__main__":
    main()
