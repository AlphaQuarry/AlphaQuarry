from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PARQUET_COMPRESSION = os.environ.get("ALPHA_MINING_PARQUET_COMPRESSION", "zstd")


def write_parquet_compat(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    compression: str | None = DEFAULT_PARQUET_COMPRESSION,
    row_group_size: int | None = None,
    **kwargs: Any,
) -> None:
    target = Path(path)
    options: dict[str, Any] = dict(kwargs)
    if compression and str(compression).lower() not in {"none", "null", "false", "0"}:
        options["compression"] = compression
    if row_group_size is not None and int(row_group_size) > 0:
        options["row_group_size"] = int(row_group_size)
    try:
        df.to_parquet(target, index=index, **options)
    except TypeError:
        options.pop("row_group_size", None)
        try:
            df.to_parquet(target, index=index, **options)
        except Exception:
            options.pop("compression", None)
            df.to_parquet(target, index=index, **options)
    except Exception:
        options.pop("compression", None)
        options.pop("row_group_size", None)
        df.to_parquet(target, index=index, **options)
