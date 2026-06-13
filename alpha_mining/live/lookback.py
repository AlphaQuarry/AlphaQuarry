from __future__ import annotations

import re
from typing import Any


KNOWN_TS_OPERATORS = {
    "ts_mean",
    "ts_rank",
    "ts_std",
    "ts_delta",
    "ts_backfill",
    "ts_sum",
    "ts_corr",
    "ts_covariance",
    "ts_min",
    "ts_max",
    "ts_zscore",
    "decay_linear",
}


def estimate_expression_lookback(expression: str, *, buffer: int = 5) -> dict[str, Any]:
    text = str(expression or "")
    warnings: list[str] = []
    max_arg = 0
    calls = _scan_function_calls(text)
    if not calls and ("ts_" in text or "decay_linear" in text):
        warnings.append("unable to parse expression function calls")
    for op, args in calls:
        if op.startswith("ts_") or op in {"decay_linear"}:
            nums = [int(x) for x in re.findall(r"(?<![A-Za-z_])(\d+)(?![A-Za-z_])", args)]
            if op not in KNOWN_TS_OPERATORS:
                warnings.append(f"unknown time-series operator lookback: {op}")
                continue
            if nums:
                max_arg = max(max_arg, max(nums))
            else:
                warnings.append(f"unable to parse lookback for operator: {op}")
    return {
        "max_lookback": int(max_arg + int(buffer)) if max_arg else int(buffer),
        "warnings": warnings,
    }


def _scan_function_calls(text: str) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if not (text[i].isalpha() or text[i] == "_"):
            i += 1
            continue
        start = i
        i += 1
        while i < n and (text[i].isalnum() or text[i] == "_"):
            i += 1
        name = text[start:i]
        j = i
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != "(":
            i = j
            continue
        depth = 0
        k = j
        while k < n:
            ch = text[k]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    args = text[j + 1 : k]
                    calls.append((name, args))
                    calls.extend(_scan_function_calls(args))
                    i = k + 1
                    break
            k += 1
        else:
            i = j + 1
    return calls


def estimate_snapshot_lookback(snapshot: dict[str, Any], *, buffer: int = 5) -> dict[str, Any]:
    max_lookback = 0
    warnings: list[str] = []
    for expr in snapshot.get("component_expressions") or [
        c.get("expression", "") for c in snapshot.get("components", [])
    ]:
        out = estimate_expression_lookback(str(expr), buffer=buffer)
        max_lookback = max(max_lookback, int(out["max_lookback"]))
        warnings.extend(out["warnings"])
    return {"max_lookback": max_lookback, "warnings": warnings}
