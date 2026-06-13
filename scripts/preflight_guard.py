from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.api.preflight import run_preflight_checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight guard for local sensitive config and runtime sanity checks")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when risk checks fail")
    parser.add_argument("--config", default="configs/datasource.local.yaml")
    args = parser.parse_args()

    result = run_preflight_checks(config=str(args.config or "configs/datasource.local.yaml"), root=Path.cwd())
    infos = list(result.get("infos") or [])
    warnings = list(result.get("warnings") or [])
    remediations = list(result.get("remediations") or [])

    print("[preflight_guard] info:")
    for item in infos:
        print(f"  - {item}")

    print("[preflight_guard] warnings:")
    if not warnings:
        print("  - none")
    else:
        for item in warnings:
            print(f"  - {item}")
    if remediations:
        print("[preflight_guard] remediation:")
        for item in remediations:
            print(f"  - {item}")

    if bool(args.strict) and warnings:
        raise SystemExit(2)
    print("[preflight_guard] done")


if __name__ == "__main__":
    main()
