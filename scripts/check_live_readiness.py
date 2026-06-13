from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.live.config import load_live_config
from alpha_mining.live.readiness import check_live_readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Live Superalpha readiness checklist.")
    parser.add_argument("--config", default="configs/live.example.yaml")
    parser.add_argument("--universe", default="")
    parser.add_argument("--superalpha-id", default="all")
    parser.add_argument("--date", default="")
    parser.add_argument("--position-path", default="")
    parser.add_argument("--account-total-value", type=float, default=None)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)

    cfg = load_live_config(args.config)
    if args.universe:
        cfg.universe = str(args.universe)
    overrides = {
        k: v
        for k, v in {
            "account_total_value": args.account_total_value,
            "cash": args.cash,
        }.items()
        if v is not None
    }
    result = check_live_readiness(
        config=cfg,
        requested_date=args.date or None,
        superalpha_id=args.superalpha_id,
        position_path=args.position_path or None,
        account_overrides=overrides,
        strict=bool(args.strict),
        json_out=args.json_out or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") == "BLOCKED" else 1 if result.get("status") == "WARN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
