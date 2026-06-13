from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_mining.live.config import load_live_config
from alpha_mining.live.parity import run_live_signal_parity
from alpha_mining.live.registry import load_active_snapshots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a Live Superalpha windowed signal with its reference backtest signal."
    )
    parser.add_argument("--config", default="configs/live.example.yaml")
    parser.add_argument("--universe", default="")
    parser.add_argument("--superalpha-id", required=True)
    parser.add_argument("--date", required=True, help="signal_date, YYYY-MM-DD")
    parser.add_argument("--reference-path", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_live_config(args.config)
    if args.universe:
        cfg.universe = str(args.universe)
    if args.strict:
        cfg.parity.strict = True
    snapshots = load_active_snapshots(base_dir=cfg.store_root, universe=cfg.universe, superalpha_id=args.superalpha_id)
    if not snapshots:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blocking_reasons": ["active_superalpha_not_found"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    try:
        result = run_live_signal_parity(
            config=cfg,
            snapshot=snapshots[0],
            signal_date=str(args.date),
            reference_path=args.reference_path or None,
        )
    except Exception as exc:
        result = {"status": "blocked", "blocking_reasons": [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
