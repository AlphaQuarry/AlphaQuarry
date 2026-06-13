from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local factor dashboard API and static frontend.")
    parser.add_argument(
        "--store-root",
        default="data/alpha_universe_store",
        help="Alpha universe store root to scan.",
    )
    parser.add_argument(
        "--frontend-dist",
        default="dashboard/frontend/dist",
        help="Built Vite frontend dist directory.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload for dashboard API development.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["FACTOR_DASHBOARD_STORE_ROOT"] = str(args.store_root)
    os.environ["FACTOR_DASHBOARD_FRONTEND_DIST"] = str(args.frontend_dist)

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Missing dependency: install fastapi and uvicorn first.") from exc

    if args.reload:
        uvicorn.run(
            "dashboard.api.app:create_env_app",
            factory=True,
            host=args.host,
            port=int(args.port),
            reload=True,
        )
    else:
        from dashboard.api.app import create_app

        app = create_app(store_root=args.store_root, frontend_dist=args.frontend_dist)
        uvicorn.run(app, host=args.host, port=int(args.port))


if __name__ == "__main__":
    main()
